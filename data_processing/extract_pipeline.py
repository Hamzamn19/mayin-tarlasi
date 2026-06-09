"""
Feature Extraction Pipeline for Aerial Landmine Detection
Extracts handcrafted features from LWIR thermal images using Pascal VOC annotations.
"""
import os
import sys
import glob
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.feature_functions import align_crop, robust_mask, compute_rts, compute_log_zero_crossings, compute_lbp_ri, compute_dct_high_energy, compute_wavelet_approx, compute_hole_count

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "landmine_flat")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "landmine_tabular_dataV3.csv")

# Mine class names in the dataset
MINE_CLASSES = {"ap_metal", "at_metal", "at_plastic", "ap_plastic"}

# Process ALL files; background will be balanced to match mine count
MAX_MINE_SAMPLES = None   # collect every annotated mine
MAX_BG_SAMPLES   = None   # set dynamically after pass 1

# ── Feature extraction helpers ─────────────────────────────────────────────

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
    """Extract 10 handcrafted features from a bounding-box crop.
    
    Features (10): area, circularity, thermal_contrast, aspect_ratio, rts,
                   log_zero_crossings, lbp_ri, dct_high_energy, wavelet_approx, hole_count
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return None

    # Convert crop to grayscale
    if crop.ndim == 3:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = crop.copy()
    gray_crop = align_crop(gray_crop)

    # 1. Object area (pixels)
    area = float(h * w)

    # 2. Circularity / compactness
    binary = robust_mask(gray_crop)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))

    # 3. Thermal contrast with background
    #    Background = proportional ring around the bounding box, clamped to image
    img_h, img_w = img_gray.shape[:2]
    pad = max(5, int(0.2 * max(h, w)))  # 20% of object size, min 5px
    bx1 = max(0, xmin - pad)
    by1 = max(0, ymin - pad)
    bx2 = min(img_w, xmax + pad)
    by2 = min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    mean_intensity = float(np.mean(gray_crop))
    # Mask out the object itself
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1 = ymin - by1
    ox1 = xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    if bg_pixels.size > 0:
        bg_mean = float(np.mean(bg_pixels))
    else:
        bg_mean = mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))

    # 4. Aspect ratio — mines are roughly circular (ratio ≈ 1.0)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # 5. Radial Thermal Symmetry — mines are circular, heat diffuses radially
    rts = compute_rts(gray_crop)

    # ── NEW FEATURES (6–10) ──────────────────────────────────────────────

    # 6. LoG zero-crossing count — captures edge transitions at multiple scales
    log_zero_crossings = compute_log_zero_crossings(gray_crop)

    # 7. LBP rotation-invariant — captures rotation-invariant texture patterns
    lbp_ri = compute_lbp_ri(gray_crop)

    # 8. DCT high-frequency energy ratio — captures sharpness / detail
    dct_high_energy = compute_dct_high_energy(gray_crop)

    # 9. Wavelet approximation coefficient energy — captures coarse structure
    wavelet_approx = compute_wavelet_approx(gray_crop)

    # 10. Hole count (binary blobs inside object) — captures internal voids
    hole_count = compute_hole_count(gray_crop)

    return {
        "area": area,
        "circularity": circularity,
        "thermal_contrast": thermal_contrast,
        "aspect_ratio": aspect_ratio,
        "rts": rts,
        "log_zero_crossings": log_zero_crossings,
        "lbp_ri": lbp_ri,
        "dct_high_energy": dct_high_energy,
        "wavelet_approx": wavelet_approx,
        "hole_count": hole_count,
    }


def parse_xml(xml_path):
    """Parse Pascal VOC XML and return list of (name, xmin, ymin, xmax, ymax)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            bb = obj.find("bndbox")
            xmin = int(float(bb.find("xmin").text))
            ymin = int(float(bb.find("ymin").text))
            xmax = int(float(bb.find("xmax").text))
            ymax = int(float(bb.find("ymax").text))
            objs.append((name, xmin, ymin, xmax, ymax))
        return objs
    except Exception:
        return []


# ── Collect all XML paths ──────────────────────────────────────────────────

print("Scanning for XML files…")
all_xml = glob.glob(os.path.join(BASE_DIR, "**", "*.xml"), recursive=True)
print(f"  Found {len(all_xml):,} XML files")

random.shuffle(all_xml)

# ══════════════════════════════════════════════════════════════════════════
# PASS 1 — Extract ALL mine samples
# ══════════════════════════════════════════════════════════════════════════
mine_records = []
mine_count = 0

MAX_XML = None  # set to int for testing limit, None = full run
all_xml = all_xml[:MAX_XML]

print("\nPass 1 — extracting all mine samples…")
for i, xml_path in enumerate(all_xml):
    img_path = xml_path.replace(".xml", ".jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape[:2]

    objects = parse_xml(xml_path)
    if not objects:
        continue

    for name, xmin, ymin, xmax, ymax in objects:
        if name not in MINE_CLASSES:
            continue
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = img[ymin:ymax, xmin:xmax]
        feats = extract_features(crop, img_gray, xmin, ymin, xmax, ymax)
        if feats:
            feats["label"] = 1
            feats["mine_type"] = name
            feats["source_file"] = os.path.basename(xml_path)
            feats["split"] = os.path.relpath(xml_path, BASE_DIR).split(os.sep)[0]
            mine_records.append(feats)
            mine_count += 1

    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1:,}/{len(all_xml):,} files | mines so far: {mine_count:,}")

print(f"\nPass 1 complete: {mine_count:,} mine samples collected")

# ══════════════════════════════════════════════════════════════════════════
# PASS 2 — Collect exactly mine_count background samples
# ══════════════════════════════════════════════════════════════════════════
TARGET_BG = mine_count
bg_records = []
bg_count = 0

print(f"\nPass 2 — extracting {TARGET_BG:,} background samples…")
random.shuffle(all_xml)   # re-shuffle for diversity

for i, xml_path in enumerate(all_xml):
    if bg_count >= TARGET_BG:
        break

    img_path = xml_path.replace(".xml", ".jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape[:2]

    objects = parse_xml(xml_path)
    if not objects:
        continue

    occupied = [(xmin, ymin, xmax, ymax) for _, xmin, ymin, xmax, ymax in objects]
    sizes = [(xmax - xmin, ymax - ymin) for _, xmin, ymin, xmax, ymax in objects]
    avg_w = max(8, int(np.mean([s[0] for s in sizes])))
    avg_h = max(8, int(np.mean([s[1] for s in sizes])))

    if img_w - avg_w < 1 or img_h - avg_h < 1:
        continue

    # Try up to 10 random placements per image to hit the target faster
    for _ in range(10):
        if bg_count >= TARGET_BG:
            break
        rx = random.randint(0, img_w - avg_w - 1)
        ry = random.randint(0, img_h - avg_h - 1)
        rx2, ry2 = rx + avg_w, ry + avg_h

        overlap = any(
            not (rx2 < xmin or rx > xmax or ry2 < ymin or ry > ymax)
            for xmin, ymin, xmax, ymax in occupied
        )
        if overlap:
            continue

        bg_crop = img[ry:ry2, rx:rx2]
        feats = extract_features(bg_crop, img_gray, rx, ry, rx2, ry2)
        if feats:
            feats["label"] = 0
            feats["mine_type"] = "background"
            feats["source_file"] = os.path.basename(xml_path)
            feats["split"] = os.path.relpath(xml_path, BASE_DIR).split(os.sep)[0]
            bg_records.append(feats)
            bg_count += 1

    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1:,}/{len(all_xml):,} files | backgrounds so far: {bg_count:,}")

print(f"\nPass 2 complete: {bg_count:,} background samples collected")

# ══════════════════════════════════════════════════════════════════════════
# Combine & save
# ══════════════════════════════════════════════════════════════════════════
records = mine_records + bg_records
mine_count = len(mine_records)
print(f"\nExtraction complete: {mine_count:,} mine samples | {bg_count:,} background samples")

df = pd.DataFrame(records)
df = df[["area", "circularity", "thermal_contrast", "aspect_ratio", "rts",
         "log_zero_crossings", "lbp_ri", "dct_high_energy", "wavelet_approx", "hole_count",
         "label", "mine_type", "split", "source_file"]]

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSplit distribution:")
print(df["split"].value_counts())
print("\nBasic stats:")
print(df[["area", "circularity", "thermal_contrast", "aspect_ratio", "rts"]].describe())
