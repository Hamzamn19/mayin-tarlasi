"""
Validation Script: Re-extract features using YOLOv8 model and compare with CSV data.
- Takes 100 random rows from landmine_tabular_data.csv
- Finds the corresponding images
- Uses YOLOv8 best.pt to detect objects
- Re-extracts the same 5 features (area, circularity, mean_intensity, thermal_contrast, edge_density)
- Compares with CSV values and reports differences
"""

import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR   = os.path.join(PROJECT_ROOT, "landmine_flat")
CSV_PATH   = os.path.join(PROJECT_ROOT, "outputs", "landmine_tabular_dataV3.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "runs", "detect", "Landmine_Detection_2026", "YOLO26_S_Standard", "weights", "best.pt")
REPORT_CSV = os.path.join(PROJECT_ROOT, "outputs", "validation_report.csv")

N_SAMPLES  = 100        # number of rows to validate
IOU_THRESH = 0.30       # IoU threshold to match YOLO box → CSV annotation
CONF_THRESH = 0.25      # YOLO confidence threshold

# ── Feature extraction (identical to original extract_pipeline.py) ─────────
def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return None

    if crop.ndim == 3:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = crop.copy()

    area = float(h * w)

    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))

    mean_intensity = float(np.mean(gray_crop))

    img_h, img_w = img_gray.shape[:2]
    pad = 5
    bx1 = max(0, xmin - pad); by1 = max(0, ymin - pad)
    bx2 = min(img_w, xmax + pad); by2 = min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1 = ymin - by1; ox1 = xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))

    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(np.sum(edges > 0)) / area if area > 0 else 0.0

    return dict(area=area, circularity=circularity, mean_intensity=mean_intensity,
                thermal_contrast=thermal_contrast, edge_density=edge_density)


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


# ── Build a lookup: source_file → image path ──────────────────────────────
print("Building image index…")
all_imgs = glob.glob(os.path.join(BASE_DIR, "**", "*.jpg"), recursive=True)
img_index = {}  # basename → full path
for p in all_imgs:
    key = os.path.basename(p).replace(".jpg", ".xml")
    img_index[key] = p

print(f"  Found {len(img_index):,} images")

# ── Load CSV — take only mine rows (label==1) for meaningful comparison ────
print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
mine_df = df[df["label"] == 1].copy()
print(f"  Total rows: {len(df):,}  |  Mine rows (label=1): {len(mine_df):,}")

# только те строки, у которых есть картинка
mine_df = mine_df[mine_df["source_file"].isin(img_index.keys())]
print(f"  Mine rows with found image: {len(mine_df):,}")

# Sample N_SAMPLES rows
sample_df = mine_df.sample(n=min(N_SAMPLES, len(mine_df)), random_state=42).reset_index(drop=True)
print(f"\nSampled {len(sample_df)} rows for validation")

# ── Load YOLO model ────────────────────────────────────────────────────────
print(f"\nLoading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to("cpu")
print("  Model loaded ✓ (running on CPU)")

# ── Process each sample ────────────────────────────────────────────────────
FEATURES = ["area", "circularity", "mean_intensity", "thermal_contrast", "edge_density"]

records = []
matched_count = 0
unmatched_count = 0

print("\nValidating samples…")
for idx, row in sample_df.iterrows():
    src_file = row["source_file"]
    img_path = img_index.get(src_file)
    if img_path is None:
        unmatched_count += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        unmatched_count += 1
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape[:2]

    # Run YOLO inference
    results = model(img_path, conf=CONF_THRESH, verbose=False)[0]

    # CSV row area → approximate original bounding box size
    csv_area = row["area"]
    # √area gives rough side length
    side = int(np.sqrt(csv_area))

    # Build reference area from CSV values to match YOLO boxes
    # We try to match every YOLO box whose area is close to csv_area
    best_feats = None
    best_iou_val = -1

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4] x1,y1,x2,y2

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            yolo_area = float((x2 - x1) * (y2 - y1))
            # Check area similarity (within 50%)
            area_ratio = yolo_area / csv_area if csv_area > 0 else 0
            if not (0.5 <= area_ratio <= 2.0):
                continue

            crop = img[y1:y2, x1:x2]
            feats = extract_features(crop, img_gray, x1, y1, x2, y2)
            if feats is None:
                continue

            # Score: inverse area difference (simple matching heuristic)
            area_diff = abs(feats["area"] - csv_area) / max(csv_area, 1)
            score = 1.0 - area_diff  # higher = better match

            if score > best_iou_val:
                best_iou_val = score
                best_feats = feats
                best_feats["yolo_box"] = f"[{x1},{y1},{x2},{y2}]"
                best_feats["match_score"] = score

    rec = {
        "sample_idx": idx,
        "source_file": src_file,
        "mine_type": row["mine_type"],
        "split": row["split"],
    }

    # CSV original values
    for f in FEATURES:
        rec[f"csv_{f}"] = row[f]

    if best_feats is not None:
        matched_count += 1
        for f in FEATURES:
            rec[f"yolo_{f}"] = best_feats.get(f, np.nan)
            rec[f"diff_{f}"] = abs(rec[f"csv_{f}"] - rec[f"yolo_{f}"])
            rec[f"pct_diff_{f}"] = 100.0 * rec[f"diff_{f}"] / (abs(rec[f"csv_{f}"]) + 1e-9)
        rec["yolo_box"] = best_feats.get("yolo_box", "")
        rec["match_score"] = best_feats.get("match_score", 0)
        rec["status"] = "matched"
    else:
        unmatched_count += 1
        for f in FEATURES:
            rec[f"yolo_{f}"] = np.nan
            rec[f"diff_{f}"] = np.nan
            rec[f"pct_diff_{f}"] = np.nan
        rec["yolo_box"] = ""
        rec["match_score"] = 0.0
        rec["status"] = "no_yolo_detection"

    records.append(rec)

    if (len(records)) % 10 == 0:
        print(f"  [{len(records)}/{len(sample_df)}] matched={matched_count}, unmatched={unmatched_count}")

# ── Summary ────────────────────────────────────────────────────────────────
report_df = pd.DataFrame(records)
report_df.to_csv(REPORT_CSV, index=False)

matched_df = report_df[report_df["status"] == "matched"]

print("\n" + "="*65)
print("  VALIDATION SUMMARY")
print("="*65)
print(f"  Total sampled rows   : {len(sample_df)}")
print(f"  YOLO matched         : {matched_count}")
print(f"  No YOLO detection    : {unmatched_count}")
print(f"  Match rate           : {100*matched_count/len(sample_df):.1f}%")

if len(matched_df) > 0:
    print("\n  Average absolute difference per feature (CSV vs YOLO):")
    print(f"  {'Feature':<22} {'Avg Diff':>12}  {'Avg % Diff':>12}")
    print("  " + "-"*48)
    for f in FEATURES:
        avg_diff     = matched_df[f"diff_{f}"].mean()
        avg_pct_diff = matched_df[f"pct_diff_{f}"].mean()
        print(f"  {f:<22} {avg_diff:>12.4f}  {avg_pct_diff:>11.2f}%")

    print("\n  Median absolute difference per feature:")
    print(f"  {'Feature':<22} {'Median Diff':>12}  {'Median % Diff':>14}")
    print("  " + "-"*50)
    for f in FEATURES:
        med_diff     = matched_df[f"diff_{f}"].median()
        med_pct_diff = matched_df[f"pct_diff_{f}"].median()
        print(f"  {f:<22} {med_diff:>12.4f}  {med_pct_diff:>13.2f}%")

    # Correlation between CSV and YOLO values
    print("\n  Pearson correlation (CSV vs YOLO values):")
    print(f"  {'Feature':<22} {'Corr R':>10}")
    print("  " + "-"*34)
    for f in FEATURES:
        col_csv  = matched_df[f"csv_{f}"].dropna()
        col_yolo = matched_df[f"yolo_{f}"].dropna()
        if len(col_csv) > 2 and len(col_yolo) > 2:
            r = np.corrcoef(col_csv, col_yolo)[0, 1]
            print(f"  {f:<22} {r:>10.4f}")

    # Verdict
    avg_pct_diffs = {f: matched_df[f"pct_diff_{f}"].mean() for f in FEATURES}
    overall_avg = np.mean(list(avg_pct_diffs.values()))
    print(f"\n  Overall average % difference: {overall_avg:.2f}%")
    if overall_avg < 5:
        verdict = "✅ Data looks CORRECT — differences are negligible (< 5%)"
    elif overall_avg < 15:
        verdict = "⚠️  Minor differences detected (5–15%) — likely bbox rounding"
    else:
        verdict = "❌ Significant differences detected (> 15%) — data may be inconsistent"
    print(f"\n  VERDICT: {verdict}")

print(f"\n  Detailed report saved → {REPORT_CSV}")
print("="*65)
