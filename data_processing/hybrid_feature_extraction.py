import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "landmine_flat")
DATA_DIR = os.path.join(PROJECT_ROOT, "landmine_flat")
YOLO_LABELS_DIR = os.path.join(PROJECT_ROOT, "results", "runs", "detect", "Landmine_Detection_2026", "YOLO26_S_Standard", "labels")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "yolo_hybrid_features.csv")

# Constants
MINE_CLASSES = {"ap_metal", "at_metal", "at_plastic", "ap_plastic"}
IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min, yi_min = max(x1_min, x2_min), max(y1_min, y2_min)
    xi_max, yi_max = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
    """Extract 10 handcrafted features (matching extract_pipeline.py)"""
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    
    # 1. Area
    area = float(h * w)
    
    # 2. Circularity
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))
            
    # 3. Mean Intensity
    mean_intensity = float(np.mean(gray_crop))
    
    # 4. Thermal Contrast (proportional padding)
    img_h, img_w = img_gray.shape[:2]
    pad = max(5, int(0.2 * max(h, w)))
    bx1, by1 = max(0, xmin - pad), max(0, ymin - pad)
    bx2, by2 = min(img_w, xmax + pad), min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1, ox1 = ymin - by1, xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))
    
    # 5. Edge Density
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(np.sum(edges > 0)) / area if area > 0 else 0.0

    # 6. Intensity Std Dev
    intensity_std = float(np.std(gray_crop))

    # 7. Aspect Ratio
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # 8. Thermal Gradient Magnitude
    sobel_x = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    thermal_gradient = float(np.mean(gradient_mag))

    # 9. Max/Min Intensity Ratio
    min_val = float(np.min(gray_crop))
    max_val = float(np.max(gray_crop))
    max_min_ratio = max_val / (min_val + 1e-6)

    # 10. Relative Size
    image_area = float(img_h * img_w)
    relative_size = area / image_area if image_area > 0 else 0.0
    
    return [area, circularity, mean_intensity, thermal_contrast, edge_density,
            intensity_std, aspect_ratio, thermal_gradient, max_min_ratio, relative_size]

def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name in MINE_CLASSES:
                bb = obj.find("bndbox")
                xmin = int(float(bb.find("xmin").text))
                ymin = int(float(bb.find("ymin").text))
                xmax = int(float(bb.find("xmax").text))
                ymax = int(float(bb.find("ymax").text))
                objs.append((xmin, ymin, xmax, ymax))
        return objs
    except: return []

def main():
    label_files = [f for f in os.listdir(YOLO_LABELS_DIR) if f.endswith(".txt")]
    print(f"Extracting features from {len(label_files)} detected images...")
    
    records = []
    
    for label_file in tqdm(label_files):
        img_name = label_file.replace(".txt", ".jpg")
        xml_name = label_file.replace(".txt", ".xml")
        
        # Search for image and xml
        img_path = None
        xml_path = None
        for root, _, files in os.walk(DATA_DIR):
            if img_name in files: img_path = os.path.join(root, img_name)
            if xml_name in files: xml_path = os.path.join(root, xml_name)
            if img_path and xml_path: break
            
        if not img_path: continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = img.shape[:2]
        
        gt_boxes = parse_xml(xml_path) if xml_path else []
        
        with open(os.path.join(YOLO_LABELS_DIR, label_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id, x_c, y_c, w_n, h_n = map(float, parts[:5])
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                
                # Denormalize
                xmin = int((x_c - w_n/2) * w_orig)
                ymin = int((y_c - h_n/2) * h_orig)
                xmax = int((x_c + w_n/2) * w_orig)
                ymax = int((y_c + h_n/2) * h_orig)
                
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w_orig, xmax), min(h_orig, ymax)
                
                if xmax <= xmin or ymax <= ymin: continue
                
                features = extract_features(img[ymin:ymax, xmin:xmax], img_gray, xmin, ymin, xmax, ymax)
                if not features: continue
                
                # Ground Truth Matching
                label = 0
                for gt in gt_boxes:
                    if calculate_iou((xmin, ymin, xmax, ymax), gt) >= IOU_THRESHOLD:
                        label = 1
                        break
                
                records.append(features + [label, conf, img_name])

    df = pd.DataFrame(records, columns=[
        "area", "circularity", "mean_intensity", "thermal_contrast", "edge_density",
        "intensity_std", "aspect_ratio", "thermal_gradient", "max_min_ratio", "relative_size",
        "label", "confidence", "source_file"
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExtraction complete! Saved {len(df)} samples to {OUTPUT_CSV}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
