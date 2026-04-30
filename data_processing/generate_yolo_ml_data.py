import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
import random

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "landmine_flat")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "yolo_ml_training_data.csv")

# Constants
MINE_CLASSES = {"ap_metal", "at_metal", "at_plastic", "ap_plastic"}
IOU_THRESHOLD_POS = 0.5
IOU_THRESHOLD_NEG = 0.1

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
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
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
    bx1, by1 = max(0, xmin - pad), max(0, ymin - pad)
    bx2, by2 = min(img_w, xmax + pad), min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1, ox1 = ymin - by1, xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))
    
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(np.sum(edges > 0)) / area if area > 0 else 0.0
    
    return [area, circularity, mean_intensity, thermal_contrast, edge_density]

def parse_xml(xml_path):
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
            objs.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})
        return objs
    except: return []

def main():
    print("Loading YOLO model...")
    model = YOLO(YOLO_WEIGHTS)
    
    xml_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".xml"):
                xml_files.append(os.path.join(root, f))
    
    print(f"Found {len(xml_files)} XML files.")
    
    records = []
    
    for xml_path in tqdm(xml_files):
        img_path = xml_path.replace(".xml", ".jpg")
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img_gray.shape[:2]
        
        gt_objects = parse_xml(xml_path)
        
        # YOLO Inference
        results = model(img, verbose=False)[0]
        
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_w, xmax), min(img_h, ymax)
            
            if xmax <= xmin or ymax <= ymin: continue
            
            features = extract_features(img[ymin:ymax, xmin:xmax], img_gray, xmin, ymin, xmax, ymax)
            if not features: continue
            
            # Determine label based on GT
            label = 0 # Default to Background
            max_iou = 0
            for gt in gt_objects:
                iou = calculate_iou((xmin, ymin, xmax, ymax), gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
            
            if max_iou >= IOU_THRESHOLD_POS:
                label = 1 # Mine
            elif max_iou < IOU_THRESHOLD_NEG:
                label = 0 # Background (False Positive)
            else:
                continue # Ambiguous overlap, skip
                
            records.append(features + [label])

    df = pd.DataFrame(records, columns=["area", "circularity", "mean_intensity", "thermal_contrast", "edge_density", "label"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset generated with {len(df)} samples.")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
