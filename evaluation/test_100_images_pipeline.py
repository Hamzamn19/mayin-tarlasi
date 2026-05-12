import os
import cv2
import glob
import random
import torch
import joblib
import pandas as pd
import numpy as np
import warnings
from ultralytics import YOLO

# Suppress warnings to keep the console clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Paths
BASE_DIR = "/home/hamzah/Desktop/beykoz/proje/Machine Learning: Estimation and Prediction/MAYIN TARLASI"
YOLO_WEIGHTS = os.path.join(BASE_DIR, "results", "runs", "detect", "Landmine_Detection_2026", "YOLO26_S_Standard", "weights", "best.pt")

XGB_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "xgboost_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.pkl")

# Feature names used during training
FEATURE_NAMES = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density', 'confidence']

# Fallback: retrain if models don't exist
if not os.path.exists(XGB_MODEL_PATH):
    print("ML models not found! Retraining quickly...")
    os.system(f"python3 {os.path.join(BASE_DIR, 'machine_learning', 'train_ml_models.py')}")

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax, conf=None):
    """Extract features matching our latest 6-feature pipeline"""
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    area = float(h * w)
    
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0: circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))
            
    mean_intensity = float(np.mean(gray_crop))
    
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
    
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(np.sum(edges > 0)) / area if area > 0 else 0.0

    feats = [area, circularity, mean_intensity, thermal_contrast, edge_density]
    if conf is not None:
        feats.append(float(conf))
    return feats

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min, yi_min = max(x1_min, x2_min), max(y1_min, y2_min)
    xi_max, yi_max = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    union_area = (x1_max - x1_min)*(y1_max - y1_min) + (x2_max - x2_min)*(y2_max - y2_min) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main():
    FINAL_TEST_DIR = "/home/hamzah/Desktop/beykoz/proje/Machine Learning: Estimation and Prediction/MAYIN TARLASI/landmine_final - Copy/elevation_test"
    print(f"1. Loading images from: {FINAL_TEST_DIR}")
    
    test_images = glob.glob(os.path.join(FINAL_TEST_DIR, "**", "*.jpg"), recursive=True)
    print(f"Total images found: {len(test_images)}")

    if not test_images:
        print("ERROR: No images found!")
        return
        
    # Limit to 100 random images for quick testing
    if len(test_images) > 100:
        random.seed(42)
        test_images = random.sample(test_images, 100)
    
    print(f"Selected {len(test_images)} images for quick testing.")

    print("Extracting ground truth from XML files...")
    total_actual_mines = 0
    img_gt_boxes = {}
    for img_path in test_images:
        xml_path = img_path.replace('.jpg', '.xml')
        gt_boxes = []
        if os.path.exists(xml_path):
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name in {"ap_metal", "at_metal", "at_plastic", "ap_plastic"}:
                    bb = obj.find("bndbox")
                    gt_boxes.append((int(float(bb.find("xmin").text)), int(float(bb.find("ymin").text)),
                                     int(float(bb.find("xmax").text)), int(float(bb.find("ymax").text))))
        total_actual_mines += len(gt_boxes)
        img_gt_boxes[img_path] = gt_boxes

    print(f"Total Actual Mines: {total_actual_mines}")

    print("\n2. Loading YOLO26 and ML Models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.to(device)
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("\n3. Running Full Inference Pipeline (Threshold: 0.4)...")
    stats = {
        "yolo": {"det": 0, "tp": 0},
        "rf":   {"det": 0, "tp": 0},
        "xgb":  {"det": 0, "tp": 0},
        "soft": {"det": 0, "tp": 0}
    }
    THRESHOLD = 0.4

    for i, img_path in enumerate(test_images):
        img = cv2.imread(img_path)
        if img is None: continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_boxes = img_gt_boxes[img_path]

        results = yolo_model(img_path, conf=0.25, device=device, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            stats["yolo"]["det"] += 1
            
            crop = img[y1:y2, x1:x2]
            feats = extract_features(crop, img_gray, x1, y1, x2, y2, conf=conf)
            if not feats: continue
                
            # ML Probabilities
            feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
            scaled_feats = scaler.transform(feats_df)
            rf_prob = rf_model.predict_proba(scaled_feats)[0][1]
            xgb_prob = xgb_model.predict_proba(scaled_feats)[0][1]

            soft_prob = (0.7 * rf_prob) + (0.3 * xgb_prob)
            
            is_rf = rf_prob > THRESHOLD
            is_xgb = xgb_prob > THRESHOLD
            is_soft = soft_prob > THRESHOLD

            if is_rf: stats["rf"]["det"] += 1
            if is_xgb: stats["xgb"]["det"] += 1
            if is_soft: stats["soft"]["det"] += 1

            is_correct = any(calculate_iou((x1, y1, x2, y2), gt) >= 0.5 for gt in gt_boxes)
            
            if is_correct:
                stats["yolo"]["tp"] += 1
                if is_rf: stats["rf"]["tp"] += 1
                if is_xgb: stats["xgb"]["tp"] += 1
                if is_soft: stats["soft"]["tp"] += 1

        if (i+1) % 100 == 0 or (i+1) == len(test_images):
            print(f"Processed {i+1}/{len(test_images)} images...")

    print("\n" + "="*50)
    print(f"📊 FINAL FULL EVALUATION ({len(test_images)} IMAGES)")
    print("="*50)
    
    def print_final(name, s):
        tp, det = s["tp"], s["det"]
        prec = (tp / det * 100) if det > 0 else 0
        rec = (tp / total_actual_mines * 100) if total_actual_mines > 0 else 0
        err = ((det - tp + (total_actual_mines - tp)) / total_actual_mines * 100) if total_actual_mines > 0 else 0
        print(f"[{name:25}] Prec: {prec:6.2f}% | Rec: {rec:6.2f}% | Error: {err:6.2f}%")

    print_final("1. YOLO26 Only", stats["yolo"])
    print_final("2. YOLO26 + Random Forest", stats["rf"])
    print_final("3. YOLO26 + XGBoost", stats["xgb"])
    print_final("4. YOLO26 + Hybrid (Soft)", stats["soft"])
    print("="*50)

if __name__ == "__main__":
    main()
