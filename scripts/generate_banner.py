import cv2
import numpy as np
import os
import sys
import joblib
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.feature_functions import compute_rts, compute_log_zero_crossings, compute_lbp_ri, compute_dct_high_energy, compute_wavelet_approx, compute_hole_count

# Resolve paths
BASE_DIR = "/home/hamzah/Desktop/beykoz/proje/Machine Learning: Estimation and Prediction/MAYIN TARLASI"
MODEL_PATH = os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
RF_MODEL_PATH = os.path.join(BASE_DIR, "outputs/random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs/scaler.pkl")

# Load models
yolo_model = YOLO(MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = [
    'area', 'circularity', 'thermal_contrast', 'aspect_ratio', 'rts',
    'log_zero_crossings', 'lbp_ri', 'dct_high_energy', 'wavelet_approx', 'hole_count'
]
THRESHOLD = 0.4

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
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
        if perimeter > 0:
            circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))
            
    img_h, img_w = img_gray.shape[:2]
    pad = max(5, int(0.2 * max(h, w)))
    bx1, by1 = max(0, xmin - pad), max(0, ymin - pad)
    bx2, by2 = min(img_w, xmax + pad), min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1, ox1 = ymin - by1, xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else float(np.mean(gray_crop))
    thermal_contrast = float(abs(float(np.mean(gray_crop)) - bg_mean))
    
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    rts = compute_rts(gray_crop)

    log_zero_crossings = compute_log_zero_crossings(gray_crop)

    lbp_ri = compute_lbp_ri(gray_crop)

    dct_high_energy = compute_dct_high_energy(gray_crop)

    wavelet_approx = compute_wavelet_approx(gray_crop)

    hole_count = compute_hole_count(gray_crop)
    
    feats = [
        area, circularity, thermal_contrast, aspect_ratio, rts,
        log_zero_crossings, lbp_ri, dct_high_energy, wavelet_approx, hole_count
    ]
    return feats

def find_good_image():
    import xml.etree.ElementTree as ET
    image_dir = os.path.join(BASE_DIR, "landmine_flat/train")
    candidates = []
    
    for file in os.listdir(image_dir):
        if file.endswith(".xml") and "Noon" in file:
            xml_path = os.path.join(image_dir, file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                # Count mines by type
                at_count = 0
                ap_count = 0
                for obj in root.findall("object"):
                    name = obj.find("name").text.strip().lower()
                    if name in {"at_metal", "at_plastic"}:
                        at_count += 1
                    elif name in {"ap_metal", "ap_plastic"}:
                        ap_count += 1
                
                # Focus on images containing 2 to 4 AT mines (very clean, not crowded)
                if 2 <= at_count <= 4:
                    jpg_file = file.replace(".xml", ".jpg")
                    jpg_path = os.path.join(image_dir, jpg_file)
                    if os.path.exists(jpg_path):
                        img = cv2.imread(jpg_path)
                        if img is not None:
                            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mean_brightness = float(np.mean(img_gray))
                            candidates.append((mean_brightness, at_count, ap_count, jpg_path, jpg_file))
            except:
                continue
                
    if candidates:
        # Sort by thermal brightness descending to get maximum contrast
        candidates.sort(key=lambda x: x[0], reverse=True)
        print("Top AT mine candidates:")
        for idx, c in enumerate(candidates[:15]):
            print(f"[{idx}] {c[4]} - Brightness: {c[0]:.2f}, AT: {c[1]}, AP: {c[2]}")
        # Let's try candidate 1 (hottest with exactly 3 AT mines and 0 AP mines)
        best = candidates[1]
        print(f"Selected candidate: {best[4]} with {best[1]} AT and {best[2]} AP objects.")
        return best[3]
    return None

def main():
    img_path = find_good_image()
    if not img_path:
        print("No image with AT mines found in Noon, using fallback first image.")
        img_path = os.path.join(BASE_DIR, "landmine_flat/train/May_May_Noon_0_lwir_0_lwir_1.jpg")
        
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = yolo_model(img, conf=0.25, device='cpu')
    
    # Apply JET colormap to grayscale to get a vivid, high-contrast thermal look
    draw_img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]
        
        crop = img[y1:y2, x1:x2]
        feats = extract_features(crop, img_gray, x1, y1, x2, y2)
        if feats is None: continue
        feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
        scaled = scaler.transform(feats_df)
        rf_prob = float(rf_model.predict_proba(scaled)[0][1])
        
        # Dual-stage pipeline filtering: Only draw high-confidence detections
        if conf < 0.45 or rf_prob < 0.40:
            continue
            
        # Bounding box color: Neon green for military/detection display contrast
        box_color = (0, 255, 0) 
        
        # Draw bounding box
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), box_color, 2)
        
        # Format text label
        text = f"{label.upper()} (YOLO: {conf:.2f}, RF: {rf_prob:.2f})"
        
        # Get text width/height for background rectangle
        (w_t, h_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # Draw solid black background box for the label to ensure readability over colormap colors
        cv2.rectangle(draw_img, (x1, y1 - h_t - 6), (x1 + w_t + 10, y1), (0, 0, 0), -1)
        
        # Put text (white)
        cv2.putText(draw_img, text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        
    output_path = os.path.join(BASE_DIR, "outputs/detection_banner.jpg")
    cv2.imwrite(output_path, draw_img)
    print(f"Successfully generated detection banner image at: {output_path}")

if __name__ == "__main__":
    main()
