from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import joblib
import pandas as pd
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.feature_functions import align_crop, robust_mask, compute_rts, compute_log_zero_crossings, compute_lbp_ri, compute_dct_high_energy, compute_wavelet_approx, compute_hole_count

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
LR_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "logistic_regression_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.pkl")

# Load models
yolo_model = YOLO(MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH) if os.path.exists(LR_MODEL_PATH) else None
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

def extract_features_from_crop(crop, img_gray, xmin, ymin, xmax, ymax):
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray_crop = align_crop(gray_crop)
    
    # 1. Area
    area = float(h * w)
    
    # 2. Circularity
    binary = robust_mask(gray_crop)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0: circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))

    # 3. Thermal Contrast
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
    
    # 4. Aspect Ratio
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    
    # 5. Radial Thermal Symmetry
    rts = compute_rts(gray_crop)
    
    # 6. Log Zero Crossings
    log_zero_crossings = compute_log_zero_crossings(gray_crop)

    # 7. LBP Rotation Invariant
    lbp_ri = compute_lbp_ri(gray_crop)

    # 8. DCT High Energy
    dct_high_energy = compute_dct_high_energy(gray_crop)

    # 9. Wavelet Approximation
    wavelet_approx = compute_wavelet_approx(gray_crop)

    # 10. Hole Count
    hole_count = compute_hole_count(gray_crop)

    return {
        "area": area, "circularity": circularity,
        "thermal_contrast": thermal_contrast, "aspect_ratio": aspect_ratio,
        "rts": rts, "log_zero_crossings": log_zero_crossings,
        "lbp_ri": lbp_ri, "dct_high_energy": dct_high_energy,
        "wavelet_approx": wavelet_approx, "hole_count": hole_count
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = yolo_model(img)
    detections = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls = float(box.conf[0]), int(box.cls[0])
            label = yolo_model.names[cls]
            
            crop = img[y1:y2, x1:x2]
            f = extract_features_from_crop(crop, img_gray, x1, y1, x2, y2)
            
            if f:
                feature_names = [
                    'area', 'circularity', 'thermal_contrast', 'aspect_ratio', 'rts',
                    'log_zero_crossings', 'lbp_ri', 'dct_high_energy', 'wavelet_approx', 'hole_count'
                ]
                f_list = [f[name] for name in feature_names]
                f_df = pd.DataFrame([f_list], columns=feature_names)
                
                # Apply scaling
                if scaler:
                    f_scaled = scaler.transform(f_df)
                else:
                    f_scaled = f_df
                
                lr_prob = float(lr_model.predict_proba(f_scaled)[0][1]) if lr_model else 0.5
                rf_prob = float(rf_model.predict_proba(f_scaled)[0][1]) if rf_model else 0.5
                
                ensemble_prob = (conf + rf_prob) / 2.0
                ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
                
                # Convert to percentages for frontend
                h, w = img.shape[:2]
                detections.append({
                    "id": len(detections),
                    "x1": (x1 / w) * 100,
                    "y1": (y1 / h) * 100,
                    "x2": (x2 / w) * 100,
                    "y2": (y2 / h) * 100,
                    "conf": conf,
                    "label": label,
                    "lr_prob": lr_prob,
                    "rf_prob": rf_prob,
                    "ensemble_prob": ensemble_prob,
                    "ensemble_pred": ensemble_pred,
                    "features": f
                })
    
    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
