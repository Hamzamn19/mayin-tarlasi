from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

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
RF_MODEL_PATH = os.path.join(BASE_DIR, "outputs/random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs/scaler.pkl")

yolo_model = YOLO(MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = [
    'area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density',
    'intensity_std', 'aspect_ratio', 'thermal_gradient', 'max_min_ratio', 'relative_size'
]
THRESHOLD = 0.4

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
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
    
    feats = [
        area, circularity, mean_intensity, thermal_contrast, edge_density,
        intensity_std, aspect_ratio, thermal_gradient, max_min_ratio, relative_size
    ]
    
    features_dict = {
        'area': area,
        'circularity': circularity,
        'mean_intensity': mean_intensity,
        'thermal_contrast': thermal_contrast,
        'edge_density': edge_density,
        'intensity_std': intensity_std,
        'aspect_ratio': aspect_ratio,
        'thermal_gradient': thermal_gradient,
        'max_min_ratio': max_min_ratio,
        'relative_size': relative_size
    }
        
    return feats, features_dict

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = yolo_model(img, conf=0.25, device='cpu')
    
    detections = []
    h, w = img.shape[:2]
    
    idx = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            
            crop = img[y1:y2, x1:x2]
            feature_res = extract_features(crop, img_gray, x1, y1, x2, y2)
            
            if feature_res is None:
                continue
                
            feats, features_dict = feature_res
            feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
            scaled_feats = scaler.transform(feats_df)
            
            rf_prob = float(rf_model.predict_proba(scaled_feats)[0][1])
            
            ensemble_pred = 1 if rf_prob > THRESHOLD else 0
            
            detections.append({
                "id": idx,
                "x1": (x1 / w) * 100,
                "y1": (y1 / h) * 100,
                "x2": (x2 / w) * 100,
                "y2": (y2 / h) * 100,
                "conf": conf,
                "label": label,
                "rf_prob": rf_prob,
                "ensemble_pred": ensemble_pred,
                "features": features_dict
            })
            idx += 1
            
    return {"detections": detections}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
