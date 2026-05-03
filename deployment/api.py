from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
import joblib
import pandas as pd
from PIL import Image
import io

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

# Load models
yolo_model = YOLO(MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH) if os.path.exists(LR_MODEL_PATH) else None
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None

def extract_features_from_crop(crop, img_gray, xmin, ymin, xmax, ymax):
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

    return {"area": area, "circularity": circularity, "mean_intensity": mean_intensity, 
            "thermal_contrast": thermal_contrast, "edge_density": edge_density}

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
                feature_names = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
                f_df = pd.DataFrame([[f['area'], f['circularity'], f['mean_intensity'], f['thermal_contrast'], f['edge_density']]], columns=feature_names)
                
                lr_prob = float(lr_model.predict_proba(f_df)[0][1]) if lr_model else 0.5
                rf_prob = float(rf_model.predict_proba(f_df)[0][1]) if rf_model else 0.5
                
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
