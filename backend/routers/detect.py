from fastapi import APIRouter, File, UploadFile, Request
from ultralytics import YOLO
import cv2
import numpy as np
import os

router = APIRouter()

@router.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Access the model from app state
    yolo_model = request.app.state.model
    results = yolo_model(img)
    
    detections = []
    h, w = img.shape[:2]
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            
            detections.append({
                "x1": (x1 / w) * 100,
                "y1": (y1 / h) * 100,
                "x2": (x2 / w) * 100,
                "y2": (y2 / h) * 100,
                "conf": conf,
                "label": label
            })
            
    return {"detections": detections}
