import cv2, numpy as np, os, sys, glob, json, time, xml.etree.ElementTree as ET
import pandas as pd
from ultralytics import YOLO
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from data_processing.feature_functions import align_crop, robust_mask, compute_rts, compute_log_zero_crossings, compute_lbp_ri, compute_dct_high_energy, compute_wavelet_approx, compute_hole_count

# Load models
yolo = YOLO(os.path.join(BASE, 'results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt'))
rf = joblib.load(os.path.join(BASE, 'outputs/random_forest_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'outputs/scaler.pkl'))

feature_names = ['area', 'circularity', 'thermal_contrast', 'aspect_ratio', 'rts',
                 'log_zero_crossings', 'lbp_ri', 'dct_high_energy', 'wavelet_approx', 'hole_count']
MINE_CLASSES = {'ap_metal', 'at_metal', 'at_plastic', 'ap_plastic'}
IOU_THRESH = 0.5

def extract_features(crop, img_gray, xmin, ymin, xmax, ymax):
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray_crop = align_crop(gray_crop)
    area = float(h * w)
    binary = robust_mask(gray_crop)
    cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if cs:
        c = max(cs, key=cv2.contourArea)
        ca = cv2.contourArea(c); per = cv2.arcLength(c, True)
        if per > 0: circularity = float(4 * np.pi * ca / (per ** 2))
    img_h, img_w = img_gray.shape[:2]
    pad = max(5, int(0.2 * max(h, w)))
    bx1, by1 = max(0, xmin - pad), max(0, ymin - pad)
    bx2, by2 = min(img_w, xmax + pad), min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    mean_intensity = float(np.mean(gray_crop))
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1, ox1 = ymin - by1, xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    rts = compute_rts(gray_crop)
    log_zc = compute_log_zero_crossings(gray_crop)
    lbp_ri_v = compute_lbp_ri(gray_crop)
    dct_he = compute_dct_high_energy(gray_crop)
    wavelet = compute_wavelet_approx(gray_crop)
    holes = compute_hole_count(gray_crop)
    return [area, circularity, thermal_contrast, aspect_ratio, rts, log_zc, lbp_ri_v, dct_he, wavelet, holes]

def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

# Process test images
test_dir = os.path.join(BASE, 'landmine_flat/test')
test_jpgs = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
print(f'Processing {len(test_jpgs)} test images...')

records = []
t0 = time.time()
for idx, jpg_path in enumerate(test_jpgs):
    xml_path = jpg_path.replace('.jpg', '.xml')
    basename = os.path.basename(jpg_path)
    
    img = cv2.imread(jpg_path)
    if img is None: continue
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img.shape[:2]
    
    # Ground truth
    gt_boxes = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path); root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name in MINE_CLASSES:
                bb = obj.find('bndbox')
                gt_boxes.append((int(float(bb.find('xmin').text)), int(float(bb.find('ymin').text)),
                                 int(float(bb.find('xmax').text)), int(float(bb.find('ymax').text))))
    
    # YOLO inference
    results = yolo(img, conf=0.25, device=0, verbose=False)
    if not results: continue
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        if x2 <= x1 or y2 <= y1: continue
        
        feats = extract_features(img[y1:y2, x1:x2], img_gray, x1, y1, x2, y2)
        if not feats: continue
        
        # Match to ground truth
        label = 0
        for gt in gt_boxes:
            if iou((x1, y1, x2, y2), gt) >= IOU_THRESH:
                label = 1
                break
        
        records.append(feats + [label, conf, basename])
    
    if (idx + 1) % 200 == 0:
        print(f'  {idx+1}/{len(test_jpgs)} images in {time.time()-t0:.0f}s')

print(f'Done. {len(records)} detections from {len(test_jpgs)} images in {time.time()-t0:.0f}s')

# Save
df = pd.DataFrame(records, columns=feature_names + ['label', 'confidence', 'source_file'])
out_path = os.path.join(BASE, 'outputs', 'yolo_hybrid_features.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(f'Class distribution: {df.label.value_counts().to_dict()}')

# Evaluate
print('\n' + '='*60)
print('TABLE III — MODEL COMPARISON ON TEST SET')
print('='*60)

# YOLO only
yolo_pred = df['confidence'].values
yolo_label = (yolo_pred >= 0.5).astype(int)
y_true = df['label'].values

# RF only
X = scaler.transform(df[feature_names])
rf_pred = rf.predict(X)

# YOLO+RF (ensemble: average confidence)
ensemble_prob = (yolo_pred + rf.predict_proba(X)[:, 1]) / 2.0
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

models = [
    ('YOLO only', yolo_label),
    ('RF only', rf_pred),
    ('YOLO+RF (avg)', ensemble_pred),
]

print(f'{"Model":<20} {"Acc":>8} {"F1":>8} {"Prec":>8} {"Rec":>8}')
print('-'*52)
for name, pred in models:
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    print(f'{name:<20} {acc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}')
