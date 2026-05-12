import os
import glob
import time
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# Resolve absolute paths based on project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAT_DATA_DIR = os.path.join(BASE_DIR, "landmine_flat")

MODELS = {
    "YOLO26": os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
}

CONF_THRESH = 0.25
BATCH_SIZE = 32  # Increased batch size for T4 GPU

def get_actual_count(img_path):
    """Reads the corresponding XML file to count the actual number of mines."""
    xml_path = img_path.replace('.jpg', '.xml')
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return len(root.findall('object'))
        except Exception:
            pass
    return 0

import torch

def evaluate_model(model_name, model_path, images):
    print(f"\n[{model_name}] Loading model from {model_path} ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{model_name}] Using device: {device.upper()}")
    
    try:
        model = YOLO(model_path, task='detect')
        if model_path.endswith('.pt'):
            model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    total_actual = 0
    total_predicted = 0
    exact_matches = 0
    
    # Pre-calculate actual counts from XML
    print(f"[{model_name}] Parsing ground truth XMLs to find actual mine counts...")
    actual_counts = [get_actual_count(img) for img in images]
    
    print(f"[{model_name}] Warming up...")
    model.predict(source=images[:4], conf=CONF_THRESH, device=device, verbose=False)
    
    print(f"[{model_name}] Starting batched inference on {len(images)} images using {device.upper()}...")
    start_time = time.time()
    
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i+BATCH_SIZE]
        batch_actuals = actual_counts[i:i+BATCH_SIZE]
        
        # Generator for efficient batched processing
        results = model.predict(source=batch_imgs, conf=CONF_THRESH, device=device, verbose=False, stream=True)
        
        for j, res in enumerate(results):
            actual_count = batch_actuals[j]
            predicted_count = len(res.boxes)
            
            total_actual += actual_count
            total_predicted += predicted_count
            if actual_count == predicted_count:
                exact_matches += 1
                
        if (i + BATCH_SIZE) % 1000 == 0 or (i + BATCH_SIZE) >= len(images):
            print(f"  -> Processed {min(i + BATCH_SIZE, len(images))}/{len(images)} images...")

    end_time = time.time()
    total_time = end_time - start_time
    fps = len(images) / total_time
    
    diff = abs(total_actual - total_predicted)
    error_rate = (diff / total_actual * 100) if total_actual > 0 else 0
    accuracy = (exact_matches / len(images)) * 100

    return {
        "time": total_time,
        "fps": fps,
        "predicted": total_predicted,
        "actual": total_actual,
        "diff": diff,
        "error_rate": error_rate,
        "exact_match_acc": accuracy
    }

def main():
    print("Scanning dataset directories...")
    all_images = []
    # Loop through all splits to get the full 12,000+ dataset
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(FLAT_DATA_DIR, split)
        images = glob.glob(os.path.join(split_dir, "*.jpg"))
        all_images.extend(images)
        
    if not all_images:
        print(f"No images found in {FLAT_DATA_DIR}")
        return
        
    print(f"✅ Found {len(all_images)} images across train/val/test splits.")
    
    results_summary = {}
    
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Model path does not exist: {path}")
            continue
            
        res = evaluate_model(name, path, all_images)
        if res:
            results_summary[name] = res

    print("\n" + "="*60)
    print("نتائج التقييم الشامل (Full Evaluation Results) - YOLO 26")
    print("="*60)
    
    for name, stats in results_summary.items():
        print(f"إجمالي الصور (Total Images)  : {len(all_images)}")
        print(f"الوقت الإجمالي (Total Time)  : {stats['time']:.2f} ثانية (Seconds)")
        print(f"السرعة (Speed - FPS)         : {stats['fps']:.2f} صورة/ثانية (Frames Per Second)")
        print("-" * 30)
        print(f"العدد الفعلي للألغام (XML)   : {stats['actual']}")
        print(f"العدد المتوقع بواسطة الموديل : {stats['predicted']}")
        print(f"فارق العدد (الاختلاف)        : {stats['diff']}")
        print(f"نسبة الخطأ في العدد          : {stats['error_rate']:.2f}%")
        print(f"نسبة التطابق التام بالصورة   : {stats['exact_match_acc']:.2f}%")
        
    print("="*60)

if __name__ == "__main__":
    main()
