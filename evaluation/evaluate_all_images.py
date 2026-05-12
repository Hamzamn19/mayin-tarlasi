import os
import glob
import time
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "landmine_flat")
MODEL_PATH = os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")

def parse_xml_ground_truth(xml_path):
    """Extract ground truth bounding boxes and labels from a Pascal VOC XML file."""
    if not os.path.exists(xml_path):
        return []
    
    gt_boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xmin = int(float(xmlbox.find('xmin').text))
        ymin = int(float(xmlbox.find('ymin').text))
        xmax = int(float(xmlbox.find('xmax').text))
        ymax = int(float(xmlbox.find('ymax').text))
        gt_boxes.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
        
    return gt_boxes

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def main():
    # 1. Gather all images
    print("🔍 Scanning directory for images...")
    image_paths = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(IMAGE_DIR, split)
        images = glob.glob(os.path.join(split_dir, "*.jpg"))
        image_paths.extend(images)
        
    total_images = len(image_paths)
    print(f"✅ Found {total_images} images in total.")
    
    # 2. Load Model
    print(f"\n🧠 Loading YOLO26 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')
    model.to('cuda') # Ensure GPU is used
    
    # Warm up GPU
    print("🔥 Warming up GPU...")
    _ = model(image_paths[:16], verbose=False, device=0)
    
    # =========================================================================
    # STEP 1: RUN YOLO ON ALL IMAGES (MEASURE TIME ONLY)
    # =========================================================================
    print(f"\n🚀 STEP 1: Running Inference on {total_images} images (Batch size = 32)...")
    batch_size = 32
    
    start_time = time.time()
    
    # We use stream=True for generator-based efficient batch processing
    results_generator = model(image_paths, batch=batch_size, stream=True, verbose=False, device=0, conf=0.25)
    
    yolo_predictions = {}
    
    # Consume the generator to actually perform inference
    for i, r in enumerate(results_generator):
        img_path = image_paths[i]
        
        preds = []
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            preds.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})
            
        yolo_predictions[img_path] = preds
        
        if (i + 1) % 1000 == 0:
            print(f"  -> Processed {i + 1} / {total_images} images...")
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_images / total_time
    
    print("\n" + "="*50)
    print("⏱️  INFERENCE TIMING RESULTS")
    print("="*50)
    print(f"Total Images Processed: {total_images}")
    print(f"Total Time Taken      : {total_time:.2f} seconds")
    print(f"Speed (Frames/Sec)    : {fps:.2f} FPS")
    print(f"Average Time per Image: {(total_time/total_images)*1000:.2f} ms")
    
    # =========================================================================
    # STEP 2: COMPARE AGAINST GROUND TRUTH (ERROR RATE)
    # =========================================================================
    print("\n🚀 STEP 2: Comparing against Manual XML Records (Ground Truth)...")
    
    total_gt_objects = 0
    total_yolo_objects = 0
    
    true_positives = 0  # YOLO found the mine correctly
    false_positives = 0 # YOLO predicted a mine, but there is none (or wrong spot)
    false_negatives = 0 # YOLO missed a mine
    
    IOU_THRESHOLD = 0.40 # Threshold to consider a prediction "correct"
    
    for img_path in image_paths:
        xml_path = img_path.replace('.jpg', '.xml')
        gt_boxes = parse_xml_ground_truth(xml_path)
        preds = yolo_predictions[img_path]
        
        total_gt_objects += len(gt_boxes)
        total_yolo_objects += len(preds)
        
        # Match predictions to GT
        matched_gt = []
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue # Already matched
                iou = compute_iou(p['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= IOU_THRESHOLD:
                true_positives += 1
                matched_gt.append(best_gt_idx)
            else:
                false_positives += 1
                
        # False negatives are GT boxes that were not matched by any prediction
        false_negatives += (len(gt_boxes) - len(matched_gt))
        
    print("\n" + "="*50)
    print("📊 ERROR RATE & COMPARISON RESULTS")
    print("="*50)
    print(f"Total Actual Mines (Manual XML) : {total_gt_objects}")
    print(f"Total YOLO Detections           : {total_yolo_objects}")
    print("-" * 30)
    print(f"✅ Exact Matches (True Positives) : {true_positives}")
    print(f"❌ Missed Mines (False Negatives) : {false_negatives}")
    print(f"⚠️  False Alarms (False Positives) : {false_positives}")
    
    if total_gt_objects > 0:
        recall = true_positives / total_gt_objects
        print(f"\n📈 Recall (Detection Rate)      : {recall*100:.2f}%")
        
    if total_yolo_objects > 0:
        precision = true_positives / total_yolo_objects
        print(f"🎯 Precision (Accuracy of Dets) : {precision*100:.2f}%")
        
    if (true_positives + false_positives + false_negatives) > 0:
         overall_error_rate = (false_positives + false_negatives) / (true_positives + false_positives + false_negatives)
         print(f"📉 Overall Error Rate           : {overall_error_rate*100:.2f}%")

if __name__ == "__main__":
    main()
