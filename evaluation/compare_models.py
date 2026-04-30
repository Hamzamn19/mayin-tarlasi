import os
import random
import glob
import time
from ultralytics import YOLO

# Resolve absolute paths based on project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# الإعدادات
IMAGES_DIR = os.path.join(BASE_DIR, "landmine_flat", "train")
LABELS_DIR = os.path.join(BASE_DIR, "landmine_flat", "train")

MODELS = {
    "Model 8": "/home/hamzah/Downloads/Landmine_Output/checkpoints/last.pt",
    "Model 26": os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
}

CONF_THRESH = 0.25
NUM_IMAGES = 1000

def get_actual_count(img_path):
    base_name = os.path.basename(img_path).replace('.jpg', '.txt')
    label_path = os.path.join(LABELS_DIR, base_name)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            return len([line for line in f.read().splitlines() if line.strip()])
    return 0

def evaluate_model(model_name, model_path, images):
    print(f"\n[{model_name}] Loading model from {model_path} ...")
    try:
        # بالنسبة لموديلات ONNX، يجب تمرير المهمة "task='detect'"
        model = YOLO(model_path, task='detect')
        if model_path.endswith('.pt'):
            # نقل الموديل صراحة إلى كرت الشاشة فقط للـ PyTorch
            model.to('cuda')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    total_actual = 0
    total_predicted = 0
    exact_matches = 0
    
    # حساب الأعداد الفعلية مسبقاً لكل الصور
    actual_counts = [get_actual_count(img) for img in images]
    
    # تسخين كرت الشاشة (Warm up GPU)
    print(f"[{model_name}] Warming up GPU...")
    model.predict(source=images[0], conf=CONF_THRESH, device=0, verbose=False)
    
    print(f"[{model_name}] Starting batched inference on {len(images)} images using GPU...")
    start_time = time.time()
    
    # تشغيل الفحص على شكل دفعات (Batches) لاستغلال كرت الشاشة بأقصى طاقة
    batch_size = 16
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_actuals = actual_counts[i:i+batch_size]
        
        # device=0 لاختبار كرت الشاشة مع تفعيل الدفعات
        results = model.predict(source=batch_imgs, conf=CONF_THRESH, device=0, verbose=False, stream=True)
        
        for j, res in enumerate(results):
            actual_count = batch_actuals[j]
            predicted_count = len(res.boxes)
            
            total_actual += actual_count
            total_predicted += predicted_count
            if actual_count == predicted_count:
                exact_matches += 1
                
        if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(images):
            print(f"[{model_name}] Processed {min(i + batch_size, len(images))}/{len(images)} images...")

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
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    if not all_images:
        print(f"No images found in {IMAGES_DIR}")
        return
        
    # اختيار 1000 صورة عشوائياً للتقييم
    sampled_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
    print(f"Selected {len(sampled_images)} random images for testing.")
    
    results_summary = {}
    
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Model path does not exist: {path}")
            continue
            
        res = evaluate_model(name, path, sampled_images)
        if res:
            results_summary[name] = res

    print("\n" + "="*60)
    print("نتائج المقارنة (Comparison Results) - GPU Testing")
    print("="*60)
    
    for name, stats in results_summary.items():
        print(f"\n--- {name} ---")
        print(f"الوقت الإجمالي (Total Time)  : {stats['time']:.2f} ثانية (Seconds)")
        print(f"السرعة (Speed - FPS)         : {stats['fps']:.2f} صورة/ثانية (Frames Per Second)")
        print(f"العدد الفعلي للألغام         : {stats['actual']}")
        print(f"العدد المتوقع للألغام        : {stats['predicted']}")
        print(f"فارق العدد                   : {stats['diff']}")
        print(f"نسبة الخطأ في العدد          : {stats['error_rate']:.2f}%")
        print(f"نسبة التطابق التام بالصورة   : {stats['exact_match_acc']:.2f}%")
        
    print("\n" + "="*60)
    if len(results_summary) == 2:
        m1, m2 = list(results_summary.keys())
        faster = m1 if results_summary[m1]['fps'] > results_summary[m2]['fps'] else m2
        better_acc = m1 if results_summary[m1]['exact_match_acc'] > results_summary[m2]['exact_match_acc'] else m2
        print(f"الموديل الأسرع (Faster Model)          : {faster}")
        print(f"الموديل الأدق (More Accurate Model)    : {better_acc}")
    print("="*60)

if __name__ == "__main__":
    main()
