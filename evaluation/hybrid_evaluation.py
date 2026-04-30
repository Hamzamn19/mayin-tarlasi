import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DETECTIONS_PATH = os.path.join(OUTPUT_DIR, "yolo_hybrid_features.csv")
GT_PATH = os.path.join(OUTPUT_DIR, "landmine_tabular_dataV3.csv")
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
LR_MODEL_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

def main():
    print("Loading data and models for Hybrid & Ensemble Evaluation...")
    df_det = pd.read_csv(DETECTIONS_PATH)
    df_gt = pd.read_csv(GT_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 1. Split Logic based on filename prefix
    df_det['split'] = df_det['source_file'].apply(lambda x: 'test' if x.startswith('elevation_test') else 'train')
    df_gt['split'] = df_gt['source_file'].apply(lambda x: 'test' if x.startswith('elevation_test') else 'train')
    
    test_det = df_det[df_det['split'] == 'test'].copy()
    test_gt = df_gt[(df_gt['split'] == 'test') & (df_gt['label'] == 1)]
    total_gt_mines = len(test_gt)

    print(f"Evaluation Scope: TEST SPLIT")
    print(f"Total Ground Truth Mines: {total_gt_mines}")
    print(f"Total YOLO Detections in Test: {len(test_det)}")

    # Features
    features = ["area", "circularity", "mean_intensity", "thermal_contrast", "edge_density"]
    X_test_scaled = scaler.transform(test_det[features])

    # 2. Get Individual Predictions
    test_det['yolo_pred'] = 1 # YOLO is already 1 for all rows in test_det
    test_det['rf_pred'] = rf_model.predict(X_test_scaled)
    test_det['lr_pred'] = lr_model.predict(X_test_scaled)

    # 3. Ensemble Strategies
    test_det['ens_and'] = (test_det['rf_pred'] == 1) & (test_det['lr_pred'] == 1)
    test_det['ens_majority'] = (test_det['rf_pred'] == 1) | (test_det['lr_pred'] == 1)
    
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    test_det['ens_soft'] = (rf_probs * 0.7 + lr_probs * 0.3) > 0.5

    def calc_metrics(pred_col, name):
        tp = test_det[(test_det['label'] == 1) & (test_det[pred_col] == 1)].shape[0]
        fp = test_det[(test_det['label'] == 0) & (test_det[pred_col] == 1)].shape[0]
        fn = total_gt_mines - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {"Strategy": name, "TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}

    results = [
        calc_metrics('yolo_pred', "YOLO Only"),
        calc_metrics('rf_pred', "YOLO + Random Forest"),
        calc_metrics('lr_pred', "YOLO + Logistic Regression"),
        calc_metrics('ens_and', "Ensemble Unanimous (YOLO & RF & LR)"),
        calc_metrics('ens_majority', "Ensemble Majority (YOLO + [RF or LR])"),
        calc_metrics('ens_soft', "Ensemble Soft Weighted (70% RF + 30% LR)")
    ]

    res_df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("COMPREHENSIVE HYBRID & ENSEMBLE SYSTEM COMPARISON")
    print("="*90)
    print(res_df.to_string(index=False))
    print("="*90)
    
    res_df.to_csv(os.path.join(OUTPUT_DIR, "hybrid_evaluation_results.csv"), index=False)
    print("\nResults saved to 'outputs/hybrid_evaluation_results.csv'")

if __name__ == "__main__":
    main()
