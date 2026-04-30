import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH = os.path.join(OUTPUT_DIR, "yolo_hybrid_features.csv")
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
LR_MODEL_PATH = os.path.join(OUTPUT_DIR, "logistic_regression_model.pkl")

def main():
    print("Loading hybrid features data...")
    df = pd.read_csv(DATA_PATH)
    
    # Define features and label
    features = ["area", "circularity", "mean_intensity", "thermal_contrast", "edge_density"]
    X = df[features]
    y = df["label"]
    
    print(f"Dataset Size: {len(df)} samples")
    print(f"Class Distribution:\n{y.value_counts()}")
    
    # 1. Smart Split: Stratified to maintain class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Scaling: Essential for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Smart Random Forest Training
    # We use class_weight='balanced' to handle the 16:1 ratio
    # n_jobs=-1 for parallel processing (speed)
    print("\nTraining Smart Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        max_depth=15 # Avoid extreme overfitting
    )
    rf.fit(X_train_scaled, y_train)
    
    # 4. Smart Logistic Regression Training
    print("Training Smart Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    
    # 5. Evaluation
    print("\n" + "="*40)
    print("RANDOM FOREST EVALUATION")
    print("="*40)
    rf_preds = rf.predict(X_test_scaled)
    print(classification_report(y_test, rf_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))
    
    print("\n" + "="*40)
    print("LOGISTIC REGRESSION EVALUATION")
    print("="*40)
    lr_preds = lr.predict(X_test_scaled)
    print(classification_report(y_test, lr_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))
    
    # 6. Save Models (Including Scaler inside a pipeline or separately)
    # For simplicity, we'll save the scaler as well because it's required for inference
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    
    print("\nModels and Scaler saved successfully.")

if __name__ == "__main__":
    main()
