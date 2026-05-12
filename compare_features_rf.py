import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = "/home/hamzah/Desktop/beykoz/proje/MAYIN TARLASI"
# Actually read the file from the original directory since outputs are there
CSV_PATH = "/home/hamzah/Desktop/beykoz/proje/Machine Learning: Estimation and Prediction/MAYIN TARLASI/outputs/landmine_tabular_dataV3.csv"

def evaluate_model(X_train, X_test, y_train, y_test, name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    error_rate = (fp + fn) / (tn + fp + fn + tp)
    
    print(f"--- {name} ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"Total Err: {error_rate*100:.2f}%")
    print(f"False Positives: {fp}, False Negatives: {fn}\n")

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # 5 features
    features_5 = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
    
    # 10 features
    features_10 = features_5 + ['intensity_std', 'aspect_ratio', 'thermal_gradient', 'max_min_ratio', 'relative_size']
    
    target = 'label'
    
    X_5 = df[features_5]
    X_10 = df[features_10]
    y = df[target]
    
    # Split
    X_train_5, X_test_5, y_train_5, y_test = train_test_split(X_5, y, test_size=0.2, random_state=42, stratify=y)
    X_train_10, X_test_10, y_train_10, _ = train_test_split(X_10, y, test_size=0.2, random_state=42, stratify=y)
    
    evaluate_model(X_train_5, X_test_5, y_train_5, y_test, "Random Forest (5 Features)")
    evaluate_model(X_train_10, X_test_10, y_train_10, y_test, "Random Forest (10 Features)")

if __name__ == "__main__":
    main()
