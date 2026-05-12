import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Skipping XGBoost training.")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_PATH = os.path.join(OUTPUT_DIR, 'landmine_tabular_dataV3.csv')

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    # Use all available features (backward compatible: works with 5 or 10)
    all_features = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density',
                    'intensity_std', 'aspect_ratio', 'thermal_gradient', 'max_min_ratio', 'relative_size']
    features = [f for f in all_features if f in df.columns]
    target = 'label'

    print(f"Total number of records: {len(df)}")
    print(f"Using {len(features)} features: {features}")
    print(f"Class distribution:\n{df[target].value_counts()}")

    # Split the train and test sets
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    if len(train_df) == 0 or len(test_df) == 0:
        print("'split' column not usable, splitting manually (stratified)...")
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Train label dist: {dict(y_train.value_counts())}")
    print(f"Test  label dist: {dict(y_test.value_counts())}")

    # Scaling features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_filename = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as '{scaler_filename}'.")

    # Results collector for final comparison
    results = {}

    # ════════════════════════════════════════════════════════════════
    # 1. Logistic Regression
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("1. Training Logistic Regression (class_weight='balanced')...")
    print("="*60)
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    print(f"Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f}")
    print(classification_report(y_test, lr_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))
    joblib.dump(lr_model, os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl'))
    results['Logistic Regression'] = {'accuracy': lr_acc, 'f1': lr_f1}
    
    # ════════════════════════════════════════════════════════════════
    # 2. Random Forest
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("2. Training Random Forest (200 trees, class_weight='balanced')...")
    print("="*60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    print(f"Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))

    # Feature Importance
    print("\nFeature Importance:")
    importances = rf_model.feature_importances_
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:20s} {imp:.4f} {bar}")
    
    joblib.dump(rf_model, os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))
    results['Random Forest'] = {'accuracy': rf_acc, 'f1': rf_f1}

    # ════════════════════════════════════════════════════════════════
    # 3. MLP Neural Network
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("3. Training MLP Neural Network (128→64→32)...")
    print("="*60)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )
    mlp_model.fit(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred)
    print(f"Accuracy: {mlp_acc:.4f} | F1: {mlp_f1:.4f}")
    print(f"Training stopped at epoch: {mlp_model.n_iter_}")
    print(classification_report(y_test, mlp_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, mlp_pred))
    
    joblib.dump(mlp_model, os.path.join(OUTPUT_DIR, 'mlp_neural_network_model.pkl'))
    results['MLP Neural Network'] = {'accuracy': mlp_acc, 'f1': mlp_f1}

    # ════════════════════════════════════════════════════════════════
    # 4. XGBoost (if available)
    # ════════════════════════════════════════════════════════════════
    if HAS_XGBOOST:
        print("\n" + "="*60)
        print("4. Training XGBoost...")
        print("="*60)
        
        # Calculate scale_pos_weight for class imbalance
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  Class ratio (neg/pos): {spw:.2f}")
        
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred)
        print(f"Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f}")
        print(classification_report(y_test, xgb_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, xgb_pred))
        
        joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, 'xgboost_model.pkl'))
        results['XGBoost'] = {'accuracy': xgb_acc, 'f1': xgb_f1}

    # ════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📊 FINAL MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 47)
    for name, scores in sorted(results.items(), key=lambda x: -x[1]['f1']):
        marker = " 🏆" if scores['f1'] == max(r['f1'] for r in results.values()) else ""
        print(f"{name:<25} {scores['accuracy']:>9.4f} {scores['f1']:>9.4f}{marker}")
    print("="*60)
    print(f"\nAll models saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

