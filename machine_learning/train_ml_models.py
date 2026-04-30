import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_PATH = os.path.join(OUTPUT_DIR, 'landmine_tabular_dataV3.csv')

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    features = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
    target = 'label'

    print(f"Total number of records: {len(df)}")

    # Split the train and test sets
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    if len(train_df) == 0 or len(test_df) == 0:
        print("'split' column not found, splitting manually...")
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Scaling features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_filename = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as '{scaler_filename}'.")

    # 1. Train Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"LR Model Accuracy: {lr_accuracy:.4f}")
    print("Classification Report (LR):")
    print(classification_report(y_test, lr_pred))
    
    lr_model_filename = os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl')
    joblib.dump(lr_model, lr_model_filename)
    
    # 2. Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"RF Model Accuracy: {rf_accuracy:.4f}")
    print("Classification Report (RF):")
    print(classification_report(y_test, rf_pred))
    
    rf_model_filename = os.path.join(OUTPUT_DIR, 'random_forest_model.pkl')
    joblib.dump(rf_model, rf_model_filename)

    print("\n" + "="*50)
    print(f"Models and Scaler successfully saved to the outputs directory.")

if __name__ == "__main__":
    main()
