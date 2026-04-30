import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os
import joblib
import re

# Resolve absolute paths based on project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration & Styling ---
st.set_page_config(page_title="Aerial Landmine Detection", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(os.path.join(BASE_DIR, "results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt"))
    lr_model = joblib.load(os.path.join(BASE_DIR, "outputs", "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "outputs", "random_forest_model.pkl"))
    return yolo_model, lr_model, rf_model

@st.cache_data
def load_ground_truth():
    csv_path = os.path.join(BASE_DIR, "outputs", "landmine_tabular_dataV3.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# --- Helper Functions ---

def extract_features_from_crop(crop, img_gray, xmin, ymin, xmax, ymax):
    if crop is None or crop.size == 0: return None
    h, w = crop.shape[:2]
    if h < 3 or w < 3: return None
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    
    area = float(h * w)
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0: circularity = float(4 * np.pi * cnt_area / (perimeter ** 2))

    mean_intensity = float(np.mean(gray_crop))
    img_h, img_w = img_gray.shape[:2]
    pad = 5
    bx1, by1 = max(0, xmin - pad), max(0, ymin - pad)
    bx2, by2 = min(img_w, xmax + pad), min(img_h, ymax + pad)
    bg_region = img_gray[by1:by2, bx1:bx2].copy().astype(float)
    obj_mask = np.zeros_like(bg_region, dtype=bool)
    oy1, ox1 = ymin - by1, xmin - bx1
    obj_mask[oy1:oy1 + h, ox1:ox1 + w] = True
    bg_pixels = bg_region[~obj_mask]
    bg_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else mean_intensity
    thermal_contrast = float(abs(mean_intensity - bg_mean))
    edges = cv2.Canny(gray_crop, 50, 150)
    edge_density = float(np.sum(edges > 0)) / area if area > 0 else 0.0

    return {"area": area, "circularity": circularity, "mean_intensity": mean_intensity, 
            "thermal_contrast": thermal_contrast, "edge_density": edge_density}

# --- Main App ---

def main():
    st.title("🔍 Aerial Landmine Detection System")
    st.markdown("### Machine Learning Project: Estimation and Prediction")
    
    yolo_model, lr_model, rf_model = load_models()
    gt_df = load_ground_truth()
    
    # --- Sidebar: Advanced ML Settings ---
    st.sidebar.header("⚙️ Advanced ML Strategies")
    st.sidebar.markdown("**1. Lowering Confidence Threshold:**")
    conf_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold", 
        min_value=0.01, max_value=0.80, value=0.25, step=0.01,
        help="Lowering this allows YOLO to act as a 'High Recall' filter, catching more suspect objects to minimize False Negatives."
    )
    
    st.sidebar.markdown("**2. Ensemble Voting:**")
    st.sidebar.info("Averaging YOLO's Computer Vision confidence with Random Forest's Feature Probability to make a balanced final decision.")
    # -------------------------------------

    uploaded_file = st.file_uploader("Choose an LWIR thermal image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        file_name = uploaded_file.name
        
        # Clean prefix and prepare XML matching name
        clean_file_name = re.sub(r'^marked_', '', file_name)
        xml_name = os.path.splitext(clean_file_name)[0] + ".xml"
        
        process_image(image, yolo_model, lr_model, rf_model, conf_threshold)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Original Image")
            st.image(image, width="stretch")
        with col_img2:
            st.subheader(f"YOLOv8 Detection (Threshold: {conf_threshold})")
            if 'annotated_image' in st.session_state:
                st.image(st.session_state.annotated_image, width="stretch")

        st.divider()
        col_res1, col_res2 = st.columns([1, 1.5])

        # --- Exact Ground Truth Matching ---
        gt_count = "N/A"
        mines_only = pd.DataFrame()
        if gt_df is not None:
            # Match exactly since names are now unique (e.g. elevation_test_Jan_Morning_..._lwir_71.xml)
            file_matches = gt_df[gt_df['source_file'] == xml_name]
            if not file_matches.empty:
                mines_only = file_matches[file_matches['label'] == 1].copy()
                gt_count = len(mines_only)

        with col_res1:
            st.header("📋 Ground Truth")
            if gt_count != "N/A":
                if gt_count > 0:
                    st.write(f"Found **{gt_count}** actual mines:")
                    st.dataframe(mines_only[['mine_type', 'area', 'circularity', 'mean_intensity']], width="stretch")
                else:
                    st.info("No mines recorded for this file (Background only).")
            else:
                st.warning(f"No records found for `{xml_name}`. Make sure to upload from the flattened dataset.")

        with col_res2:
            st.header("🤖 Detailed Predictions")
            if 'results_df' in st.session_state and not st.session_state.results_df.empty:
                def highlight_mines(val):
                    if isinstance(val, str) and "Mine (1)" in val: return 'color: red; font-weight: bold'
                    if isinstance(val, str) and "Not a Mine (0)" in val: return 'color: green'
                    return ''
                st.dataframe(st.session_state.results_df.style.map(highlight_mines, 
                             subset=['Logistic_Regression', 'Random_Forest', 'Ensemble (YOLO+RF)']), width="stretch")
            elif 'results_df' in st.session_state:
                st.warning("No objects detected.")

        # --- SUMMARY COMPARISON TABLE ---
        st.divider()
        st.header("📊 Advanced Models Comparison Summary")
        if 'results_df' in st.session_state and not st.session_state.results_df.empty:
            df = st.session_state.results_df
            
            yolo_count = len(df)
            yolo_conf = df['Conf'].astype(float).mean() * 100
            
            lr_mines = df[df['Logistic_Regression'].str.startswith("Mine (1)")]
            lr_count = len(lr_mines)
            lr_conf = lr_mines['Logistic_Regression'].str.extract(r'\(([\d.]+)%\)').astype(float).mean()[0] if lr_count > 0 else 0
            
            rf_mines = df[df['Random_Forest'].str.startswith("Mine (1)")]
            rf_count = len(rf_mines)
            rf_conf = rf_mines['Random_Forest'].str.extract(r'\(([\d.]+)%\)').astype(float).mean()[0] if rf_count > 0 else 0
            
            ens_mines = df[df['Ensemble (YOLO+RF)'].str.startswith("Mine (1)")]
            ens_count = len(ens_mines)
            ens_conf = ens_mines['Ensemble (YOLO+RF)'].str.extract(r'\(([\d.]+)%\)').astype(float).mean()[0] if ens_count > 0 else 0

            summary_data = {
                "Model / Strategy": [
                    "📋 Ground Truth (Actual Mines)", 
                    f"👁️ YOLOv8 (High Recall - Thr: {conf_threshold})", 
                    "📈 Logistic Regression", 
                    "🌳 Random Forest",
                    "🤝 Ensemble Voting (YOLO + RF Avg)"
                ],
                "Detected Mines (Count)": [str(gt_count), str(yolo_count), str(lr_count), str(rf_count), str(ens_count)],
                "Average Confidence / Probability": [
                    "100%", 
                    f"{yolo_conf:.1f}%", 
                    f"{lr_conf:.1f}%" if lr_count > 0 else "N/A", 
                    f"{rf_conf:.1f}%" if rf_count > 0 else "N/A",
                    f"{ens_conf:.1f}%" if ens_count > 0 else "N/A"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            def style_summary(row):
                if "Ground Truth" in row['Model / Strategy']: return ['background-color: #e6f2ff; font-weight: bold'] * len(row)
                if "Ensemble" in row['Model / Strategy']: return ['background-color: #e6ffe6; font-weight: bold'] * len(row)
                return [''] * len(row)
                
            st.table(summary_df.style.apply(style_summary, axis=1))

def process_image(pil_image, yolo_model, lr_model, rf_model, conf_threshold):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = yolo_model(img, conf=conf_threshold)
    detected_objects = []
    feature_names = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls = float(box.conf[0]), int(box.cls[0])
            yolo_label = yolo_model.names[cls]
            crop = img[y1:y2, x1:x2]
            f = extract_features_from_crop(crop, img_gray, x1, y1, x2, y2)
            if f:
                f_df = pd.DataFrame([[f['area'], f['circularity'], f['mean_intensity'], f['thermal_contrast'], f['edge_density']]], columns=feature_names)
                lr_pred, lr_prob = lr_model.predict(f_df)[0], lr_model.predict_proba(f_df)[0][1]
                rf_pred, rf_prob = rf_model.predict(f_df)[0], rf_model.predict_proba(f_df)[0][1]
                
                ensemble_prob = (conf + rf_prob) / 2.0
                ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
                
                detected_objects.append({
                    "YOLO": yolo_label, "Conf": f"{conf:.2f}",
                    "Logistic_Regression": f"{'Mine (1)' if lr_pred == 1 else 'Not a Mine (0)'} ({lr_prob*100:.1f}%)",
                    "Random_Forest": f"{'Mine (1)' if rf_pred == 1 else 'Not a Mine (0)'} ({rf_prob*100:.1f}%)",
                    "Ensemble (YOLO+RF)": f"{'Mine (1)' if ensemble_pred == 1 else 'Not a Mine (0)'} ({ensemble_prob*100:.1f}%)",
                    "Area": f[ 'area'], "Intensity": f"{f['mean_intensity']:.1f}"
                })
                
                color = (0, 0, 255) if ensemble_pred == 1 else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"Ens: {ensemble_prob*100:.0f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.session_state.results_df = pd.DataFrame(detected_objects)
    st.session_state.annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    main()