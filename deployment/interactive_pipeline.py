import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os
import joblib
import re
import time

# --- Configuration & Styling ---
st.set_page_config(page_title="Interactive Landmine Detection Pipeline", layout="wide")

# Custom CSS for a compact, single-viewport look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 100%);
        overflow: hidden;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 1200px;
    }
    .step-header {
        color: #1E3A8A;
        font-size: 1.8rem !important;
        font-weight: 850;
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    .step-subtitle {
        color: #3B82F6;
        font-size: 0.9rem !important;
        font-weight: 600;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .feature-card {
        background: white;
        padding: 0.8rem !important;
        border-radius: 10px;
        border-top: 3px solid #3B82F6;
        margin-bottom: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        font-size: 0.9rem;
    }
    .model-card {
        background: linear-gradient(145deg, #ffffff, #F1F5F9);
        padding: 1rem !important;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .stImage > img {
        max-height: 350px !important;
        object-fit: contain;
    }
    .stButton>button {
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 700;
        height: 3rem !important;
    }
    div[data-testid="stExpander"] {
        margin-top: 0 !important;
    }
    /* Reduce spacing between streamlit elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Resolve absolute paths based on project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(os.path.join(BASE_DIR, "results", "runs", "detect", "Landmine_Detection_2026", "YOLO26_S_Standard", "weights", "best.pt"))
    lr_model = joblib.load(os.path.join(BASE_DIR, "outputs", "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "outputs", "random_forest_model.pkl"))
    return yolo_model, lr_model, rf_model

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

# --- Session State Management ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {}

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def restart():
    st.session_state.step = 0
    st.session_state.uploaded_image = None
    st.session_state.pipeline_data = {}

# --- Helper to Run Pipeline ---
def run_full_pipeline(pil_image):
    with st.spinner("🧠 Intelligence system is processing the data..."):
        yolo_model, lr_model, rf_model = load_models()
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = yolo_model(img, conf=0.25)
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                crop = img[y1:y2, x1:x2]
                features = extract_features_from_crop(crop, img_gray, x1, y1, x2, y2)
                
                if features:
                    f_df = pd.DataFrame([list(features.values())], columns=list(features.keys()))
                    lr_prob = lr_model.predict_proba(f_df)[0][1]
                    rf_prob = rf_model.predict_proba(f_df)[0][1]
                    ensemble_prob = (conf + rf_prob) / 2.0
                    
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "crop": crop,
                        "features": features,
                        "lr_prob": lr_prob,
                        "rf_prob": rf_prob,
                        "ensemble_prob": ensemble_prob
                    })
        
        st.session_state.pipeline_data['detections'] = detections
        st.session_state.pipeline_data['original_img'] = img
        
        # Pre-generate annotated images
        img_yolo = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (59, 130, 246), 2)
        st.session_state.pipeline_data['yolo_annotated'] = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
        
        img_ens = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            prob = d['ensemble_prob']
            color = (0, 0, 239) if prob >= 0.5 else (0, 185, 16)
            cv2.rectangle(img_ens, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_ens, f"{prob*100:.0f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        st.session_state.pipeline_data['ensemble_annotated'] = cv2.cvtColor(img_ens, cv2.COLOR_BGR2RGB)

# --- Slides ---

def slide_welcome():
    st.markdown('<div class="step-header">Project Revelation</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Landmine Detection Intelligence</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### Strategic Intelligence System
        This interactive pipeline demonstrates the advanced methodology for detecting buried landmines using LWIR Thermal signatures.
        
        **The 4 Pillars of Analysis:**
        1. **Vision:** YOLOv8 identifies spatial hotspots.
        2. **Physics:** Handcrafted geometric & thermal feature extraction.
        3. **Logic:** Statistical inference via Logistic Regression.
        4. **Wisdom:** Ensemble voting across multiple ML models.
        """)
        
        uploaded_file = st.file_uploader("Upload Thermal Data to Begin", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.success("✅ Image verified. Click 'Start Analysis' to proceed.")
    
    with col2:
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, width='stretch', caption="Input Data Stream")
        else:
            st.info("System awaiting input data...")

def slide_yolo():
    st.markdown('<div class="step-header">Phase 1: Neural Scanning</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">YOLOv8 Object Localization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        if 'yolo_annotated' in st.session_state.pipeline_data:
            st.image(st.session_state.pipeline_data['yolo_annotated'], width='stretch')
        
    with col2:
        st.markdown(f"""
        <div class="model-card">
        <h3>Vision Strategy</h3>
        The system uses <b>YOLOv8</b> to scan for thermal anomalies. 
        <br><br>
        - <b>Objects Found:</b> <code>{len(st.session_state.pipeline_data.get('detections', []))}</code>
        <br>
        - <b>Recall Priority:</b> Catching all potential threats before filtering.
        </div>
        """, unsafe_allow_html=True)

def slide_features():
    st.markdown('<div class="step-header">Phase 2: Feature Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Handcrafted Metric Extraction</div>', unsafe_allow_html=True)
    
    detections = st.session_state.pipeline_data.get('detections', [])
    if not detections:
        st.warning("Analysis complete: No suspicious objects found.")
        return

    det_idx = st.selectbox("Select Object ID for Deep Analysis", range(len(detections)))
    det = detections[det_idx]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(cv2.cvtColor(det['crop'], cv2.COLOR_BGR2RGB), width=300, caption=f"Object #{det_idx}")
    
    with col2:
        f = det['features']
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'<div class="feature-card"><b>Hotspot Area</b><br>{f["area"]:.1f} px²</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-card"><b>Geometric Circularity</b><br>{f["circularity"]:.3f}</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="feature-card"><b>Thermal Contrast</b><br>{f["thermal_contrast"]:.1f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-card"><b>Edge Sharpness</b><br>{f["edge_density"]:.3f}</div>', unsafe_allow_html=True)

def slide_ml_logic(model_name):
    title = "Logistic Regression" if model_name == "LR" else "Random Forest"
    st.markdown(f'<div class="step-header">Phase 3: {title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-subtitle">Statistical Inference Layer</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        if model_name == "LR":
            st.markdown("""
            ### The Math behind LR
            Logistic regression determines the probability of a 'Mine' vs 'Background' using a sigmoidal curve. 
            """)
            st.latex(r"P(y=1|x) = \sigma(\beta^T x)")
        else:
            st.markdown("""
            ### Random Forest Strategy
            An ensemble of hundreds of decision trees voting on the final outcome. 
            Highly robust against sensor noise.
            """)
            st.write("🌲 Voting consensus achieved.")

    with col2:
        st.write("### Inference Results")
        results = []
        for i, d in enumerate(st.session_state.pipeline_data['detections']):
            prob = d['lr_prob'] if model_name == "LR" else d['rf_prob']
            results.append({
                "ID": i, 
                "Probability": f"{prob*100:.1f}%", 
                "Decision": f"{'MINE' if prob >= 0.5 else 'SAFE'}"
            })
        st.table(results)

def slide_ensemble():
    st.markdown('<div class="step-header">Phase 4: Ensemble Consensus</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Final Decision Fusion</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
    The final decision is reached via <b>Weighted Voting</b> between YOLO (Spatial) and Random Forest (Structural). 
    This maximizes precision while maintaining high recall.
    </div>
    """, unsafe_allow_html=True)
    
    if 'ensemble_annotated' in st.session_state.pipeline_data:
        st.image(st.session_state.pipeline_data['ensemble_annotated'], width='stretch')

def slide_results():
    st.markdown('<div class="step-header">Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-subtitle">Consolidated Detection Report</div>', unsafe_allow_html=True)
    
    # Create numeric dataframe for proper gradient calculation
    df = pd.DataFrame([
        {
            "Object ID": f"#{i}",
            "YOLO Conf": d['conf'],
            "RF Prob": d['rf_prob'],
            "FINAL Confidence": d['ensemble_prob'],
            "Classification": "⚠ LANDMINE" if d['ensemble_prob'] >= 0.5 else "✅ CLEAR"
        } for i, d in enumerate(st.session_state.pipeline_data.get('detections', []))
    ])
    
    if not df.empty:
        # Style with gradient on numeric values, then format as percentage for display
        st.dataframe(
            df.style.background_gradient(subset=["FINAL Confidence"], cmap="RdYlGn")
            .format({
                "YOLO Conf": "{:.1%}",
                "RF Prob": "{:.1%}",
                "FINAL Confidence": "{:.1%}"
            }), 
            width='stretch'
        )
    else:
        st.info("No data available to display.")
    
    st.divider()
    if st.button("🔄 Reset System", on_click=restart):
        pass

# --- Main Logic ---
def main():
    steps = ["Intro", "YOLO", "Features", "Logistic Reg", "Random Forest", "Ensemble", "Results"]
    
    if st.session_state.step == 0:
        slide_welcome()
    elif st.session_state.step == 1:
        slide_yolo()
    elif st.session_state.step == 2:
        slide_features()
    elif st.session_state.step == 3:
        slide_ml_logic("LR")
    elif st.session_state.step == 4:
        slide_ml_logic("RF")
    elif st.session_state.step == 5:
        slide_ensemble()
    elif st.session_state.step == 6:
        slide_results()

    st.divider()
    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.step > 0:
            if st.button("← Previous", key="global_prev", type="secondary"):
                prev_step()
                st.rerun()
                
    with col_next:
        if st.session_state.step < len(steps) - 1:
            btn_label = "Start Analysis →" if st.session_state.step == 0 else "Next Stage →"
            if st.button(btn_label, key="global_next", type="primary"):
                if st.session_state.step == 0:
                    if st.session_state.uploaded_image:
                        run_full_pipeline(st.session_state.uploaded_image)
                        next_step()
                        st.rerun()
                    else:
                        st.error("Please upload an image first!")
                else:
                    next_step()
                    st.rerun()

if __name__ == "__main__":
    main()
