# 💣 MAYIN TARLASI — Aerial Landmine Detection System

> **A Hybrid Deep Learning & Statistical Analysis Pipeline for Detecting Buried Landmines Using LWIR Thermal Imagery**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=ultralytics)](#)
[![YOLO26](https://img.shields.io/badge/YOLO26-Latest-orange)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikitlearn)](#)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#)

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
  - [Stage 1: Data Processing](#stage-1-data-processing)
  - [Stage 2: YOLO Training](#stage-2-yolo-training)
  - [Stage 3: Feature Extraction & EDA](#stage-3-feature-extraction--eda)
  - [Stage 4: ML Model Training](#stage-4-ml-model-training)
  - [Stage 5: Evaluation](#stage-5-evaluation)
  - [Stage 6: Deployment](#stage-6-deployment)
- [Extracted Features](#-extracted-features)
- [Models](#-models)
- [Benchmarking Results](#-benchmarking-results)
- [Installation & Usage](#-installation--usage)
- [Technologies Used](#-technologies-used)
- [Acknowledgements](#-acknowledgements)

---

## 📝 Abstract

This project presents a comprehensive **dual-stage landmine detection pipeline** using Long-Wave Infrared (LWIR) imagery captured by Unmanned Aerial Systems (UAS). The system combines:

1. **Deep Learning (YOLO)** — For real-time object detection and spatial localization of thermal anomalies.
2. **Classical Machine Learning (Random Forest + Logistic Regression)** — For feature-based verification using handcrafted thermal and geometric features.

The hybrid ensemble approach overcomes the limitations of pure thermal sensing and achieves a **peak mAP@50 of 91.9%**, significantly outperforming baseline results published by the original AMLID researchers.

---

## 🏆 Key Results

### ⚡ Quick Summary (Key Performance Indicators)

| Metric | Value |
|:---|:---:|
| **YOLO26 mAP@50 (Overall)** | **91.9%** |
| **AT Plastic Precision** | 99.1% |
| **AT Metal Precision** | 97.4% |
| **AP Plastic Precision** | 86.7% |
| **AP Metal Precision** | 84.6% |
| **Random Forest Accuracy** | 80.72% |
| **Inference Speed** | 4.9ms/image (~204 FPS) |
| **Optimized Model Size** | 20.3 MB |

### 📊 Comprehensive Evaluation Results (1,320 Images - 14,905 Landmines)

| System | Precision | Recall | Total Error Rate |
|:---|:---:|:---:|:---:|
| **YOLO26 Only** | 94.27% | 95.78% | 10.04% |
| **YOLO + Random Forest 🏆 (Threshold 0.4)** | **98.29%** | **95.55%** | **6.11%** |
| **YOLO + XGBoost** | 98.76% | 88.37% | 12.75% |
| **Hybrid System (Soft Voting)** | 98.50% | 95.04% | 6.40% |

#### 🔍 Final Comprehensive Analysis:

1. **The Absolute Winner (Random Forest):**
   * The Random Forest model (with a 0.4 safety threshold) proved to be the most stable and reliable across the entire dataset.
   * It reduced the total error rate from 10% (in YOLO alone) to just 6.11%.
   * It boosted precision to 98.29%, meaning that out of every 100 mines detected by the system, 98 are indeed real mines (near-zero false alarms).

2. **Strong YOLO Performance:**
   * YOLOv26 shows immense capability in initial detection (Recall 95.78%), serving as the solid foundation upon which the remaining filters are built.

3. **XGBoost:**
   * Despite being highly precise (98.76%), it tends to miss mines more frequently than RF, which inflated its overall error rate.

4. **Hybrid System (Soft Voting):**
   * Its performance is very close to RF. It is an excellent choice if a more "conservative" system is desired to combine the strengths of both models, but RF alone was better at minimizing total errors.

---

### 📊 Performance Comparison: 5 Features vs 10 Features (Random Forest)

| Metric | 5 Features Model | 10 Features Model 🏆 | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 86.02% | **89.96%** | 🟢 +3.94% |
| **Precision** | 87.16% | **90.64%** | 🟢 +3.48% |
| **Recall** | 84.63% | **89.22%** | 🟢 +4.59% |
| **Total Error Rate** | 13.98% | **10.04%** | 📉 -3.94% |
| **False Positives** | 2,871 | **2,120** | 🛡️ 751 fewer false alarms |
| **False Negatives** | 3,538 | **2,481** | 🎯 1,057 more mines detected |

#### 💡 Analysis: Why is the difference so significant?

1. **The 5 Features (Basic):**
   *(Area, Circularity, Mean Intensity, Thermal Contrast, Edge Density)*
   These focus strictly on **size, shape, and basic contrast**. While effective, they fall into the trap of naturally occurring circular hot objects (like heated rocks or scattered metal pieces), resulting in high false alarms (2,871).

2. **The 10 Features (Advanced):**
   *Addition of: (Intensity Std Dev, Aspect Ratio, Thermal Gradient, Max/Min Ratio, Relative Size)*
   These features give the system a much deeper dimension for understanding the **structure and heat distribution inside the object**:
   - **(Thermal Gradient):** Distinguishes the geometrically engineered edges of a mine from the random edges of rocks.
   - **(Intensity Std Dev & Max/Min Ratio):** Differentiates the uniform heat distribution of a mine from the random distribution of natural objects.
   - **Result:** A massive drop in missed mines (False Negatives reduced by nearly a third) and a significant increase in confidence, effectively avoiding false alarms.

**Conclusion:**
Adding the 5 advanced features was not just padding data; it gave the `Random Forest` model the necessary statistical awareness to accurately distinguish between a "hot circular object" and a "real landmine," making it the strongest defensive filter after the initial YOLO detection.

---

#### 💡 Final Project Recommendation:
Based on this comprehensive evaluation, the best technical configuration for the project is:
**YOLOv26 + Random Forest (Threshold: 0.4) + Physical Features (10 features) + YOLO Confidence.**
This system is the safest and most accurate across all weather conditions and times captured in the dataset (Morning, Noon, Afternoon).

*(Note: Logistic Regression was initially tested but failed to generalize effectively due to the non-linear nature of thermal features, making it a bad example of classical modeling for this domain. It has been removed from the final inference pipeline.)*

### Comparative Benchmarking vs Original AMLID Research

| Metric | Original AMLID (YOLOv11) | Our System (YOLO26) | Improvement |
|:---|:---:|:---:|:---:|
| Peak mAP@50 | 86.80% | **91.90%** | +5.10% |
| Precision (AT Plastic) | 70.30% | **99.10%** | +28.80% |
| Precision (AP Metal) | 19.30% | **84.60%** | +65.30% |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AERIAL LANDMINE DETECTION PIPELINE                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐ │
│  │ LWIR     │    │ YOLOv8/26  │    │  Feature    │    │  Ensemble    │ │
│  │ Thermal  │───>│ Object     │───>│ Extraction  │───>│  Voting      │ │
│  │ Image    │    │ Detection  │    │ (10 Features)│    │ (YOLO + RF)  │ │
│  └──────────┘    └────────────┘    └─────────────┘    └──────────────┘ │
│                        │                  │                  │          │
│                        v                  v                  v          │
│                  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│                  │ Bounding │    │ Random Forest│    │ Final Mine   │  │
│                  │ Boxes    │    │ + Logistic   │    │ Classification│  │
│                  │ + Conf   │    │ Regression   │    │ Decision     │  │
│                  └──────────┘    └──────────────┘    └──────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The project uses the **AMLID (Airborne Multi-sensor Landmine Image Dataset)** — a high-resolution multi-class landmine dataset captured using LWIR thermal cameras mounted on UAS platforms.

| Property | Details |
|:---|:---|
| **Total Images** | ~12,067 LWIR thermal images |
| **Annotation Format** | Pascal VOC (XML) + YOLO (TXT) |
| **Classes (4)** | `at_plastic`, `ap_plastic`, `at_metal`, `ap_metal` |
| **Image Resolution** | Variable (resized to 640×640 for training) |
| **Data Split** | Train (80%) / Val (10%) / Test (10%) |
| **Capture Conditions** | Multiple times of day (Morning, Afternoon, Noon), multiple elevations (0–100m) |

### Mine Classes

| Class | Full Name | Description |
|:---|:---|:---|
| `at_plastic` | Anti-Tank Plastic | Large, plastic-cased anti-tank mines |
| `at_metal` | Anti-Tank Metal | Large, metal-cased anti-tank mines |
| `ap_plastic` | Anti-Personnel Plastic | Small, plastic-cased anti-personnel mines |
| `ap_metal` | Anti-Personnel Metal | Small, metal-cased anti-personnel mines |

---

## 📁 Project Structure

```
MAYIN TARLASI/
├── data_processing/               # Data preparation & feature extraction
│   ├── rename_dataset.py          # Flatten hierarchical dataset into unique filenames
│   ├── split_raw_data.py          # Split dataset into train/val/test (80/10/10)
│   ├── extract_pipeline.py        # Extract 5 handcrafted features from Pascal VOC annotations
│   ├── hybrid_feature_extraction.py # Extract features from YOLO detections with GT matching
│   ├── generate_yolo_ml_data.py   # Generate ML training data using YOLO inference + GT
│   └── eda_analysis.py            # Exploratory Data Analysis with visualizations
│
├── machine_learning/              # ML model training
│   ├── train_ml_models.py         # Train Logistic Regression & Random Forest classifiers
│   ├── smart_retrain_hybrid.py    # Retrain with class balancing (class_weight='balanced')
│   └── codeforkaggleyolo26.txt    # YOLO26 training code for Kaggle T4 GPU
│
├── evaluation/                    # Model evaluation & benchmarking
│   ├── compare_models.py          # Benchmark YOLOv8 vs YOLO26 (speed, accuracy, GPU)
│   ├── hybrid_evaluation.py       # Evaluate all ensemble strategies (AND/OR/Soft Voting)
│   ├── validate_csv_with_yolo.py  # Cross-validate CSV features against YOLO re-extraction
│   ├── evaluate_all_images.py     # Run comprehensive evaluation on the entire test set
│   └── test_100_images_pipeline.py # Rapid validation of the pipeline on 100 sample images
│
├── compare_features_rf.py         # Comparative analysis of 5 vs 10 features for Random Forest
│
├── deployment/                    # Web apps & inference
│   └── inference_tensorrt.cpp     # C++ TensorRT inference template for edge deployment
│
├── landmine_flat/                 # Flattened dataset (train/val/test splits)
│   ├── train/                     # Training images + XML annotations
│   ├── val/                       # Validation images + XML annotations
│   └── test/                      # Test images + XML annotations
│
├── outputs/                       # Generated outputs
│   ├── landmine_tabular_dataV3.csv  # Extracted tabular features dataset
│   ├── yolo_hybrid_features.csv     # YOLO-based hybrid features
│   ├── logistic_regression_model.pkl # Trained LR model
│   ├── random_forest_model.pkl      # Trained RF model (~300MB)
│   ├── scaler.pkl                   # StandardScaler for feature normalization
│   └── plots/                       # EDA visualization outputs
│       ├── correlation_heatmap.png
│       ├── feature_distributions.png
│       └── outliers_distributions.png
│
├── results/                       # YOLO training results
│   ├── landmine_kaggle.yaml       # YOLO dataset configuration
│   ├── yolo26n.pt                 # YOLO26 Nano pretrained weights
│   └── runs/detect/               # Training runs & best weights
│
├── docs/                          # Documentation & reports
│   ├── FINAL_TECHNICAL_REPORT.md  # Complete technical evaluation report
│   ├── REPORT_DRAFT.md            # Initial report draft
│   ├── VISUALIZATIONS_GUIDE.md    # Guide for interpreting EDA visualizations
│   ├── Project Proposal.pdf       # Original project proposal
│   ├── Statistical_Aerial_Mine_Detection.pdf   # Final presentation (PDF)
│   └── Statistical_Aerial_Mine_Detection.pptx  # Final presentation (PowerPoint)
│
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🔧 Pipeline Stages

### Stage 1: Data Processing

The raw AMLID dataset has a hierarchical structure organized by capture conditions. The data processing pipeline flattens and organizes it:

```bash
# 1. Flatten the hierarchical dataset into unique filenames
python data_processing/rename_dataset.py

# 2. Split into train/val/test (80/10/10)
python data_processing/split_raw_data.py
```

**`rename_dataset.py`** — Walks the nested directory structure and creates unique flat filenames by encoding the path hierarchy (e.g., `Jan_Jan_Afternoon_0_lwir_1.jpg`).

**`split_raw_data.py`** — Performs a stratified 80/10/10 split ensuring both `.jpg` and `.xml` pairs are kept together.

### Stage 2: YOLO Training

The YOLO models were trained on **Kaggle** using a **Tesla T4 GPU** for maximum performance:

```python
# YOLO26 Small — 50 epochs, 640px, batch 32
model = YOLO('yolo26s.pt')
results = model.train(
    data='landmine_kaggle.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    patience=15,
    project='Landmine_Detection_2026',
    name='YOLO26_S_Standard'
)
```

**Key Training Parameters:**
| Parameter | Value | Rationale |
|:---|:---:|:---|
| Image Size | 640×640 | Standard YOLO resolution |
| Batch Size | 32 | Maximum for T4 GPU (16GB VRAM) |
| Epochs | 50 | With early stopping (patience=15) |
| Optimizer Stripping | ✅ | Reduced model from ~40MB to 20.3MB |

### Stage 3: Feature Extraction & EDA

Five handcrafted features are extracted from each detected/annotated region:

```bash
# Extract features from all annotated bounding boxes
python data_processing/extract_pipeline.py

# Run Exploratory Data Analysis
python data_processing/eda_analysis.py
```

**`extract_pipeline.py`** uses a **two-pass approach**:
1. **Pass 1** — Extracts ALL mine samples from annotated bounding boxes.
2. **Pass 2** — Generates balanced background samples by random placement in non-overlapping regions.

The EDA script generates:
- **Correlation Heatmap** — Feature-to-label correlations
- **Boxplots** — Class-separated distributions with median annotations
- **Histograms with KDE** — Density overlays for mine vs. background

### Stage 4: ML Model Training

Two classical ML models are trained on the extracted tabular features:

```bash
# Train LR and RF on the tabular dataset
python machine_learning/train_ml_models.py

# OR: Smart retrain with class balancing
python machine_learning/smart_retrain_hybrid.py
```

| Model | Configuration | Purpose |
|:---|:---|:---|
| **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'` | Linear decision boundary, probabilistic output |
| **Random Forest** | `n_estimators=200`, `max_depth=20`, `class_weight='balanced'` | Non-linear ensemble, noise-robust |

Both models use **StandardScaler** normalization and are saved as `.pkl` files for inference.

### Stage 5: Evaluation

Multiple evaluation strategies are implemented:

```bash
# Compare YOLOv8 vs YOLO26 on 1000 random images (GPU batched)
python evaluation/compare_models.py

# Evaluate all ensemble strategies on test set
python evaluation/hybrid_evaluation.py

# Cross-validate CSV features against YOLO re-extraction
python evaluation/validate_csv_with_yolo.py
```

**Ensemble Strategies Evaluated:**
| Strategy | Formula | Description |
|:---|:---|:---|
| YOLO Only | `yolo_pred = 1` | All YOLO detections are mines |
| YOLO + RF | `rf.predict(features)` | RF filters YOLO detections |
| YOLO + LR | `lr.predict(features)` | LR filters YOLO detections |
| Unanimous | `RF=1 AND LR=1` | Both must agree |
| Majority | `RF=1 OR LR=1` | Either model suffices |
| Soft Weighted | `0.7×RF_prob + 0.3×LR_prob > 0.5` | Weighted probability fusion |

### Stage 6: Deployment

The system features a **FastAPI backend** and a modern **Next.js** web application with advanced interactive UI capabilities, alongside a C++ TensorRT template for edge deployment:

```bash
# 1. Start the FastAPI Backend
uvicorn backend.main:app --reload --port 8000

# 2. Start the Next.js Frontend Dashboard
cd nextjs-app
npm run dev
```

**Modern UI (`nextjs-app`):**
- **Glassmorphism Design:** Modern aesthetic with refined visual hierarchy.
- **Stage 3 Grid Animations:** Row-by-row animated grid scanning that synchronously reveals cells and highlights suspicious regions.
- **Detection Details Card:** Professional crosshair overlay for precise bounding box visualization and detailed Region of Interest (ROI) mapping.
- **Real-time Pipeline:** Visualizes the 7-stage analytical process directly in the browser.

**`backend/main.py`** — FastAPI Server:
- Handles high-performance asynchronous REST endpoints (`/detect`, `/pipeline`).
- Processes image uploads, coordinates YOLOv8, extracts features, and runs Random Forest & Logistic Regression inference.

**Edge Deployment:**
**`inference_tensorrt.cpp`** — C++ template for edge deployment via TensorRT.

---

## 🔬 Extracted Features

Each bounding box region is analyzed to extract 10 handcrafted physical and thermal features:

| # | Feature | Description | Formula |
|:---:|:---|:---|:---|
| 1 | **Area** | Physical size in pixels² | `h × w` |
| 2 | **Circularity** | Shape compactness (mines ≈ circular) | `4π × contour_area / perimeter²` |
| 3 | **Mean Intensity** | Average thermal signature (grayscale mean) | `mean(gray_crop)` |
| 4 | **Thermal Contrast** | Temperature difference from background | `\|mean_object − mean_background\|` |
| 5 | **Edge Density** | Structural complexity (Canny edges / area) | `count(Canny_edges > 0) / area` |
| 6 | **Intensity Std Dev** | Thermal uniformity (mines are more uniform) | `std(gray_crop)` |
| 7 | **Aspect Ratio** | Width-to-height ratio (mines ≈ 1.0) | `w / h` |
| 8 | **Thermal Gradient** | Edge sharpness via Sobel operators | `mean(√(Sobel_x² + Sobel_y²))` |
| 9 | **Max/Min Ratio** | Extreme thermal contrast within region | `max(gray) / (min(gray) + ε)` |
| 10 | **Relative Size** | Object area as fraction of image area | `area / image_area` |

> **Background Region** is defined as a proportional ring (20% of object size, min 5px) around the bounding box, excluding the object itself.

---

## 🤖 Models

### Deep Learning Models

| Model | Architecture | Training | Weight Size | mAP@50 |
|:---|:---|:---|:---:|:---:|
| YOLOv8 (`last.pt`) | YOLOv8 Medium | Local GPU | ~50MB | 88.3% |
| YOLO26 (`best.pt`) | YOLO26 Small | Kaggle T4 GPU | 20.3MB | 91.9% |

### Classical ML Models

| Model | Algorithm | Features | Accuracy |
|:---|:---|:---:|:---:|
| `logistic_regression_model.pkl` | Logistic Regression | 10 | ~75% |
| `random_forest_model.pkl` | Random Forest (200 trees) | 10 | ~80.7% |

---

## 📈 Benchmarking Results

### YOLOv8 vs YOLO26 Comparison (1000 images, GPU)

| Metric | YOLOv8 (Model 8) | YOLO26 (Model 26) |
|:---|:---:|:---:|
| Inference Speed (FPS) | ~150 | ~204 |
| Total Detection Count | — | — |
| Exact Match Accuracy | — | Higher |
| Error Rate | Higher | Lower |

### Class-Specific YOLO26 Performance

| Class | mAP@50 | Analysis |
|:---|:---:|:---|
| `at_plastic` | 99.1% | Largest spatial footprint → easiest for CNN |
| `at_metal` | 97.4% | Strong thermal signature |
| `ap_plastic` | 86.7% | Small object, fewer pixels at 640×640 |
| `ap_metal` | 84.6% | Most challenging — small + low contrast |

---

## 🚀 Installation & Usage

### 1. Prerequisites
- **Python 3.12**
- **Node.js 18+ & npm**
- **Git**

### 3. Setup

```bash
# Clone the repository
git clone https://github.com/Hamzamn19/mayin-tarlasi.git
cd mayin-tarlasi

# Install Backend Dependencies
pip install -r requirements.txt

# Install Frontend Dependencies
cd nextjs-app
npm install
cd ..
```

### 🍎 macOS Specific Notes
If you are using a Mac (especially Apple Silicon M1/M2/M3):
- **Acceleration:** YOLOv8 will automatically attempt to use the **MPS (Metal Performance Shaders)** for acceleration. You don't need to change `device=0` (which is for NVIDIA CUDA) to use MPS in the Python scripts; it handles it internally, but you can explicitly use `device='mps'` in your code if needed.
- **OpenCV:** If you face issues with GUI windows, try: `pip install opencv-python-headless`.
- **Node.js:** Ensure you have the ARM64 version of Node.js installed for better performance.

### 4. Dataset & Weights
Note: The actual dataset and trained weights are large and may be excluded from the repository.
- **Weights:** Ensure `results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt` exists.
- **Dataset:** The `landmine_flat/` directory should contain the `train/`, `val/`, and `test/` splits if you intend to run the processing/training scripts.

### 4. Running the Application

To run the full interactive system:

**Terminal 1: Start Backend (FastAPI)**
```bash
python deployment/api.py
```

**Terminal 2: Start Frontend (Next.js)**
```bash
cd nextjs-app
npm run dev
```
The application will be available at `http://localhost:3000`.

### 5. Training & Evaluation
If you want to run the pipeline stages from scratch:
```bash
# Data Preparation
python data_processing/rename_dataset.py
python data_processing/split_raw_data.py

# Feature Extraction
python data_processing/extract_pipeline.py

# ML Training
python machine_learning/train_ml_models.py
```

### GPU Inference

```python
from ultralytics import YOLO

model = YOLO("results/runs/detect/Landmine_Detection_2026/YOLO26_S_Standard/weights/best.pt")
model.to("cuda")
results = model("thermal_image.jpg", conf=0.25, device=0)
```

---

## 🛠 Technologies Used

| Category | Technologies |
|:---|:---|
| **Deep Learning** | YOLOv8, YOLO26 (Ultralytics) |
| **Classical ML** | scikit-learn (Random Forest, Logistic Regression) |
| **Backend API** | FastAPI, Uvicorn, Python 3.12 |
| **Frontend UI** | Next.js 14, React 18, Tailwind CSS, Framer Motion, Zustand |
| **Computer Vision** | OpenCV (Otsu, Canny, contour analysis) |
| **Data Analysis** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Edge Deployment** | TensorRT, CUDA, C++ |
| **Training Compute** | Kaggle (Tesla T4 GPU, 16GB VRAM) |
| **Annotation Format** | Pascal VOC (XML) → YOLO (TXT) |

---

## 🙏 Acknowledgements

- **U.S. Army Counter Explosive Hazards Center (CEHC)** — For providing access to the landmine simulant data used in the AMLID dataset.
- **Kaggle** — For providing Tesla T4 GPU resources that enabled training the YOLO26 architecture.
- **Gallagher & Oughton (2023–2025)** — For the original AMLID research and baseline benchmarks.
- **Ultralytics** — For the YOLO framework and community support.

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@project{mayin_tarlasi_2026,
  title   = {Aerial Landmine Detection using Hybrid Deep Learning and Statistical Analysis},
  year    = {2026},
  school  = {Beykoz University},
  note    = {Estimation and Prediction Course Project}
}
```

---

<p align="center">
  <b>⚠️ This project is developed for academic and humanitarian research purposes only.</b><br>
  <i>Beykoz University — CME6102 Estimation and Prediction — April 2026</i>
</p>
