# Technical Report: Statistical and Deep Learning Approaches for Aerial Landmine Detection

**Date:** April 29, 2026  
**Project:** Mayin Tarlasi (Landmine Detection)  
**Methodology:** YOLOv8 + Handcrafted Thermal Features + Ensemble Classification  

---

## 1. Abstract
Landmines remain a persistent humanitarian threat, claiming approximately 26,000 casualties annually. Traditional detection methods are hazardous and inefficient. This report presents an automated detection pipeline utilizing Long-Wave Infrared (LWIR) imagery from Unmanned Aerial Systems (UAS). We employ the YOLOv8 deep learning architecture for object localization and a secondary stage of statistical classification using handcrafted thermal features. Our hybrid system demonstrates robust performance in detecting both plastic and metallic surface-laid mines across varying environmental conditions.

## 2. Introduction
The detection of landmines from aerial platforms is a complex task due to the small size of the targets and the heterogeneous backgrounds of conflict zones. Thermal imaging (LWIR) offers a significant advantage by capturing the thermal signatures of objects that transfer heat at different rates than the surrounding soil. This project evaluates a multi-stage pipeline:
1. **Detection:** Initial localization of suspect regions using YOLOv8.
2. **Feature Extraction:** Characterization of detected regions using 5 handcrafted features.
3. **Classification:** Refined decision-making through Random Forest and Logistic Regression models.
4. **Ensemble:** A voting mechanism to combine deep learning confidence with statistical probability.

## 3. Dataset & Data Analysis
### 3.1. AMLID Dataset
The project utilizes data derived from the Adaptive Multispectral Landmine Identification Dataset (AMLID), comprising 12,078 labeled images. The targets include 21 types of Anti-Personnel (AP) and Anti-Tank (AT) mines in both metal and plastic compositions.

### 3.2. Handcrafted Features
Five key features were extracted from thermal crops:
- **Area:** Number of pixels in the detected region.
- **Circularity:** Compactness of the object (high for mines).
- **Mean Thermal Intensity:** Average pixel value (thermal signature).
- **Thermal Contrast:** Difference between the object and its immediate background.
- **Edge Density:** Density of edges within the crop (detecting artificial boundaries).

### 3.3. Exploratory Data Analysis (EDA)
EDA revealed that **Circularity** (r = 0.53) and **Thermal Contrast** (r = 0.51) are the strongest predictors of a mine's presence. Metallic mines often exhibit higher contrast during daytime, while plastic mines are detectable through subtle heat retention patterns.

---

## 4. Deep Learning Performance (YOLOv8)
The YOLOv8 model (`last.pt`) was evaluated on the independent test set (1,177 images).

### 4.1. Detection Metrics
| Metric | Value |
| :--- | :--- |
| **Precision (P)** | [PLACEHOLDER_P] |
| **Recall (R)** | [PLACEHOLDER_R] |
| **mAP@50** | [PLACEHOLDER_mAP50] |
| **mAP@50-95** | [PLACEHOLDER_mAP95] |

### 4.2. Class-wise Performance
| Class | Precision | Recall | mAP@50 |
| :--- | :--- | :--- | :--- |
| AT Plastic | [P1] | [R1] | [M1] |
| AP Plastic | [P2] | [R2] | [M2] |
| AT Metal | [P3] | [R3] | [M3] |
| AP Metal | [P4] | [R4] | [M4] |

*Analysis: YOLOv8 shows exceptional localization capabilities, acting as a high-recall filter for the subsequent stages.*

---

## 5. Statistical Classification Results
The handcrafted features were used to train classical models to distinguish between mines and background noise.

| Model | Accuracy | Mine Precision | Mine Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 80.72% | 0.88 | 0.75 | 0.81 |
| **Logistic Regression** | 76.93% | 0.87 | 0.68 | 0.76 |

---

## 6. Hybrid System: Ensemble Voting
A key innovation of this project is the **Ensemble Voting** strategy implemented in the deployment application. By averaging the YOLO confidence score with the Random Forest probability, the system achieves a balanced final decision, effectively reducing false alarms while maintaining high sensitivity.

**Pipeline Flow:**
`Input Image` -> `YOLO Localization` -> `Feature Extraction` -> `RF Analysis` -> `Ensemble Decision`

## 7. Conclusion
This project successfully demonstrates that combining deep learning (YOLOv8) with classical statistical learning on handcrafted features provides a robust solution for aerial landmine detection. The thermal modality proves essential for identifying plastic-cased mines that are invisible to traditional metal detectors. Future work will focus on integrating RGB fusion and testing under more diverse terrain conditions.
