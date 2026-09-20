# 01 — Project Overview: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Project Name
* **Primary Project Name**: **RidgeVision AI**
* **Framework Iteration Names Found in Repository**:
  * **RidgeVision AI v1**: Prototype dual-branch / ensemble framework (Model 88 & Model 91)
  * **RidgeVision AI v2 (LeakSafe-CGN)**: Hierarchical Multi-Task Network with Conformal Safety Abstention
  * **RidgeVisionNet**: Standalone end-to-end architecture with on-device deterministic ridge orientation tensor and ROAM attention

---

## 2. Research Objective
To investigate whether digital optical fingerprint impressions can serve as a non-invasive biometric phenotype proxy for determining human ABO blood groups and Rhesus (Rh) factors, while establishing methodological rigor through:
1. Eliminating near-duplicate partition contamination via perceptual-hash (`pHash`) clustering audits.
2. Decoupling the independent genetic loci of ABO antigens (Chromosome 9q34.2) and Rh factor (Chromosome 1p36.11) into hierarchical multi-task classification heads.
3. Integrating domain-grounded explainability (Grad-CAM++, Orientation-Attention Alignment, and Minutiae-Causal Attribution).
4. Providing distribution-free uncertainty quantification via Split Conformal Prediction with a clinical abstention reject option (`PREDICTION_WITHHELD`).

---

## 3. Research Problem
1. **The Biological Disconnect**: Clinical and anthropological literature (e.g., Susmiarsih et al., 2016; Bharadwaja et al., 2004) documents only weak, population-level statistical skews between primary dermatoglyphic pattern types (loops, whorls, arches) and ABO blood phenotypes (p < 0.05). No clinical study supports that a single fingerprint carries sufficient information for deterministic individual 8-way ABO/Rh classification at ~90% accuracy.
2. **Methodological Vulnerabilities in Existing AI Works**: Prior computer vision studies reporting high accuracy frequently suffer from:
   * **Uncontrolled data leakage**: Absence of donor IDs leads to identical or near-duplicate impressions appearing in both training and test partitions.
   * **Conflated biological targets**: Monolithic 8-class softmax formulations treat genetic substitutions symmetrically, ignoring independent chromosomal inheritance.
   * **Heuristic XAI**: Using Sobel edge maps as proxies for neural saliency without auditing network attention.
   * **Forced uncalibrated predictions**: Forcing predictions on degraded or ambiguous impressions without uncertainty bounds or abstention options.

---

## 4. What the Existing System Does
The existing software pipeline:
1. Ingests a digital fingerprint image (file upload or live webcam capture via Web UI, or programmatic API request).
2. Executes standardized image preprocessing: grayscale conversion, spatial resampling to 224x224 (or 300x300 in v1 B3), Contrast-Limited Adaptive Histogram Equalization (CLAHE), Gaussian smoothing, and multi-orientation Gabor filtering.
3. Extracts two parallel feature streams:
   * **Deep Spatial Representations**: Extracted via an EfficientNetB0 backbone reinforced by trainable Convolutional Block Attention Modules (CBAM).
   * **Biometric Texture Descriptors**: Computes a canonical 30-dimensional biometric texture vector spanning Local Binary Patterns (LBP), Gray-Level Co-occurrence Matrix (GLCM), and ridge morphology metrics.
4. Performs learned gated multimodal fusion between deep convolutional features and texture descriptors.
5. Emits calibrated class probabilities across:
   * 4-way ABO classification (A, B, AB, O)
   * 2-way binary Rh classification (+, -)
   * Joint 8-way blood group classification (A+, A-, AB+, AB-, B+, B-, O+, O-)
6. Gates predictions through a Split Conformal Predictor with Temperature Scaling:
   * Outputs guaranteed marginal coverage prediction sets (1 - alpha = 0.90).
   * Automatically triggers clinical abstention (`PREDICTION_WITHHELD`) if prediction set size > 2 or top confidence < 0.50.
7. Generates a 3-tier explainability package:
   * **Tier 1**: Higher-order Grad-CAM++ saliency heatmap.
   * **Tier 2**: Orientation-Attention Alignment Score (OAAS) with spatial permutation null-model control.
   * **Tier 3**: Minutiae-Causal Attribution (MCA) using crossing-number extraction and localized inpainting ablation.

---

## 5. Main Workflow / Pipeline
```
Raw Fingerprint Image (Disk / Upload / API)
                ↓
decode_image() [OpenCV cv2.imdecode as BGR uint8]
                ↓
enhance_fingerprint() [Grayscale → 224×224 → CLAHE(clip=2.6, 8×8) → GaussianBlur(3×3) → 8-Dir Gabor Bank → Max-Response]
                ↓
        ┌───────┴───────────────────────────────────────────┐
        ▼                                                   ▼
Deep Spatial Tensor Branch                          Unified Texture Extractor
(224×224×3 float32, EfficientNetB0 + CBAM)          (30-dim canonical vector: LBP10 + GLCM12 + Ridge8)
        └───────┬───────────────────────────────────────────┘
                ↓
Multimodal Concatenation (1344-dim) & Learned Sigmoid Gating
                ↓
Shared Latent Representation (Dense 256, ReLU, Dropout 0.35)
                ↓
Hierarchical Decoupled Heads:
   ├── Head 1: ABO Group (4-way Softmax, Chr 9)
   ├── Head 2: Rh Factor (Binary Sigmoid, Chr 1)
   └── Head 3: Joint Blood Group (8-way Softmax)
                ↓
Uncertainty Governance:
   ├── Stage 1: Temperature Scaling Calibration (T = 1.365)
   └── Stage 2: Split Conformal Predictor (alpha = 0.10, coverage >= 90%)
                ↓
Clinical Safety Filter:
   ├── IF |C(X)| <= 2 and Max Conf >= 0.50 → ACCEPTED (Return Prediction & Sets)
   └── IF |C(X)| > 2 or Max Conf < 0.50   → PREDICTION_WITHHELD (Clinical Abstention)
                ↓
3-Tier Explainability Engine:
   ├── Tier 1: Grad-CAM++ Saliency Heatmap
   ├── Tier 2: OAAS Singularity Alignment & 100-permutation null test
   └── Tier 3: Minutiae-Causal Attribution (MCA) inpainting ablation
```

---

## 6. Input and Output
* **Input**:
  * Digital optical fingerprint image file.
  * Supported formats: PNG, JPG, JPEG, WEBP, BMP.
  * Input tensor dimension to neural models: (1, 224, 224, 3) float32, along with auxiliary (1, 30) float32 texture vector.
* **Output**:
  * JSON diagnostic response (or interactive UI display) containing:
    * Predicted blood group phenotype (e.g., `AB-`)
    * Confidence percentage (e.g., `92.4%`)
    * Clinical decision status: `ACCEPTED` or `PREDICTION_WITHHELD`
    * Conformal prediction set C(X) and marginal coverage guarantee (90%)
    * Hierarchical head probabilities for ABO (A, B, AB, O) and Rh (+, -)
    * 30 extracted biometric texture metrics
    * Base64-encoded visual explanation maps for Tiers 1, 2, and 3.

---

## 7. Technologies Used
* **Programming Language**: Python 3.11.9 (specified in `.python-version` and `runtime.txt`)
* **Core Deep Learning Framework**: TensorFlow / Keras (>=2.16, <3.0; notebook execution verified on TensorFlow 2.19.0)
* **Image Processing & Computer Vision**: OpenCV (`opencv-python-headless >= 4.10, < 5.0`), scikit-image (`>= 0.25, < 1.0`), Pillow (`>= 11.0, < 12.0`)
* **Scientific Computing & Statistics**: NumPy (`>= 1.26, < 3.0`), SciPy (`>= 1.11, < 2.0`), scikit-learn (`>= 1.4, < 2.0`)
* **Backend API & Web Server**: FastAPI (`>= 0.115, < 1.0`), Uvicorn (`>= 0.34, < 1.0`), python-multipart (`>= 0.0.20, < 1.0`), HTTPX (`>= 0.28, < 1.0`)
* **Testing Framework**: pytest (`>= 8.3, < 9.0`)
* **Frontend Web Application**: HTML5, Vanilla Modern JavaScript (ES6+), Vanilla CSS3 (custom responsive dark-mode glassmorphic interface, zero external framework dependencies)

---

## 8. Current Implementation Status
* **Core Machine Learning Models**:
  * Model binary present: `models/ridgevision_model.keras` (51.8 MB)
  * Model weights present: `ridgevisionnet_results/ridgevision_full_model.weights.h5` (176.3 MB)
* **Evaluation & Benchmark Artifacts**:
  * 3-fold cross-validation baseline comparison summary and fold results (`baseline_comparison_summary.json`, `baseline_comparison_fold_results.json`)
  * 8-variant architectural ablation study results (`ablation_results.json`)
  * 6-axis physical perturbation robustness evaluation results (`robustness_results.json`)
  * Full-scale ANOVA statistical validation across 10 texture metrics on 876 samples (`statistical_validation_full_scale.json`)
  * Full computational parameter cost analysis (`compute_cost.json`)
* **API & Deployment**:
  * FastAPI server functional with endpoints `/health`, `/predict`, and frontend static file serving
  * Deployed demonstrations documented on Render and Hugging Face Spaces
* **Research Documentation**:
  * Comprehensive master technical specification (`RidgeVision_v2_Comprehensive_Master_Report.md`)
  * Initial journal manuscript draft (`RidgeVision_v2_Manuscript.md`)
  * Research strengthening plan (`RidgeVision-AI-v2-Research-Plan.md`)
