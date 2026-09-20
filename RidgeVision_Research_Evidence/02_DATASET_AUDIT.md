# 02 — Dataset Audit: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Identified Datasets Across the Project

The codebase references two primary datasets used across different development stages, plus two external comparative references:

### Dataset A: Primary v2 Audited Benchmark Dataset
* **Dataset Name**: Fingerprint Blood Group Classification Dataset
* **Dataset Source / Repository**: Kaggle
* **Dataset URL / Identifier**: [`https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset`](https://www.kaggle.com/datasets/sravani2006/fingerprint-blood-group-classification-dataset) (`sravani2006/fingerprint-blood-group-classification-dataset`)
* **Role in Project**: Primary corpus for v2 benchmarking, 3-fold cross-validation, ablation experiments, robustness stress testing, and ANOVA feature validation.

### Dataset B: Prototype v1 Balanced Dataset (from `90-accuracy.ipynb`)
* **Dataset Name**: Fingerprint Based Blood Group Detection
* **Dataset Source / Repository**: Kaggle
* **Dataset URL / Identifier**: `abhiramshibaraya/fingerprint-based-blood-group-detection` (referenced in `notebooks/90-accuracy.ipynb` Cell 4: `/kaggle/input/datasets/abhiramshibaraya/fingerprint-based-blood-group-detection/dataset`)
* **Role in Project**: Used to train the initial prototype Model 88 (`ridgevision_b0_88_style_best.keras`) and Model 91 (`ridgevision_b3_91_style_best.keras`) soft-voting ensemble.

### External References (from `datasets/README.md`)
* `rajumavinmar/finger-print-based-blood-group-dataset` (labeled ABO/Rh dataset)
* `ruizgara/socofing` (unlabeled SOCOFing biometric dataset referenced for ridge-quality benchmarking and domain adaptation)

### Local Repository Presence
* Local directory `datasets/raw/` is **EMPTY** in the git workspace (per `.gitignore`, large datasets kept out of repository).
* Generated synthetic benchmark prints: 12 images located in `ridgevisionnet_results/synthetic_test_prints/`.

---

## 2. Quantitative Sample Statistics

### Dataset A (5,837 Images — v2 Primary Benchmark):
* **Total Image Count**: **5,837**
* **Number of Subjects / Participants**: **NOT FOUND IN PROJECT FILES**  
  *(Subject/donor IDs were not provided in the raw Kaggle upload. To prevent identity leakage, the project engineered a 64-bit perceptual-hash `pHash` clustering protocol to infer pseudo-subject clusters).*
* **Class Distribution (8 Phenotypic Classes)**:
  * **A+**: 402 images (6.89%, class weight: 1.815)
  * **A−**: 1,009 images (17.29%, class weight: 0.723)
  * **AB+**: 708 images (12.13%, class weight: 1.031)
  * **AB−**: 761 images (13.04%, class weight: 0.959)
  * **B+**: 652 images (11.17%, class weight: 1.119)
  * **B−**: 741 images (12.69%, class weight: 0.985)
  * **O+**: 852 images (14.60%, class weight: 0.856)
  * **O−**: 712 images (12.20%, class weight: 1.025)
  * **Total**: **5,837 images** (100.00%)
* **Biological Sub-Cohort Disaggregation**:
  * **ABO Phenotypes (Chromosome 9)**:
    * Group A: 1,411 images (24.18%)
    * Group AB: 1,469 images (25.17%)
    * Group B: 1,393 images (23.87%)
    * Group O: 1,564 images (26.79%)
    * *ABO Cohort Summary*: Exhibits natural epidemiological equilibrium near ~25% each.
  * **Rhesus Factor (Chromosome 1)**:
    * Rh-positive (+): 2,614 images (44.78%)
    * Rh-negative (-): 3,223 images (55.22%)

### Dataset B (5,837 Images — Prototype v1 Balanced Cohort in `90-accuracy.ipynb`):
* **Total Image Count**: **5,837**
* **Number of Subjects**: **NOT FOUND IN PROJECT FILES**
* **Class Distribution**:
  * A+: 1,000 images (12.5%)
  * A−: 1,000 images (12.5%)
  * AB+: 1,000 images (12.5%)
  * AB−: 1,000 images (12.5%)
  * B+: 1,000 images (12.5%)
  * B−: 1,000 images (12.5%)
  * O+: 1,000 images (12.5%)
  * O−: 1,000 images (12.5%)
  * **Total**: **5,837 images** (100.0%, perfectly balanced, class weights: 1.0 for all classes)

---

## 3. Image Characteristics & Formats
* **File Formats**: Windows Bitmap (`.bmp`), Portable Network Graphics (`.png`), JPEG (`.jpg`, `.jpeg`). Verified in `backend/ml/training/dataset_manifest.py`:
  ```python
  IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
  ```
* **Raw Image Resolutions**: Variable across original uploads; standardized during pipeline ingestion to 224 x 224 pixels (and 300 x 300 for Model 91).
* **Color Channels**: Acquired as single-channel grayscale or 3-channel BGR; converted to single-channel 8-bit grayscale for enhancement and 3-channel normalized tensors for CNN backbones.

---

## 4. Data Collection & Provenance Information
* **Sensor / Scanner Specifications**: **NOT FOUND IN PROJECT FILES** (Scanner manufacturer, optical vs capacitive sensor type, DPI resolution, and capture interface are unrecorded in dataset metadata).
* **Participant Demographics**: **NOT FOUND IN PROJECT FILES** (Age, biological sex, ethnicity, and geographic origin of donors are undocumented).
* **Ground-Truth Serological Verification**: **NOT FOUND IN PROJECT FILES** (The medical serology laboratory protocol used to establish ground truth blood groups is not documented in the Kaggle source).
* **Missing Values**: None identified in class labels; all images are labeled into one of the 8 canonical classes.
* **Duplicate / Near-Duplicate Information**: Audited in v2 via 64-bit perceptual hashing (`pHash`) and Structural Similarity (SSIM). Pairs exhibiting perceptual similarity > 0.82 are flagged as identical donor finger impressions.

---

## 5. Dataset Limitations Visible from Project Files
1. **Donor Provenance Void**: Lack of explicit donor/participant IDs makes true subject-level splitting impossible without heuristic clustering (`pHash`).
2. **Multiple Impressions from Same Individual**: Without donor IDs, public datasets inevitably contain multiple captures from the same individual fingers, creating a severe partition leakage vulnerability if split purely at the image level.
3. **Class Imbalance in Primary Dataset**: In Dataset A (5,837 images), class A+ is significantly underrepresented (N=402, 6.89%) relative to A− (N=1,009, 17.29%), requiring inverse-frequency class weighting (w_A+ = 1.815 vs w_A- = 0.723).
4. **Dual Dataset Usage Across Phases**: The transition from the 5,837-image balanced dataset (`abhiramshibaraya`) in v1 to the 5,837-image imbalanced dataset (`sravani2006`) in v2 creates an experimental discrepancy that must be explicitly acknowledged in the research paper.
