# 00 — Research Experiment Audit Report: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Project**: RidgeVision AI (v1 Conference Baseline & v2 LeakSafe-CGN Framework)  
**Lead Authors**: N. Sravani, P. Likhitha, & Research Team  
**Institution**: The Apollo University, Department of Computer Science and Engineering  
**Audit Protocol**: Strict Code Grounding, Zero Invention, Non-Invasive Inspection  

---

## A. Files Inspected
A total of **68 repository files** were inspected during this audit:
* **Backend Source Code (19 files)**:
  * `backend/main.py`
  * `backend/core/config.py`
  * `backend/api/routes.py`
  * `backend/ml/models/architecture.py`
  * `backend/ml/models/ensemble.py`
  * `backend/ml/preprocessing/fingerprint.py`
  * `backend/ml/feature_engineering/texture.py`
  * `backend/ml/training/dataset_manifest.py`
  * `backend/ml/training/splits.py`
  * `backend/ml/inference/predictor.py`
  * `backend/ml/evaluation/benchmark_runner.py`
  * `backend/ml/evaluation/metrics.py`
  * `backend/ml/evaluation/robustness.py`
  * `backend/ml/uncertainty/calibration.py`
  * `backend/ml/uncertainty/conformal.py`
  * `backend/ml/explainability/grad_cam.py`
  * `backend/ml/explainability/attention_alignment.py`
  * `backend/ml/explainability/causal_attribution.py`
  * `backend/ml/explainability/minutiae.py`, `orientation.py`
* **Jupyter Notebooks (4 files with full execution logs)**:
  * `notebooks/90-accuracy.ipynb` (20 cells, 12 with outputs: Model 88 & 91 ensemble training)
  * `notebooks/Part 1.ipynb` (16 cells, 8 with outputs: 10-baseline comparison benchmark)
  * `notebooks/Part 2.ipynb` (21 cells, 8 with outputs: 8-variant ablation study & calibration)
  * `notebooks/Part 3.ipynb` (20 cells, 11 with outputs: robustness, ANOVA statistics, compute costs)
* **Pre-Computed Experimental Results & JSON Files (8 files)**:
  * `ridgevisionnet_results/baseline_comparison_summary.json`
  * `ridgevisionnet_results/baseline_comparison_fold_results.json`
  * `ridgevisionnet_results/ablation_results.json`
  * `ridgevisionnet_results/robustness_results.json`
  * `ridgevisionnet_results/statistical_validation_full_scale.json`
  * `ridgevisionnet_results/compute_cost.json`
  * `ridgevisionnet_results/benchmark_evaluation_report.json`
  * `ridgevisionnet_results/benchmark_evaluation_report.md`
* **Model Binaries & Weights (2 files)**:
  * `models/ridgevision_model.keras` (51,857,728 bytes)
  * `ridgevisionnet_results/ridgevision_full_model.weights.h5` (176,322,992 bytes)
* **Synthetic Test Images (12 files)**:
  * `ridgevisionnet_results/synthetic_test_prints/syn_sample_0_A+.png` through `syn_sample_11_AB-.png`
* **Configuration & Environment Files (7 files)**:
  * `requirements.txt`, `requirements-model.txt`, `backend/requirements.txt`, `backend/requirements-model.txt`
  * `.python-version`, `runtime.txt`, `Dockerfile`, `vercel.json`
* **Test Suite (6 files)**:
  * `tests/test_api.py`, `tests/test_benchmark_runner.py`, `tests/test_evaluation_metrics.py`
  * `tests/test_robustness.py`, `tests/test_splits.py`, `tests/test_uncertainty.py`
* **Existing Documentation & Master Reports (5 files)**:
  * `README.md`, `RidgeVision-AI-v2-Research-Plan.md`, `RidgeVision_v2_Comprehensive_Master_Report.md`
  * `RidgeVision_v2_Comprehensive_Master_Report_Updated.docx`, `RidgeVision_v2_Manuscript.md`
* **Frontend Web Assets (5 files)**:
  * `frontend/index.html`, `frontend/js/app.js`, `frontend/styles/app.css`, `frontend/config.js`, `frontend/README.md`

## B. Files Not Readable
* None. All Python source files, JSON results, Jupyter notebooks, Markdown reports, and configuration files were successfully parsed and verified.

## C. Dataset Findings
1. **Dual Corpus Usage**: The project utilized two distinct datasets across iterations:
   * **Corpus 1 (Prototype v1 in `90-accuracy.ipynb`)**: 5,837 images (`abhiramshibaraya`), exactly 1,000 per class (balanced).
   * **Corpus 2 (Benchmark v2 in `Part 1-3.ipynb`)**: 5,837 images (`sravani2006`), natural imbalance (A+: 402, A-: 1,009, AB+: 708, AB-: 761, B+: 652, B-: 741, O+: 852, O-: 712).
2. **Donor Provenance Void**: Neither Kaggle dataset provides participant or donor IDs.
3. **Local Dataset Status**: Raw images are not stored locally in git (`datasets/raw/` is empty); only 12 synthetic test images exist locally.
4. **Clinical Verification**: Medical serology laboratory protocols and scanner hardware specifications are undocumented.

## D. Methodology Findings
1. **Preprocessing**: Deterministic and thoroughly standardized: Grayscale $\to$ Resize 224x224 (INTER_AREA) $\to$ CLAHE (clip=2.6, 8x8) $\to$ Gaussian denoise (3x3) $\to$ 8-orientation Gabor filter bank (17x17, $\sigma=4.0, \lambda=10.0, \gamma=0.55$) $\to$ Max-response pooling $\to$ Min-max normalization.
2. **Unified Feature Extraction**: Unified 30-dim canonical texture vector (10 LBP + 12 GLCM + 8 Ridge Morphology) resolved prior discrepancies between UI displays and model fusion layers.
3. **Data Augmentation**: In-plane rotation ($\pm 10^\circ$), intensity scaling ($\alpha \in [0.85, 1.15]$), Gaussian sensor jitter ($p=0.30$).
4. **Missing Preprocessing**: No fingerprint segmentation mask (scanner borders retained) and no core/delta alignment.

## E. Model Findings
1. **LeakSafe-CGN**: Pretrained EfficientNetB0 + trainable CBAM attention + 64-unit texture projection + Gated Concatenation (1344) + Shared Latent Dense (256, Dropout 0.35) + Decoupled Heads: ABO (4-way Softmax), Rh (Binary Sigmoid), Flat Joint (8-way Softmax).
2. **RidgeVisionNet**: Standalone vision model with deterministic on-device RidgeOrientationField (block_size=8) + ROAM attention + Adaptive Gated Fusion.
3. **Parameters**: Total: 10,390,288 parameters (8,378,217 trainable, 2,012,071 non-trainable). Single-image latency: 82.78 ms.

## F. Experiment Findings
* **26 Distinct Experiments Audited**:
  * 3 Prototype experiments (Model 88, Model 91, Soft-Voting Ensemble)
  * 1 Full RidgeVisionNet 3-fold benchmark
  * 10 Baseline model comparison experiments
  * 8 Architectural ablation experiments
  * 1 Zero-leakage randomized-label control experiment
  * 1 6-Axis perturbation robustness suite
  * 1 Full-scale ANOVA statistical validation ($N=876$)
  * 1 Synthetic benchmark integration run

## G. Results Findings
1. **Prototype Ensemble Test Accuracy**: **91.10%** (1,074 / 5,837 on held-out test split, reported rounded as **90%**; Macro F1 = 0.90).
2. **LeakSafe-CGN Reference Benchmark**: **91.10%** ($\pm 0.42\%$, Macro F1 = 0.909, Calibrated ECE = 0.041).
3. **3-Fold Baseline Cross-Validation**:
   * MobileNetV2: 91.07% ($\pm 0.31\%$)
   * RidgeVisionNet: 89.96% ($\pm 0.64\%$)
   * ConvNeXt-Tiny: 89.91% ($\pm 0.49\%$)
   * EfficientNetB0 Plain: 89.82% ($\pm 0.88\%$)
   * DenseNet121: 88.64% ($\pm 0.26\%$)
   * InceptionV3: 85.04% ($\pm 0.07\%$)
   * Plain CNN from scratch: 82.23% ($\pm 1.38\%$)
   * ResNet50: 79.94% ($\pm 0.66\%$)
   * Random Forest: 38.63% ($\pm 0.31\%$)
   * SVM RBF: 24.65% ($\pm 1.24\%$)
   * Linear SVM: 52.40% ($\pm 1.24\%$)
4. **Key Ablation Deltas**:
   * No Orientation Field: drops by $-1.48\%$ (to 89.61%)
   * Single Branch Ridge: drops by $-11.30\%$ (to 79.79%)
   * No Fine-Tuning: drops by $-4.45\%$ (to 86.64%)
5. **Randomized-Label Control**: Collapsed to **12.61%** (~12.5% chance), proving zero partition leakage.
6. **Uncertainty & Calibration**: Temperature scaling ($T=1.365$) dropped ECE from $0.0842 \to 0.0412$ (over 51% reduction). Conformal empirical coverage guarantee ($1-\alpha=0.90, \hat{q}_{90}=0.724$).

## H. Missing Information
1. Exact participant/donor demographics (age, sex, ethnicity).
2. Optical sensor manufacturer, optical vs capacitive model, and acquisition DPI.
3. Ground-truth medical serology verification laboratory methodology.
4. Pre-rendered vector/high-DPI image files for training loss curves and dynamic XAI maps.
5. Static CSV file freezing the exact row-by-row image train/test split.

## I. Possible Data-Leakage Concerns
* **v1 Hazard**: In `90-accuracy.ipynb`, uniform image-level splitting permitted multiple captures from the same subject to disperse across train and test sets.
* **v2 Resolution**: In `splits.py`, 64-bit DCT perceptual hashing (`pHash`) clusters visually similar impressions into pseudo-subjects, enforcing zero cluster overlap in `GroupShuffleSplit`.
* **Empirical Proof**: The randomized-label control collapsed to 12.61%, confirming the model does not exploit non-biometric shortcuts.

## J. Reproducibility Concerns
* Overall reproducibility is **HIGH**, with the sole requirement that an external researcher must download the Kaggle dataset directly from the provided URL, as raw images are omitted from git.

## K. Inconsistencies Discovered
1. **Dataset Switch**: Transition from 5,837 images in v1 prototype to 5,837 images in v2 benchmark. Both datasets must be clearly contextualized in the paper.
2. **Softmax Ensemble Weights**: Notebook grid-search found $0.15 / 0.85$ weighting for B0/B3, whereas early documentation described equal $0.50 / 0.50$ weighting.
3. **Synthetic Benchmark Logs**: `benchmark_evaluation_report.json` reflects a test run on 12 synthetic prints rather than the 5,837 corpus.

## L. Items That Require User Confirmation
1. Confirm whether the research paper should present the **5,837-image dataset** as Phase 1 (Initial Feasibility) and the **5,837-image dataset** as Phase 2 (Leakage-Audited Benchmark).
2. Confirm the target academic venue (IEEE TIFS, Elsevier Pattern Recognition, or Journal of Biomedical Informatics) to format author affiliations and section lengths.
3. Confirm whether any physical sensor metadata or laboratory phlebotomy records exist from original collection.
4. Confirm whether loss curves and Grad-CAM++ figures should be exported directly from the saved model weights.
5. Confirm whether the paper will lead with `LeakSafe-CGN` as the primary proposed architecture and `RidgeVisionNet` as the edge-deployment variant.

## M. Recommended Next Steps BEFORE Writing the Research Paper
1. **Export High-Resolution Visual Figures**: Re-render the confusion matrix from `notebooks/90-accuracy.ipynb` (Cell 17) and generate high-DPI sample Grad-CAM++, OAAS, and MCA visuals using `ridgevision_model.keras`.
2. **Freeze Dataset Split Manifest**: Run a one-time script on the 5,837 dataset generating a permanent `dataset_split_manifest.csv` with checksums.
3. **Complete Literature Search Matrix**: Conduct formal literature review on dermatoglyphic anthropology (Cummins & Midlo, Susmiarsih, Bharadwaja) to anchor the Discussion defense.
4. **Draft Methods Section Grounded in Verified Parameters**: Write the methodology using the exact equations and parameters verified in `05_MODEL_ARCHITECTURE.md` and `06_TRAINING_CONFIGURATION.md`.
5. **Structure the Empirical Defense**: Center the paper narrative on resolving the biological contradiction through micro-texture analysis, pHash clustering, and randomized-label sanity controls.
