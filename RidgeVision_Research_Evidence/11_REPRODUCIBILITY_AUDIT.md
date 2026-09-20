# 11 — Reproducibility Audit: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Item-by-Item Reproducibility Matrix

Each critical reproducibility dimension has been examined against the actual repository files and classified as **COMPLETE**, **PARTIAL**, or **MISSING**.

| Audit Dimension | Status | Verified Evidence & Code Grounding | Reproducibility Assessment |
| :--- | :---: | :--- | :--- |
| **Dataset Availability** | **PARTIAL** | URLs to Kaggle sources (`sravani2006` and `abhiramshibaraya`) are provided in `datasets/README.md` and notebook cells. However, local files in `datasets/raw/` are absent from the git repository. A third party must manually download the Kaggle corpus. | The dataset is publicly downloadable from Kaggle, but raw images are not versioned locally in git. |
| **Preprocessing Pipeline** | **COMPLETE** | Implemented deterministically in `backend/ml/preprocessing/fingerprint.py` (`enhance_fingerprint`) and `backend/ml/feature_engineering/texture.py` (`UnifiedTextureExtractor`). | Exactly reproducible: all parameters (CLAHE clip 2.6, 8 Gabor angles, LBP P=8/R=1, 12 GLCM features) are hardcoded and deterministic. |
| **Code Implementation** | **COMPLETE** | Full Python codebase present: FastAPI backend, Keras architecture builders, evaluation runner, inference predictor, explainability modules, and test suite. | Complete source code is present and executable. |
| **Dependencies & Environment** | **COMPLETE** | Specified in `requirements.txt`, `backend/requirements.txt`, `.python-version` (3.11.9), and `runtime.txt`. | Version ranges for all primary libraries (TensorFlow, OpenCV, scikit-learn, scikit-image) are specified. |
| **Random Seeds** | **COMPLETE** | `SEED = 42` is explicitly set across all notebooks (`90-accuracy.ipynb`, `Part 1-3.ipynb`) and split utilities (`splits.py`). | Controlled randomness is enforced across NumPy, Python, and scikit-learn splits. |
| **Train / Test Split** | **PARTIAL** | Split algorithms are fully coded (`GroupShuffleSplit` in `splits.py` and `train_test_split` in `90-accuracy.ipynb`). However, a frozen CSV/JSON manifest listing exact sample-to-fold assignments is **MISSING** from disk. | Re-running `GroupShuffleSplit` with seed 42 will regenerate identical splits only if file iteration order remains completely invariant. |
| **Hyperparameters** | **COMPLETE** | Documented in `backend/ml/models/architecture.py`, `notebooks/`, and `RidgeVision_v2_Comprehensive_Master_Report.md` (Table 5.1). | Learning rates, Adam parameters, batch sizes, warmup epochs, fine-tuning epochs, and loss weights are fully specified. |
| **Hardware Specifications** | **COMPLETE** | Documented in `RidgeVision_v2_Comprehensive_Master_Report.md` (Section 5.1) and verified via notebook logs: Dual NVIDIA Tesla T4 (16GB VRAM each) on Kaggle Cloud. | Compute requirements are fully known. |
| **Model Configuration & Weights** | **COMPLETE** | Architecture builders defined in `architecture.py`. Trained weights present: `models/ridgevision_model.keras` (51.8 MB) and `ridgevisionnet_results/ridgevision_full_model.weights.h5` (176.3 MB). | Both the standalone Keras model and the weights file are available on disk. |
| **Evaluation Procedures** | **COMPLETE** | Comprehensive metrics implemented in `backend/ml/evaluation/metrics.py`, `benchmark_runner.py`, `robustness.py`, `calibration.py`, and `conformal.py`. | Multi-task metrics, bootstrap confidence intervals, McNemar tests, ECE, conformal prediction, and ANOVA are fully implemented. |

---

## 2. Summary Assessment
* **Overall Reproducibility Rating**: **HIGH (with Minor Data Download Dependency)**
* Another researcher can reproduce the pipeline and evaluate the pre-trained models immediately using the codebase and saved weights. To retrain from scratch, the researcher must download the 5,837-image Kaggle dataset from the provided URL.
