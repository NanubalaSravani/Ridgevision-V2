# Methods vs. Codebase Audit (Step 3B)

I have verified the manuscript's "Materials and Methods" section against the codebase implementation in `backend/`. 

## ✅ Verified and Accurate Implementations

1.  **Data Preprocessing (`backend/ml/preprocessing/fingerprint.py`)**
    *   **Verified**: Grayscale conversion, resizing to $224 \times 224$, CLAHE with clip limit `2.6`, Gaussian filter kernel `(3,3)`, 8-orientation Gabor filter, maximum pooling, normalization to `[0, 1]`.
    *   **Verified**: Explicit absence of morphological foreground masking and singular point alignment.
2.  **Handcrafted Texture Representation (`backend/ml/feature_engineering/texture.py`)**
    *   **Verified**: 30-dimensional descriptor is precisely constructed via concatenation: 10-bin uniform LBP, 12 GLCM features (mean and std dev across 6 properties), and 8 ridge spatial features.
3.  **LeakSafe-CGN Architecture (`backend/ml/models/architecture.py`)**
    *   **Verified**: `EfficientNetB0` branch with `CBAM` attention $\rightarrow$ Global Average Pooling.
    *   **Verified**: Texture branch $\rightarrow$ `Dense(64, activation='relu')` $\rightarrow$ `LayerNormalization()`.
    *   **Verified**: Adaptive Gated Fusion concatenation $(1280 + 64 = 1344)$ followed by a learned sigmoid gate.
    *   **Verified**: Shared representation `Dense(256)` with `Dropout(0.35)`, culminating in three task heads (ABO Softmax, Rh Sigmoid, Joint 8-way Softmax).
4.  **Leakage-Mitigation Protocol (`backend/ml/training/splits.py`)**
    *   **Verified**: Images are hashed using a 64-bit pHash (DCT-based median thresholding truncated to 16 hex chars = 64 bits), and clustered into grouped splits to avoid near-duplicate leakage.
5.  **Calibration and Uncertainty (`backend/ml/uncertainty/calibration.py` & `conformal.py`)**
    *   **Verified**: Temperature scaling optimized via Negative Log-Likelihood (`Nelder-Mead` minimization).
    *   **Verified**: Split Conformal Predictor with a $1 - \alpha = 0.90$ marginal guarantee.
    *   **Verified**: Abstention policy triggered precisely when `len(prediction_set) > 2` or `max(probabilities) < 0.50`.
6.  **Explainability (`backend/ml/explainability/`)**
    *   **Verified**: `compute_gradcam_plus_plus` calculates 1st, 2nd, and 3rd order gradients for alpha weighting.
    *   **Verified**: `null_model_permutation_test` spatial tile permutation for OAAS statistical significance.
    *   **Verified**: `minutiae_causal_attribution` uses targeted in-painting (occlusion) on bifurcations and ridge endings to record prediction drops.

---

## ⚠️ Discrepancies Found

I identified two discrepancies between the instructions/manuscript text and the actual underlying Python code:

### 1. Cross-Validation Protocol
*   **Manuscript states**: "3-fold stratified cross-validation protocol... yielding approximately 3,891 training and 1,946 test images per fold."
*   **Codebase reality (`splits.py`, `benchmark_runner.py`)**: The `get_leaksafe_train_test_split` function utilizes a **single 80/20 train/test split** via `GroupShuffleSplit(n_splits=1, test_size=0.20)`. The codebase does *not* execute an iterative K-Fold CV loop. The prior `ablation_results.json` and `robustness_results.json` files were generated from this 80/20 split, not 3-fold CV.

### 2. Multi-Task Loss Weighting
*   **Manuscript states**: "ABO Cross-Entropy (0.30), Rh Binary Cross-Entropy (0.20), and Joint 8-class Cross-Entropy (0.50)."
*   **Codebase reality (`architecture.py`)**: The `get_multitask_loss_weights()` function assigns default weights as:
    ```python
    lambda_abo = 0.5   # Code (0.50) vs Text (0.30)
    lambda_rh = 0.3    # Code (0.30) vs Text (0.20)
    lambda_flat = 0.2  # Code (0.20) vs Text (0.50)
    ```

### Recommended Action
Would you like me to:
A) Keep the Methods section as-is (assuming the paper reflects a hypothetical/future final run), or
B) Edit the manuscript text to align with the actual code execution (i.e., change to an 80/20 grouped hold-out split and update the loss weights)?
