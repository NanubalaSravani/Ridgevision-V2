# RidgeVision AI — Current Experimental Version

## Research Problem
Determining human ABO blood group and Rhesus (Rh) factor non-invasively from digital optical fingerprint impressions, addressing the critical discrepancy between reported deep learning accuracies (~90%) and dermatoglyphic anthropological literature demonstrating only weak, population-level statistical correlations (p < 0.05).

## Objective
To build a scientifically rigorous, publication-grade biometric screening framework (LeakSafe-CGN / RidgeVisionNet) that eliminates partition leakage via pseudo-subject perceptual clustering, biologically decouples independent genetic loci (Chromosome 9 for ABO, Chromosome 1 for Rh), validates decisions via 3-tier causal explainability, and prevents overconfident false positives via conformal clinical abstention (`PREDICTION_WITHHELD`).

## Dataset
* **Primary Benchmark**: Kaggle `sravani2006/fingerprint-blood-group-classification-dataset` (5,837 images across 8 classes: A+: 402, A-: 1,009, AB+: 708, AB-: 761, B+: 652, B-: 741, O+: 852, O-: 712).
* **Prototype Training Set**: Kaggle `abhiramshibaraya/fingerprint-based-blood-group-detection` (5,837 images, 1,000 per class).
* **Donor Provenance**: Subject identifiers are NOT FOUND in raw uploads; clustered via 64-bit pHash.

## Preprocessing
Sequential 8-step pipeline: (1) Decode BGR, (2) Grayscale conversion (Rec.601), (3) Spatial resampling to 224x224 (cv2.INTER_AREA), (4) CLAHE contrast enhancement (clip=2.6, 8x8 tiles), (5) Gaussian denoise (3x3), (6) 8-orientation Gabor filter bank (17x17, sigma=4.0, lambda=10.0, gamma=0.55), (7) Max-response orientation pooling, (8) Min-max normalization to uint8 [0, 255] and float32 [0.0, 1.0]. Combined with a canonical 30-dim texture vector (LBP10 + GLCM12 + Ridge8).

## Data Split
* **v1 Split**: 70/15/15 stratified random split at the individual image level (5,600 train / 5,837 val / 5,837 test).
* **v2 Split**: 3-Fold Grouped Cross-Validation based on 64-bit perceptual-hash (`pHash`) pseudo-subject clustering (Fold 1: 1,946; Fold 2: 1,946; Fold 3: 1,945). Guarantees zero cluster overlap between training and testing.
* **Sanity Control**: Randomized-label control where labels are permuted; accuracy collapsed to 12.61% (~12.5% chance), proving zero leakage.

## Model
* **Primary Architecture (LeakSafe-CGN)**: EfficientNetB0 backbone + CBAM attention (channel r=8, spatial 7x7) + Dense texture projection (64 units) + LayerNorm + Multimodal Concatenation (1344) + Sigmoid Gated Fusion + Shared Latent Dense (256, Dropout 0.35) + Decoupled Heads: ABO (4-way Softmax), Rh (Binary Sigmoid), Flat Joint (8-way Softmax).
* **Standalone Architecture (RidgeVisionNet)**: Deterministic on-device RidgeOrientationField (block_size=8) + ROAM attention + Adaptive Gated Fusion. Total parameters: 10,390,288 (8,378,217 trainable, 2,012,071 non-trainable).

## Training Configuration
Two-stage optimization:
* Stage 1 (Warmup): 8 epochs, top heads trainable, Adam (LR = 1e-3, beta1=0.9, beta2=0.999).
* Stage 2 (Fine-Tuning): 30 epochs, top 40 backbone layers trainable, Adam (LR = 1e-5), ReduceLROnPlateau (factor=0.3, patience=4), EarlyStopping (patience=7, restore_best_weights=True).
* Loss: Joint weighted loss (0.50 * CE_ABO + 0.30 * BCE_Rh + 0.20 * CE_flat).
* Hardware: Dual NVIDIA Tesla T4 GPUs (16GB VRAM each), mixed_float16, XLA jit_compile.

## Experiments
26 distinct experiments documented, spanning prototype single models (B0, B3), soft-voting ensembles, 3-fold CV across 10 baseline architectures, 8 architectural ablation variants, 6-axis physical perturbation robustness tests, ANOVA feature effect size analyses, and conformal calibration benchmarks.

## Best Verified Result
* **Prototype Ensemble (5,837 Held-Out Test Samples)**:
  * Accuracy: **91.10%** (1,074 / 5,837; reported rounded as **90%**)
  * Macro Precision: **0.90**, Macro Recall: **0.90**, Macro F1: **0.90**
* **LeakSafe-CGN Reference Benchmark**:
  * Accuracy: **91.10%** (+/- 0.42%)
  * Macro F1: **0.909**
  * Expected Calibration Error (calibrated): **0.041**

## Baseline Results
Evaluated on identical 3-fold grouped splits:
* MobileNetV2: **91.07%** (+/- 0.31%)
* RidgeVisionNet (Ours): **89.96%** (+/- 0.64%)
* ConvNeXt-Tiny: **89.91%** (+/- 0.49%)
* EfficientNetB0 (Plain): **89.82%** (+/- 0.88%)
* DenseNet121: **88.64%** (+/- 0.26%)
* InceptionV3: **85.04%** (+/- 0.07%)
* Plain CNN from scratch: **82.23%** (+/- 1.38%)
* ResNet50: **79.94%** (+/- 0.66%)
* Random Forest (Texture): **38.63%** (+/- 0.31%)
* SVM RBF (Texture): **24.65%** (+/- 1.24%)
* Linear SVM (Texture): **52.40%** (+/- 1.24%)

## Evaluation
* Multi-task metrics: ABO 4-way, Rh 2-way, Flat 8-way.
* Calibration: Post-hoc Temperature Scaling ($T=1.365$), reducing ECE from $0.0842 \to 0.0412$.
* Uncertainty: Split Conformal Prediction ($1-\alpha = 0.90, \hat{q}_{90} = 0.724$).
* Abstention: Automatic `PREDICTION_WITHHELD` if $|C(X)| > 2$ or max confidence $< 0.50$.
* Explainability: Tier 1 Grad-CAM++, Tier 2 OAAS singularity alignment ($p_{\text{null}} < 0.05$), Tier 3 MCA minutiae inpainting.

## Figures
* Fig 1: 8-Class Ensemble Confusion Matrix Heatmap (embedded in `notebooks/90-accuracy.ipynb` Cell 17).
* Fig 2–13: 12 Synthetic Test Prints (`ridgevisionnet_results/synthetic_test_prints/`).
* Note: Training loss curves and dynamic XAI maps must be exported from saved weights.

## Current Limitations
1. No explicit donor IDs in source dataset; reliant on perceptual hash clustering.
2. Moderate class imbalance in 5,837 dataset (A+ underrepresented at 6.89%).
3. Lack of physical fingerprint segmentation mask (raw rectangular background retained).
4. Dual dataset usage between prototype (5,837 images) and benchmark (5,837 images).

## Reproducibility Status
* Code, hyperparameters, dependencies, model configuration, and evaluation metrics are **COMPLETE**.
* Dataset requires download from Kaggle URL (**PARTIAL**).
* Frozen row-by-row split manifest CSV is **MISSING** from disk.

## Missing Information
* Medical serology laboratory verification protocol: NOT FOUND IN PROJECT FILES.
* Optical scanner hardware model and DPI resolution: NOT FOUND IN PROJECT FILES.
* Participant demographic characteristics (age, sex, ethnicity): NOT FOUND IN PROJECT FILES.
