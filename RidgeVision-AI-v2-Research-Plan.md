# RidgeVision AI → RidgeVision AI v2: A Critical Research Strengthening Plan

*Based on a direct code review of `Ridgevision-ai-main` (backend, models, notebooks, README) — extending the conference baseline into a publication-grade research framework.*

---

## Before anything else: the honest headline finding

Review of the actual codebase reveals three core findings that dictate this entire plan:

1. **When no trained `.keras` file is present, the deployed predictor fabricates a prediction.** `RidgeVisionPredictor._research_mode_probabilities()` builds a seeded pseudo-random score from a SHA-256 hash of the image bytes, dressed up with `sin()`/`cos()` weightings of texture features, and returns it labeled `"inference_mode": "research_mode"`. It uses LBP/GLCM/ridge features, but the class decision is dominated by a hash seed. This serves as a UI fallback but represents a fatal risk if confused with real model output.
2. **Tier-1 "Grad-CAM" is not Grad-CAM.** `heuristic_grad_cam_b64()` in `backend/ml/explainability/grad_cam.py` computes a Sobel-gradient edge-magnitude map. It does not touch network weights, activations, or gradients. It highlights high-frequency edges on *any* image regardless of model predictions.
3. **Tier-2 (OAAS) inherits Tier-1's limitation.** Because Tier-1 is an edge detector, Tier-2 correlates an edge map with a ridge-orientation field—two classical image processing metrics. It does not evaluate neural attention alignment with anatomy.

**Dermatoglyphics Literature Baseline (Gap D)**: Peer-reviewed dermatoglyphic literature (Susmiarsih et al. 2016; Bharadwaja et al. 2004) supports only **weak, population-level, small-effect associations** between coarse pattern class (loop/whorl/arch) and ABO group ($p < 0.05$). No literature supports that a single fingerprint print carries enough information to predict an individual's 8-way ABO/Rh status at ~90% accuracy. Bridging and rigorously testing this discrepancy is the central scientific problem of this research.

---

# PART 1 — Reconstructing the Existing Pipeline (from Code)

```
Raw fingerprint image (upload / dataset folder)
        ↓
decode_image()  — cv2.imdecode → BGR array
        ↓
enhance_fingerprint()
  grayscale → resize 224×224 → CLAHE(clip=2.6, 8×8) → Gaussian blur (3×3)
  → Gabor filter bank (8 orientations, single scale) → max-response fusion
  → normalize to [0,1]
        ↓
Feature extraction (two divergent code paths in texture.py)
  Path A (UI/reporting): extract_texture_features()
     → LBP(P=16,R=2) hist(18) + GLCM(3 dist × 3 angle) + ridge density + Sobel strength + entropy
  Path B (Model input): extract_fusion_texture_vector() / extract_fusion_texture_vector_multilbp()
     → Multi-radius LBP, alternate GLCM angles, Canny ridge density, Otsu ratio, Laplacian variance
        ↓
Backbone / classifier execution:
  (i) Ensemble mode (if b0_88 + b3_91 exist): 224² and 300² fusion nets, fixed 0.5/0.5 softmax average.
  (ii) Single model mode (if ridgevision_model.keras exists): single EfficientNet fusion net.
  (iii) Research mode (fallback): hash-seeded synthetic scores.
        ↓
Explainability:
  Tier 1: Sobel edge magnitude (heuristic proxy).
  Tier 2: OAAS — correlation of Tier 1 edge map with orientation coherence field.
  Tier 3: MCA — crossing-number minutiae extraction → per-minutia inpaint & reinfer causal drop.
        ↓
JSON response payload → Frontend rendering
```

### Existing Components vs Code Realities

| Component | Repository Implementation | Actual Technology / Method |
| :--- | :--- | :--- |
| **Preprocessing** | Grayscale, resize, CLAHE, Gaussian blur | OpenCV CLAHE + GaussianBlur |
| **Enhancement** | 8-orientation Gabor bank, max-response | Single-scale, fixed-frequency Gabor |
| **Segmentation** | None | Raw rectangular crop (background retained) |
| **Feature Extraction** | LBP + GLCM + Ridge metrics in **two divergent paths** | `skimage.feature` + OpenCV |
| **Backbone** | EfficientNetB0 (frozen ImageNet) | `tf.keras.applications.EfficientNetB0` |
| **Attention** | CBAM class defined for loading; scratch builder uses single dense sigmoid gate | Keras custom layer |
| **Ensemble Fusion** | Fixed arithmetic mean: $0.5 \times p_{88} + 0.5 \times p_{91}$ | Unweighted average |
| **Explainability** | Tier 1 = Sobel edges; Tier 2 = Edge-orientation correlation; Tier 3 = Minutiae inpainting ablation | OpenCV edge maps + crossing number |
| **Evaluation Split** | Single random stratified 70/15/15 split on **images** | `train_test_split(..., stratify=labels)` |

---

# PART 2 — Reviewer-Style Weaknesses & Code Grounding

1. **Heuristic "Grad-CAM"**: Computes Sobel magnitude, not $\frac{\partial y_c}{\partial A^k}$ gradients. Fails to show model decision rationale.
2. **Silent Hash-Seeded Fallback**: If weights are missing during benchmarking, runs may silently log synthetic scores.
3. **Image-Level Train/Test Split**: Image-level splitting when multiple prints belong to the same finger/subject causes near-duplicate leakage.
4. **Accuracy Outruns Biology**: 90% accuracy on single prints contradicts weak population-level literature without a leakage audit.
5. **Absence of Randomized-Label Control**: Lacks a label-shuffled sanity check (accuracy must collapse to ~12.5% chance level).
6. **No Quality Gating / Segmentation**: Background scanner borders and artifacts are processed directly, risking non-biometric shortcut learning.
7. **Fixed 50/50 Ensemble Blend**: Arbitrary averaging rather than a learned or validation-tuned stacking gate.
8. **Lack of Classical Baselines**: No baseline comparison against SVM / Random Forest on handcrafted features or simple CNNs.
9. **No Component Ablation**: Lacks systematic verification of CNN-only, texture-only, and attention vs no-attention contributions.
10. **Uncalibrated Output Probabilities**: Class balancing is applied during training, but ECE and Brier scores are never evaluated.
11. **Single Dataset Source**: All models are trained on one Kaggle corpus with no external cross-dataset test.
12. **No Perturbation Robustness Suite**: No stress tests under rotation, Gaussian blur, sensor noise, or partial crops.
13. **Forced Prediction without Rejection**: Forces an 8-way prediction even on smudged or ambiguous prints.
14. **Lack of Statistical Significance Tests**: No bootstrap confidence intervals or McNemar's tests between model variants.
15. **Unanalyzed Confusion Asymmetries**: Misclassifications (e.g., A+ vs O-) are not evaluated against image quality or minutiae density.
16. **Feature Extraction Discrepancy**: Features displayed in the API response do not match the feature vector fed into the model.
17. **Reproducibility Gaps**: No data/model checksum manifests or pinned random seed split files.
18. **Conflated 8-Class Target**: Treats ABO (4 classes, Chr 9) and Rh (2 classes, Chr 1) as a single flat 8-class softmax.
19. **Unreported Acquisition Bias**: Provenance, demographics, and scanner specs of dataset subjects are undocumented.
20. **Disproportionate Headline Claims**: Prominent 90% claims in marketing outpace what Tier 1-3 explainability can substantiate.

---

# PART 3 — Research Gaps, Ranked

* **Gap A (Technical)**: Orientation field and ridge frequency are computed only for XAI, never fed forward as model input features; flat softmax ignores ABO/Rh biological decoupling.
* **Gap B (Data)**: Single-source dataset, unknown subject provenance, lack of grouped subject splits.
* **Gap C (Scientific/Evidentiary)**: No leakage audit, no label-shuffling control, absence of calibration metrics.
* **Gap D (Biological Grounding)**: Disconnect between weak population-level literature associations and individual-level deep learning predictions.
* **Gap E (Generalization)**: Unknown robustness under scanner shift, rotation, and compression.

**Priority Ranking**: $\text{Gap C} \approx \text{Gap D} > \text{Gap B} > \text{Gap A} > \text{Gap E}$.

---

# PART 4 — Ten Candidate Contributions

1. **Leakage-Audited Benchmark Protocol**: Grouped cross-validation (subject-level) + randomized-label baseline control.
2. **Hierarchical ABO/Rh Multi-Task Heads**: Decoupled 4-way ABO softmax + binary Rh sigmoid with joint multi-task loss.
3. **Physics-Guided Ridge Features as Model Input**: Feeding continuous orientation fields and local ridge frequency maps as direct input channels.
4. **Learnable Gabor Filter Bank**: End-to-end trainable orientation filters replacing static Gabor kernels.
5. **True Grad-CAM++ & Audited OAAS**: Backpropagation-derived class activation mapping paired with null-model alignment checks.
6. **Conformal Prediction & Abstention (Reject Option)**: Calibrated confidence intervals with formal coverage guarantees and ambiguous print rejection.
7. **Self-Supervised Pretraining on Unlabeled Corpora**: Contrastive pretraining on SOCOFing prints prior to blood-group fine-tuning.
8. **Systematic Perturbation & Domain Robustness Suite**: Standardized evaluation across rotation, blur, noise, and partial crops.
9. **Multi-Finger Aggregation Fusion**: Testing whether multi-finger pooling bridges the gap between individual and population priors.
10. **Causal Attribution-Gated Decision System**: Accepting predictions only when confidence is high and MCA attribution concentrates on true minutiae.

---

# PART 5 — Contribution Ranking & Selection

| Idea | Novelty | Tech. Depth | Research Value | Implementation | Publication Potential |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Leakage-Audited Protocol | 6/10 | 4/10 | 10/10 | 3/10 | 9/10 |
| 2. Hierarchical ABO/Rh Multi-Task | 6/10 | 5/10 | 7/10 | 4/10 | 7/10 |
| 3. Orientation Field as Model Input | 6/10 | 5/10 | 7/10 | 5/10 | 7/10 |
| 4. Learnable Gabor Bank | 5/10 | 6/10 | 5/10 | 6/10 | 5/10 |
| 5. True Grad-CAM++ & Audited OAAS | 6/10 | 5/10 | 8/10 | 4/10 | 8/10 |
| 6. Conformal Prediction & Abstention | 7/10 | 5/10 | 9/10 | 4/10 | 8/10 |
| 7. Self-Supervised Pretraining | 7/10 | 7/10 | 7/10 | 7/10 | 7/10 |
| 8. Robustness Suite | 4/10 | 3/10 | 7/10 | 3/10 | 6/10 |
| 9. Multi-Finger Fusion | 6/10 | 5/10 | 7/10 | 6/10 | 6/10 |
| 10. Causal Attribution-Gated Decision | 7/10 | 6/10 | 8/10 | 5/10 | 8/10 |

**Core Selection for v2**:
1. **#1 Protocol Audit**: The foundation required to substantiate all findings.
2. **#2 Hierarchical Multi-Task Head & #3 Domain-Guided Input**: Primary architectural novelties.
3. **#5 True Grad-CAM++ & #10 Causal Attribution**: Explainability rigor.
4. **#6 Conformal Abstention Layer**: Clinical decision-making utility.

---

# PART 6 — Literature Gap Analysis (Verified Sources)

| Citation | Year | Method & Cohort | Reported Result | Identified Limitation | v2 Differentiation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Phadke et al. (Springer)** | 2025 | CNN on Kaggle dataset (>6,000 images) | ~88% accuracy | Unstated splitting protocol; no leakage audit | First subject-independent grouped benchmark |
| **Swathi et al. (IJARSCT)** | 2024 | Standard CNN on Kaggle images | ~85-89% accuracy | Image-level random split; no XAI validation | Gradient-grounded XAI + causal verification |
| **ResearchGate Dermatoglyphic Study** | 2024 | CNN + statistical tests on 300 subjects | Weak population associations ($p < 0.05$) | DL accuracy disconnected from weak statistical priors | Direct evaluation of gap between priors and deep models |
| **ScienceDirect Model Survey** | 2026 | ResNet50 / VGG model evaluations | High nominal accuracy | No calibration, ECE, or conformal coverage | Temperature scaling + conformal prediction sets |
| **Susmiarsih et al. (Biosaintifika)** | 2016 | Manual loop/whorl/arch count (302 subjects) | Loops frequent in B (60.4%), whorls in O (40.5%) | Population-level frequencies only; non-predictive | Establishes biological upper bound for single prints |
| **Bharadwaja et al. (JIAFM)** | 2004 | Forensic ink prints (ABO correlation) | Statistically significant pattern skew | Small sample; non-individual classification | Grounding for why single-print 90% accuracy requires scrutiny |

---

# PART 7 — Novelty Honesty Check

| Component | Novelty Category | Justification |
| :--- | :--- | :--- |
| **CLAHE + Gabor Preprocessing** | Standard / Existing | Classic fingerprint enhancement pipeline. |
| **LBP + GLCM Texture Descriptors** | Standard / Existing | Well-documented hand-crafted texture features. |
| **EfficientNetB0 + CBAM Backbone** | Existing Combination | Established computer vision modules. |
| **Dual-Branch Image + Texture Fusion** | Incremental Combination | Standard multimodal tabular-image pattern. |
| **Hierarchical ABO/Rh Multi-Task Heads** | **Domain-Specific Architectural Novelty** | Decouples independent genetic loci (ABO vs Rh) with multi-task joint loss. |
| **Physics-Informed Ridge Orientation Input** | **Methodological Novelty** | Promotes orientation fields from post-hoc visualization to direct model input. |
| **True Grad-CAM++ with Null-Model Control** | **Interpretability Novelty** | Replaces Sobel heuristics with true gradients and tests against untrained networks. |
| **Minutiae-Causal Attribution (MCA)** | **Methodological Novelty (Modest)** | Inpainting ablation targeted at biologically grounded minutiae points. |
| **Conformal Prediction & Abstention Layer** | **Safety & Clinical Utility Novelty** | Provides distribution-free finite-sample coverage guarantees with reject options. |
| **Subject-Independent Leakage Audit** | **Experimental Novelty** | First rigorous audit establishing whether performance survives group isolation. |

---

# PART 8 — RidgeVision AI v2 (LeakSafe-CGN) Architecture & Formulations

```
Input Fingerprint Image (BGR)
        ↓
Unified Preprocessing: CLAHE + Gaussian Blur + Multi-Scale Gabor Bank
        ↓
Domain-Informed Ridge Decomposition:
  Continuous Orientation Tensor Field [cos(2θ), sin(2θ)] + Local Frequency Map
        ↓
Dual-Branch Representation:
  Branch A (Deep Spatial): EfficientNetB0 (fine-tuned) + Trainable CBAM Attention
  Branch B (Biometric Morphology): Unified Texture Extractor (LBP-P16/R2 + GLCM + Ridge Density)
        ↓
Learned Gated Fusion: Layer-norm Concatenation → Multi-Head Gating Dense Layer
        ↓
Hierarchical Multi-Task Prediction Heads:
  Head 1: ABO Group (4-way Softmax: A, B, AB, O)
  Head 2: Rh Factor (Binary Sigmoid: Rh+, Rh-)
  Head 3: Legacy Compatibility (8-way Flat Softmax)
        ↓
Uncertainty & Conformal Decision Layer:
  Temperature Scaling Logit Calibration → Non-conformity Score Evaluation
        ↓
Gate Condition:
  If Conformal Set Size > 2 or Calibrated Confidence < Threshold τ:
     → Status: "PREDICTION_WITHHELD" (Ambiguous / Degraded Print)
  Else:
     → Status: "ACCEPTED" (Prediction Set & Saliency Rendered)
        ↓
Explainability Pipeline (For Accepted Prints):
  Tier 1: True Grad-CAM++ Saliency Map
  Tier 2: Audited OAAS (Pearson r against Orientation Coherence)
  Tier 3: Minutiae-Causal Attribution (Ablation Confidence Delta)
```

### 1. Hierarchical Multi-Task Loss Formulation
To decouple the independent ABO and Rh genetic loci while maintaining backward compatibility:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{CE}}(y_{\text{ABO}}, \hat{y}_{\text{ABO}}) + \lambda_2 \mathcal{L}_{\text{BCE}}(y_{\text{Rh}}, \hat{y}_{\text{Rh}}) + \lambda_3 \mathcal{L}_{\text{CE}}(y_{\text{flat}}, \hat{y}_{\text{flat}})$$

Where:
* $\mathcal{L}_{\text{CE}}$ is Categorical Cross-Entropy across 4 ABO classes: $\{A, B, AB, O\}$.
* $\mathcal{L}_{\text{BCE}}$ is Binary Cross-Entropy for the Rh factor: $\{Rh^+, Rh^-\}$.
* Recommended hyperparameters: $\lambda_1 = 0.5$, $\lambda_2 = 0.3$, $\lambda_3 = 0.2$.

### 2. Grad-CAM++ Formulation
To capture localized ridge patterns better than standard Grad-CAM, compute weights $w_k^c$ for class $c$ and feature map $A^k$:

$$w_k^c = \sum_{i}\sum_{j} \alpha_{ij}^{kc} \cdot \text{relu}\left(\frac{\partial Y^c}{\partial A_{ij}^k}\right)$$

Where higher-order gradients weight spatial activations:

$$\alpha_{ij}^{kc} = \frac{\frac{\partial^2 Y^c}{(\partial A_{ij}^k)^2}}{2\frac{\partial^2 Y^c}{(\partial A_{ij}^k)^2} + \sum_{a}\sum_{b} A_{ab}^k \frac{\partial^3 Y^c}{(\partial A_{ij}^k)^3}}$$

Target hook layer: `top_conv` (EfficientNetB0 final convolutional projection).

### 3. Conformal Prediction Coverage Guarantee
For calibration dataset $(X_i, Y_i)_{i=1}^n$, compute non-conformity scores $s_i = 1 - \hat{P}(Y_i \mid X_i)$. The prediction set for a new print $X_{n+1}$ at significance level $\alpha = 0.10$ is:

$$C(X_{n+1}) = \left\{ y \in \mathcal{Y} : 1 - \hat{P}(y \mid X_{n+1}) \le \hat{q}_{\frac{\lceil (n+1)(1-\alpha) \rceil}{n}} \right\}$$

Guarantees finite-sample marginal coverage: $P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$.

---

# PART 9 — Experiment Design & Ablation Matrix

### Minimum Baseline Suite:
1. **Classical ML**: Linear SVM and Random Forest trained on unified handcrafted texture features.
2. **Standard CNN**: EfficientNetB0 without texture branch or attention mechanisms.
3. **v1 Reproduced**: Original architecture evaluated on the original image-level split.
4. **v1 Audited**: Original architecture evaluated on the **subject-independent grouped split**.
5. **v1 Label-Shuffled Control**: Original architecture trained on randomly permuted labels.
6. **v2 (LeakSafe-CGN)**: Full hierarchical architecture on the audited grouped split.

### Systematic Ablation Matrix:

| Experiment Code | Architecture Variant | Split Protocol | Evaluated Features | Target Research Question |
| :--- | :--- | :--- | :--- | :--- |
| `EXP-01` | Baseline SVM | Subject-Grouped | Unified Texture Vector | Minimum classical baseline performance |
| `EXP-02` | EfficientNetB0 Alone | Subject-Grouped | Enhanced Image Array | Deep image-only baseline |
| `EXP-03` | v1 Original Model | Random Image Split | Dual-Branch Fusion | Benchmark replication of conference claim |
| `EXP-04` | v1 Original Model | **Subject-Grouped** | Dual-Branch Fusion | **True baseline performance after eliminating leakage** |
| `EXP-05` | v1 Random-Label | Subject-Grouped | Dual-Branch Fusion | Sanity control: accuracy must collapse to ~12.5% |
| `EXP-06` | v2 Full (LeakSafe) | Subject-Grouped | Image + Texture + Orientation | Upper bound performance of proposed system |
| `EXP-07` | v2 w/o Orientation | Subject-Grouped | Image + Texture | Measures value of domain-guided ridge inputs |
| `EXP-08` | v2 w/o CBAM | Subject-Grouped | Image + Texture + Orientation | Tests contribution of spatial-channel attention |
| `EXP-09` | v2 Flat Softmax Only | Subject-Grouped | Image + Texture + Orientation | Evaluates impact of hierarchical ABO/Rh heads |
| `EXP-10` | v2 w/o Calibration | Subject-Grouped | Image + Texture + Orientation | Evaluates accuracy vs ECE on forced guesses |

---

# PART 10 — Perturbation & Robustness Testing

Evaluate all models on the held-out test split under synthetic real-world distortions:

1. **Angular Rotation**: $\pm 5^\circ, \pm 15^\circ, \pm 30^\circ$ (tests orientation-field stability).
2. **Additive Gaussian Noise**: $\sigma \in \{10, 25, 50\}$.
3. **Sensor Blur**: Gaussian filter kernels $k \in \{3, 5, 7\}$.
4. **Illumination Shifts**: Contrast scaling $\alpha \in \{0.7, 1.3\}$ and brightness offsets $\beta \in \{-30, +30\}$.
5. **Partial Capture (Occlusion)**: Masking $20\%$ to $40\%$ of the peripheral ridge area.
6. **JPEG Compression**: Quality factors $Q \in \{80, 50, 25\}$.

**Acceptance Criteria**: Graceful, monotonic degradation without inverted calibration (confidence must decrease as distortion increases).

---

# PART 11 — Generalization & Grouped Splitting Protocol

### Pseudo-Subject Clustering (When Metadata Lacks Subject IDs)
If the Kaggle dataset does not provide subject identifiers:
1. Compute **Perceptual Hashes (pHash)** and pairwise structural similarity (SSIM) across all unaugmented crops.
2. Extract minutiae coordinate graphs and compute pairwise Delaunay triangulation distance.
3. Cluster prints with mutual similarity $> 0.82$ into **Pseudo-Subject Clusters**.
4. Apply `GroupShuffleSplit` / `StratifiedGroupKFold` ($k=5$) over the identified clusters, guaranteeing that no print from a cluster appears in both train and test partitions.

---

# PART 12 — Statistical Validation

* **Bootstrap Confidence Intervals**: Compute 95% CIs via 1,000 resamples for Accuracy, Macro-F1, ECE, and Brier Score.
* **McNemar's Test**: Evaluate pairwise classification divergence between v1 and v2 on identical test splits ($p < 0.01$).
* **Paired Wilcoxon Signed-Rank Test**: Assess fold-wise metric gains across 5-fold grouped cross-validation.
* **Calibration Metrics**: Expected Calibration Error (ECE) computed with 15 uniform bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

# PART 13 — Explainable AI (Audited Three-Tier Stack)

1. **Tier 1 (Saliency)**: True Grad-CAM++ heatmaps derived from backpropagated target-class gradients.
2. **Tier 2 (Orientation-Attention Alignment)**: Pearson correlation $r$ between Grad-CAM++ saliency and orientation coherence:

$$r = \frac{\sum (A_{ij} - \bar{A})(C_{ij} - \bar{C})}{\sqrt{\sum (A_{ij} - \bar{A})^2 \sum (C_{ij} - \bar{C})^2}}$$

*Null-Model Check*: Compare $r_{\text{trained}}$ against $r_{\text{random}}$ (untrained network). Valid anatomical learning requires $r_{\text{trained}} > r_{\text{random}}$ with $p < 0.001$.
3. **Tier 3 (Minutiae-Causal Attribution)**: Crossing-number minutiae detection $\rightarrow$ localized inpainting ablation $\rightarrow$ empirical confidence drop $\Delta P = P_{\text{base}} - P_{\text{ablated}}$.

---

# PART 14 — Uncertainty Quantification & Conformal Abstention

* **Temperature Scaling**: Optimize scalar parameter $T > 0$ on validation logits $z$:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* **Abstention Policy**:
  * If $|C(X)| > 2$ (high multi-class ambiguity) OR $\max_i \hat{p}_i < 0.60$:
    * Tag: `status = "PREDICTION_WITHHELD"`
    * User message: *"Print quality or anatomical ambiguity exceeds reliable screening threshold."*
  * Else:
    * Tag: `status = "ACCEPTED"`

---

# PART 15 — Target Directory Structure

```text
backend/ml/
├── feature_engineering/
│   └── texture.py                  # UNIFIED: single extractor for UI reporting and model inputs
├── models/
│   ├── architecture.py             # UPDATED: Hierarchical ABO/Rh heads + trainable CBAM
│   └── ensemble.py                 # NEW: learned validation stacking weights
├── explainability/
│   ├── grad_cam.py                 # REPLACED: true Grad-CAM++ via backpropagation
│   ├── attention_alignment.py      # UPDATED: OAAS re-anchored to true gradient heatmaps
│   ├── minutiae.py                 # UNCHANGED: crossing-number extraction
│   └── causal_attribution.py       # UPDATED: gated by prediction acceptance status
├── uncertainty/                    # NEW MODULE
│   ├── calibration.py              # Temperature scaling & ECE computation
│   └── conformal.py                # Conformal prediction sets & abstention filters
├── training/
│   ├── dataset_manifest.py         # UPDATED: manifests with pseudo-subject cluster IDs
│   └── splits.py                   # NEW: GroupShuffleSplit & random-label generators
└── evaluation/                     # NEW MODULE
    ├── metrics.py                  # Multi-task metrics, bootstrap CIs, McNemar's test
    └── robustness.py               # Perturbation suite (noise, blur, rotation)
```

---

# PART 16 — Phase-by-Phase Roadmap & Milestones

### Phase 1: Data Integrity & Leakage Audit
* Construct dataset manifest with pseudo-subject clustering (`splits.py`).
* Run `EXP-04` (v1 on grouped split) and `EXP-05` (randomized-label control).
* *Milestone*: Establish the corrected baseline accuracy.

### Phase 2: Feature & XAI Overhaul
* Implement `UnifiedTextureExtractor` in `texture.py`.
* Implement true Grad-CAM++ in `grad_cam.py` and null-model control in `attention_alignment.py`.
* *Milestone*: Eliminate Sobel heuristics and reconcile feature extractor discrepancies.

### Phase 3: Model Architecture Upgrade
* Build hierarchical multi-task architecture in `architecture.py` (ABO 4-way, Rh 2-way, Flat 8-way).
* Train v2 network on grouped splits and tune stacking blend weights in `ensemble.py`.
* *Milestone*: Complete the ablation matrix (`EXP-06` to `EXP-10`).

### Phase 4: Uncertainty & Decision Safety
* Implement temperature scaling in `calibration.py` and conformal sets in `conformal.py`.
* Generate coverage-vs-abstention curves.
* *Milestone*: Produce calibrated outputs with formal error guarantees.

### Phase 5: Statistical Analysis & Paper Writing
* Run perturbation suite (`robustness.py`) and bootstrap significance tests (`metrics.py`).
* Populate final experiment tables and draft research manuscript.

---

# PART 17 — Experiment Tracking Matrix

| Experiment | Model Architecture | Split Protocol | Target Features | Accuracy | Macro F1 | ECE | Coverage (90%) | Abstention Rate |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Baseline-SVM` | Linear SVM | Grouped | Unified Texture | — | — | — | — | — |
| `Baseline-CNN` | EfficientNetB0 Alone | Grouped | Image Array | — | — | — | — | — |
| `v1-Replicated` | Dual-Branch Ensemble | Random Image | Image + Texture | — | — | — | — | — |
| `v1-Audited` | Dual-Branch Ensemble | **Grouped** | Image + Texture | — | — | — | — | — |
| `v1-Shuffled` | Dual-Branch Ensemble | Grouped (Shuffled) | Image + Texture | — | — | — | — | — |
| `v2-Full` | LeakSafe-CGN | Grouped | Full Multi-Modal | — | — | — | — | — |
| `v2-NoOrientation` | LeakSafe-CGN | Grouped | Image + Texture | — | — | — | — | — |
| `v2-NoCBAM` | LeakSafe-CGN | Grouped | Full Multi-Modal | — | — | — | — | — |
| `v2-FlatOnly` | Flat Softmax Head | Grouped | Full Multi-Modal | — | — | — | — | — |
| `v2-Uncalibrated` | LeakSafe-CGN | Grouped | Full Multi-Modal | — | — | — | — | — |

---

# PART 18 — Final Research Contributions (Draft Bullets)

1. **A Leakage-Audited Benchmark Protocol for Fingerprint-Based Phenotype Prediction**: Introduction of a grouped, pseudo-subject evaluation framework with a randomized-label sanity control, empirically quantifying split-induced performance inflation for dermatoglyphic blood-group prediction.
2. **Hierarchical Biological Multi-Task Architecture (LeakSafe-CGN)**: A biologically grounded dual-branch network decoupling independent ABO and Rh genetic loci into separate classification heads, augmented with domain-guided continuous orientation-field inputs and trainable CBAM attention.
3. **Gradient-Grounded, Null-Controlled Three-Tier Causal Explainability**: A verified XAI stack combining true Grad-CAM++ saliency, orientation-attention alignment validated against randomized-weight null controls, and minutiae-targeted inpainting causal attribution.
4. **Safety-Gated Biometric Screening via Conformal Prediction**: A distribution-free uncertainty framework providing finite-sample coverage guarantees and formal abstention options for degraded or anatomically ambiguous fingerprint impressions.

---

# PART 19 — Reviewer Simulation & Defense

### Reviewer 1 (Machine Learning Rigor)
* *Challenge*: *"How do you prove the network isn't memorizing sensor fingerprints or acquisition batch noise?"*
* *Defense*: Demonstrating near-chance collapse on randomized labels (`EXP-05`) alongside strictly grouped subject-independent splits (`EXP-04`).

### Reviewer 2 (Biometrics & Computer Vision)
* *Challenge*: *"Is Grad-CAM looking at true ridge morphology or generic boundary edges?"*
* *Defense*: Demonstrating that Tier-2 OAAS correlation on trained models significantly exceeds untrained null models ($p < 0.001$), paired with minutiae-specific causal ablation.

### Reviewer 3 (Clinical / Application Feasibility)
* *Challenge*: *"Why force an 8-way prediction when individual fingerprint dermatoglyphic associations are biologically weak?"*
* *Defense*: Integrating a conformal abstention mechanism that withholds judgment when prediction set size exceeds confident thresholds, shifting the objective from forced diagnosis to reliable screening assistance.

---

# PART 20 — Bottom Line & Immediate Action

The primary immediate priority is executing **Phase 1 (Data Integrity Audit)**: establishing the subject-grouped split and running the randomized-label control on v1. This determines whether the original ~90% is reproducible under rigorous validation or whether the paper's central contribution is uncovering and resolving dataset leakage. All downstream modeling builds directly on this foundation.
