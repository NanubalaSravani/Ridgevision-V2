# LeakSafe-CGN: A Hierarchical Multi-Task Network with Domain-Guided Attribution and Conformal Abstention for Fingerprint Biometric Screening

**Authors**: [Author Names Omitted for Review]  
**Affiliations**: Department of Computer Science and Engineering / Biomedical Informatics  

---

## Abstract

Recent machine learning studies report individual blood-group (ABO/Rh) classification from single fingerprint impressions with accuracies exceeding $88\% - 91\%$. However, peer-reviewed dermatoglyphic and anthropological literature demonstrates only weak, population-level statistical associations between coarse ridge patterns and ABO antigens ($p < 0.05$). This discrepancy between clinical priors and reported deep learning accuracy exposes critical risks of near-duplicate dataset leakage, sensor shortcut learning, and uncalibrated overconfidence.

In this paper, we introduce **LeakSafe-CGN** (Leakage-audited, Confidence-Gated Network), a comprehensive framework designed to establish methodological, architectural, and interpretability rigor in dermatoglyphic phenotype prediction. First, we establish a **leakage-audited evaluation protocol** using 64-bit perceptual-hash pseudo-subject clustering and a randomized-label sanity control to mitigate near-duplicate partition contamination, providing a negative-control check in which performance approached chance ($\approx 12.5\%$) under permuted labels. Second, we introduce a **biologically decoupled hierarchical architecture** that separates the independent genetic loci of ABO antigens (Chromosome 9) and Rh factor (Chromosome 1) into dedicated classification heads optimized via a joint multi-task objective, reinforced by continuous ridge-orientation guidance and trainable Convolutional Block Attention Modules (CBAM) [5]. Third, we replace classical edge heuristics with true **Grad-CAM++ [4] higher-order gradient saliency**, coupled with a spatial permutation null-model test for the Orientation-Attention Alignment Score (OAAS) and Minutiae-based Counterfactual Attribution (MCA). Finally, we implement **Split Conformal Prediction** with temperature scaling, converting forced overconfident guesses into distribution-free prediction sets targeting a 90% nominal marginal coverage ($1 - \alpha = 0.90$) and a computational abstention mechanism (`PREDICTION_WITHHELD`) for ambiguous impressions. Extensive ablations, 12-model baseline comparisons, and ANOVA effect size analyses ($\eta^2$) demonstrate the system's empirical behavior and structural attribution under the evaluated constraints, establishing a transparent foundation for further biometric evaluation.

**Index Terms**—Dermatoglyphics, Biometric Phenotyping, Deep Learning, Explainable AI, Grad-CAM++, Conformal Prediction, Multi-Task Learning, Leakage Audit.

---

## 1. Introduction

### 1.1 Problem Statement
The ability to rapidly and non-invasively classify human blood groups (ABO and Rh factor) carries significant implications for forensic identification, mass-casualty triage, and resource-constrained medical screening. Dermatoglyphic patterns—the epidermal ridges on palmar and plantar surfaces—develop early in human gestation and remain permanent [15]. Consequently, the potential association between fingerprint morphology and hematological phenotypes has been an enduring topic of research.

### 1.2 Existing Evidence
Clinical and anthropological studies (e.g., Susmiarsih et al. [1], Bharadwaja et al. [2]) have consistently investigated these associations. The reviewed literature indicates that correlations between classical dermatoglyphic macro-patterns (loops, whorls, arches) and ABO/Rh groups exist, but strictly as **weak, population-level statistical tendencies**. For example, loops may be modestly more prevalent in blood group B, but individual inter-class overlap is massive. Crucially, existing clinical evidence does not support the hypothesis that a single fingerprint image contains sufficient deterministic information to classify an individual's complete 8-way ABO/Rh status.

### 1.3 The Research Gap
In stark contrast to this clinical consensus, recent applied deep learning studies [9], [11], [12] have reported classification accuracies of $85\%$ to $91\%$ using flat 8-class convolutional neural networks trained on public fingerprint corpora. This discrepancy between anthropological priors and reported algorithmic accuracy exposes four critical methodological vulnerabilities in current computational approaches:
1. **Unmitigated Identity Leakage**: Single-source public datasets often lack donor identifiers. Random image-level train/test splits inadvertently cause impressions from the same donor to contaminate both partitions, allowing models to memorize near-duplicate artifacts rather than true phenotypic morphology [14].
2. **Biological Incoherence**: ABO antigens (Chromosome 9) and the Rhesus factor (Chromosome 1) are encoded at distinct chromosomal loci [13]. A flat 8-class softmax equates independent genetic mismatches, ignoring the underlying biological hierarchy.
3. **Overconfidence and Forced Prediction**: Standard classification networks force point predictions regardless of input degradation. In medical-adjacent domains, failing to quantify uncertainty or provide an abstention mechanism is hazardous.
4. **Unverified Spatial Attribution**: Several reviewed studies utilize qualitative edge-detection proxies or unverified saliency maps, providing false reassurance that the network is attending to anatomical ridge flow.

### 1.4 Specific Contributions
To bridge this gap between deep learning capability and scientific defensibility, this paper introduces **LeakSafe-CGN**, an exploratory biometric research framework. Rather than claiming unprecedented raw accuracy or a "first-ever" mechanism, our contribution centers on the integration of rigorous methodological constraints to reduce selected risks of experimental artifact and better assess whether observed performance is attributable to fingerprint morphology. Specifically, we contribute a verified combination of:
1. **Leakage Mitigation**: A 64-bit perceptual-hash (pHash) near-duplicate grouping protocol combined with a randomized-label negative control.
2. **Decoupled Multi-Task Formulation**: A hierarchical architecture that separates the ABO and Rh targets into independent classification heads optimized via a joint multi-task objective.
3. **Uncertainty Quantification**: The application of Temperature Scaling and Split Conformal Prediction to establish a functional abstention mechanism (`PREDICTION_WITHHELD`) for ambiguous impressions.
4. **Attribution Analysis**: The deployment of Grad-CAM++ integrated with an Orientation-Attention Alignment Score (OAAS) and Minutiae-based Counterfactual Attribution (MCA) to evaluate the spatial alignment of model reasoning against true ridge structures.

---

## 2. Related Work and Literature Gap Analysis

### 2.1 Dermatoglyphics and Hematological Associations
Classical studies have examined the statistical distribution of dermatoglyphic patterns across ABO blood phenotypes:
* **Susmiarsih et al. [1]** evaluated 302 subjects, finding loops most frequent in group B ($60.36\%$), whorls most frequent in group O ($40.45\%$), and arches most frequent in group AB ($5.12\%$). While statistically significant ($\chi^2, p < 0.05$), the effect sizes were small, concluding that patterns represent population skews rather than deterministic markers.
* **Bharadwaja et al. [2]** analyzed forensic ink prints across Indian cohorts, corroborating that while pattern frequencies differ between Rh-positive and Rh-negative individuals, inter-individual overlap prevents direct individual identification.

### 2.2 Deep Learning Approaches for Fingerprint Blood Grouping
* **Phadke et al. [9]** applied convolutional networks to Kaggle fingerprint datasets, reporting $\sim 88\%$ nominal accuracy. However, their published methodology describes data splitting at the image level without explicit mention of subject grouping or leakage audits.
* **Swathi et al. [11]** explored standard CNN backbones on similar datasets, reaching high test accuracies without reporting probability calibration or statistical significance tests against chance baselines.
* **Vineela et al. [12]** evaluated deep architectures (ResNet, VGG) for biometric healthcare, emphasizing feature representations but omitting conformal uncertainty guarantees.

### 2.3 Methodological Gaps
As summarized in Table 1, the reviewed studies generally lack subject-independent split audits, biological target decoupling, and formal uncertainty-gated abstention policies.

**Table 1: Comparative characteristics reported or identifiable in the reviewed studies**
| Citation | Backbone / Method | Dataset Split Protocol | Target Formulation | XAI Verification | Uncertainty / Rejection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Phadke et al. [9] | Custom CNN | Random Image Split | Flat 8-Class | Not reported | Not reported (Forced Guess) |
| Swathi et al. [11] | Standard CNN | Random Image Split | Flat 8-Class | Qualitative CAM | Not reported (Forced Guess) |
| Vineela et al. [12] | ResNet50 / VGG | Stratified Random | Flat 8-Class | Heatmap Visuals | Uncalibrated Softmax |
| **LeakSafe-CGN (Ours)** | **EfficientNet + CBAM** | **pHash-based pseudo-subject / near-duplicate grouping** | **Decoupled Multi-Task (ABO + Rh + Joint 8-Class)** | **Grad-CAM++ & Null OAAS** | **Split Conformal Sets** |

---

## 3. Materials and Methods

### 3.1 Dataset and Problem Formulation
Experiments were conducted on Dataset A, a corpus of 5,837 fingerprint images representing eight phenotype classes: $A^+$, $A^-$, $AB^+$, $AB^-$, $B^+$, $B^-$, $O^+$, and $O^-$. The problem is formulated in three distinct ways: a 4-class ABO formulation ($A$/$B$/$AB$/$O$), a binary Rh formulation (positive/negative), and a joint 8-class formulation predicting the exact blood group phenotype. Due to natural demographic variations, the dataset exhibits class imbalance, which we address through class weights during model training. We note specific dataset limitations: an audit confirms that the raw dataset does not provide individual donor IDs, age, sex, or finger indices, necessitating a robust mitigation strategy for identity leakage.

### 3.2 Data Preprocessing
The preprocessing pipeline is strictly defined. Images are converted to grayscale and resized to a standard $224 \times 224$ resolution. Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied with a clip limit of 2.6 to normalize ridge-valley contrast. To reduce high-frequency sensor noise, we apply $3 \times 3$ Gaussian filtering ($\sigma = 0.8$). Directional ridge information is enhanced using an eight-orientation Gabor filter bank, followed by maximum-response pooling across orientations. All pixel intensities are normalized to the $[0, 1]$ range. Data augmentation includes rotation ($\pm 10^\circ$), intensity scaling ($0.85$–$1.15$), and Gaussian sensor noise injection. Crucially, the preprocessing pipeline does **not** employ morphological foreground masking or singular-point alignment.

### 3.3 Handcrafted Texture Representation
To complement deep visual features, we extract a canonical 30-dimensional handcrafted descriptor focusing on biometric texture and ridge morphology. This descriptor consists of:
* **10 LBP features**: Computed via a uniform Local Binary Pattern histogram.
* **12 GLCM features**: Derived from the Gray-Level Co-occurrence Matrix computed at four orientations.
* **8 ridge morphology/spatial features**: Including intensity statistics, Shannon entropy, Canny edge density, Otsu threshold ratios, orientation coherence, and Laplacian variance.

### 3.4 LeakSafe-CGN Architecture
LeakSafe-CGN implements a biologically informed hierarchical architecture featuring two primary branches:

1. **Visual Branch**: An EfficientNetB0 [10] backbone extracts deep spatial features, passed through a trainable Convolutional Block Attention Module (CBAM) [5] to focus on task-relevant ridge features, followed by global average pooling to yield a 1,280-dimensional representation.
2. **Texture Branch**: The 30-dimensional handcrafted texture descriptor is passed through a Dense layer (64 units, ReLU activation) and normalized via LayerNorm.

The representations from both branches are fused into a $1,280 + 64 = 1,344$-dimensional vector using an adaptive learned gate. This fused representation is projected through a shared Dense latent layer (256 units, ReLU activation) and regularized with 35% Dropout (`Dropout(0.35)`). The network culminates in three independent task heads:
1. **ABO**: 4-class softmax output.
2. **Rh**: Binary sigmoid output.
3. **Joint Phenotype**: 8-class softmax output.

![Figure 1: Overview of the LeakSafe-CGN hierarchical architecture.](figures/fig1_architecture.png)
*Figure 1: Overview of the LeakSafe-CGN hierarchical architecture. The model integrates a deep visual branch (EfficientNetB0 with CBAM spatial-channel attention) and a handcrafted biometric texture branch. Both representations are fused via an adaptive gate and passed to three task-specific heads (ABO, Rh, and Joint Phenotype) optimized under a multi-task objective.*

### 3.5 Training Procedure
The network is optimized using the Adam optimizer ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$). We employ a learning rate schedule starting with a warm-up learning rate of $10^{-4}$, later reducing to a fine-tuning rate of $10^{-5}$ utilizing a `ReduceLROnPlateau` callback. Models are trained with a batch size of 32 for 40–50 epochs. `EarlyStopping` is configured to monitor validation loss, restoring the best weights upon convergence. The joint multi-task loss function is a weighted sum of the individual head losses: ABO Cross-Entropy ($0.50$), Rh Binary Cross-Entropy ($0.30$), and Joint 8-class Cross-Entropy ($0.20$).

### 3.6 Leakage-Mitigation Protocol
Given that the source dataset lacks verified donor identifiers, random image-level splitting poses a significant risk of identity leakage, where near-duplicate impressions from the same individual contaminate both training and testing partitions. To counteract this, **pHash-based pseudo-subject/near-duplicate grouping was used as a leakage-mitigation strategy.** Images with high visual similarity (64-bit perceptual hashing) were clustered into the same partition. We note that while this robustly separates visually analogous impressions, we do not claim it establishes true donor-level independence, because verified donor identities were not available.

### 3.7 Benchmarking Protocol
The full 5,837-image Dataset A was evaluated using an 80/20 grouped train/test split protocol (`GroupShuffleSplit`), employing the pHash near-duplicate separation to govern the splits (yielding approximately 4,669 training and 1,168 test images). Performance was evaluated against multiple baselines, including classical ML models and deep transfer-learning architectures. Primary evaluation metrics were Macro-F1 score and accuracy, alongside additional measurements for uncertainty calibration and computational efficiency where verified.

### 3.8 Ablation and Robustness Evaluation

**Ablation Study**
To quantify the individual contribution of each architectural component within LeakSafe-CGN, we performed a systematic ablation study. Modifications included removing the orientation field, omitting the CBAM spatial-channel attention gate, swapping adaptive gating for static average or concatenation fusion, employing single-branch (texture or ridge) networks, and removing backbone fine-tuning. We evaluated the test accuracy delta against the full model ($91.10\%$ accuracy) to evaluate each component's empirical contribution. For instance, the model without fine-tuning dropped to $86.64\%$, while the single-branch texture model yielded $79.79\%$.

**Robustness Evaluation**
We subjected the model to synthetic physical distortions simulating real-world sensor degradation at varying severities (0, 1, 2). The perturbations and resulting accuracies were:
* **Optical Blur**: Degraded from $89.84\%$ (Severity 0) down to $83.90\%$ (Severity 1) and $66.21\%$ (Severity 2).
* **Additive Noise**: Remained highly robust, achieving $91.55\%$, $90.75\%$, and $89.50\%$ across severities.
* **Occlusion/Crop**: Degraded from $87.44\%$ to $85.16\%$ and $75.34\%$.
* **Angular Rotation**: Degraded slightly from $91.10\%$ to $88.58\%$ and $82.53\%$.

### 3.9 Calibration and Uncertainty
To evaluate predictive reliability, we implemented rigorous uncertainty quantification. We applied temperature scaling [6] ($T = 1.365$) to the network logits, which notably reduced the Expected Calibration Error (ECE) before and after calibration. Furthermore, we utilized split conformal prediction [7], [8] to generate prediction sets with a target nominal marginal coverage of 90%. An abstention policy (`PREDICTION_WITHHELD`) was established to automatically withhold predictions when the uncertainty criterion exceeded empirical non-conformity thresholds.

### 3.10 Explainability and Attribution
Three distinct mechanisms were implemented to ensure the model's visual and spatial reasoning were interpretable and spatially aligned with relevant fingerprint structures:
* **Grad-CAM++ [3], [4]**: Utilized for deep visual attribution to generate high-resolution, class-discriminative saliency maps.
* **Orientation-Attention Alignment Score (OAAS)**: Computed a comparison between the deep attention gradients and underlying ridge-orientation information, evaluated against a permutation-based null evaluation to ensure statistical significance.
* **Minutiae-based Counterfactual Attribution (MCA)**: Conducted through Crossing Number skeletonization and targeted minutiae-region occlusion to measure the corresponding drop in prediction confidence.
These mechanisms serve strictly as interpretability/attribution analyses, not as proof that a particular fingerprint structure biologically determines blood group.

### 3.11 Statistical Analysis
Statistical analyses were applied to the experimental components for which inferential testing or effect-size estimation was appropriate, including ablation comparisons, permutation-based attribution testing, and handcrafted-feature analysis. Benchmark, calibration, and perturbation results are reported descriptively under the evaluated conditions.

### 3.12 Reproducibility
The framework was implemented using Python and the documented TensorFlow, OpenCV, scikit-image, and scikit-learn dependencies. Random seed 42 was configured for the reported stochastic operations. Reproduction depends on access to the corresponding source code, configuration, model weights, and dataset; the public dataset lacks verified donor identifiers. The repository also contains synthetic-image checks for the inference pipeline, but these should be interpreted as software tests rather than validation of scientific performance.

---

## 4. Results

### 4.1 Benchmark Performance
The primary evaluation of the proposed framework was conducted using an 80/20 grouped train/test split governed by the pHash near-duplicate grouping protocol. The overall classification performance demonstrated that LeakSafe-CGN achieved the highest measured accuracy across the evaluated model configurations. Specifically, LeakSafe-CGN achieved an accuracy of $91.10\% \pm 0.42\%$ with a Macro-F1 score of $0.909$. Among the evaluated transfer-learning backbones, MobileNetV2 achieved closely comparable performance at $91.07\% \pm 0.31\%$ (Macro-F1 $= 0.908$). While it achieved the highest point estimate among the evaluated models, the margin over MobileNetV2 was only $0.03$ percentage points, indicating parity in raw discriminative accuracy rather than a substantial predictive advantage.

Other competitive architectures included our preceding RidgeVisionNet baseline ($89.96\% \pm 0.64\%$, Macro-F1 $= 0.897$), ConvNeXt-Tiny ($89.91\% \pm 0.49\%$, Macro-F1 $= 0.895$), and EfficientNetB0 [10] ($89.82\% \pm 0.88\%$, Macro-F1 $= 0.894$). Classical models relying purely on extracted texture features performed substantially lower, consistent with a contribution from deep spatial feature extraction for this task. Table 2 summarizes the comparative performance.

**Table 2: Baseline Benchmark Comparison Under the 80/20 Grouped Evaluation Protocol**
| Model Architecture | Mean Test Accuracy | Standard Deviation ($\pm \sigma$) |
| :--- | :---: | :---: |
| **LeakSafe-CGN (Ours)** | **91.10%** | **$\pm 0.42$ percentage points** |
| MobileNetV2 | 91.07% | $\pm 0.31$ percentage points |
| RidgeVisionNet (Baseline) | 89.96% | $\pm 0.64$ percentage points |
| ConvNeXt-Tiny | 89.91% | $\pm 0.49$ percentage points |
| EfficientNetB0 (Plain) | 89.82% | $\pm 0.88$ percentage points |
| DenseNet121 | 88.64% | $\pm 0.26$ percentage points |
| InceptionV3 | 85.04% | $\pm 0.07$ percentage points |
| Plain CNN (Scratch) | 82.23% | $\pm 1.38$ percentage points |
| ResNet50 | 79.94% | $\pm 0.66$ percentage points |
| Linear SVM | 52.40% | $\pm 1.24$ percentage points |
| Random Forest | 38.63% | $\pm 0.31$ percentage points |
| SVM (RBF Kernel) | 24.65% | $\pm 1.24$ percentage points |

### 4.2 Architectural Ablation
A systematic ablation analysis isolated the contributions of individual network components to ascertain which mechanisms materially affected performance (Table 3). Removing the continuous orientation field resulted in a measurable accuracy drop ($-1.49\%$). Altering the fusion strategy from adaptive gating to direct concatenation ($-1.15\%$) or unweighted mean pooling ($-0.12\%$) similarly degraded accuracy. Interestingly, removal of the CBAM spatial-channel attention gate induced a slight accuracy increase ($+0.11\%$), yielding $91.21\%$. This demonstrates that the CBAM module is not strictly necessary for achieving peak discriminative accuracy in this architecture. Dissecting the dual-branch topology revealed that isolating the texture branch alone severely degraded accuracy ($-11.31\%$), whereas removing the texture branch in favor of single-branch appearance yielded a relatively minor decrease ($-0.92\%$). Furthermore, freezing the backbone weights instead of fine-tuning the top layers caused a $4.46\%$ decline. 

Finally, a randomized-label control was executed, yielding a catastrophic collapse to $12.61\%$ accuracy (approximating random chance for eight classes). This serves strictly as a negative-control experiment demonstrating that the network does not learn dataset-wide ordering artifacts; however, it does not constitute absolute proof that all forms of subtle identity leakage are impossible.

**Table 3: Systematic Architectural Ablation Results**
| Ablation Configuration | Test Accuracy | $\Delta$ vs Full Model |
| :--- | :---: | :---: |
| **Full LeakSafe-CGN** | **91.10%** | **-** |
| w/o CBAM Gate | 91.21% | +0.11% |
| Static Average Fusion | 90.98% | -0.12% |
| Single-Branch Appearance (Visual Only) | 90.18% | -0.92% |
| Direct Concat Fusion | 89.95% | -1.15% |
| w/o Orientation Field | 89.61% | -1.49% |
| No Fine-Tuning | 86.64% | -4.46% |
| Single-Branch Texture (Texture Only) | 79.79% | -11.31% |
| Randomized-Label Control | 12.61% | -78.49% |

### 4.3 Robustness Analysis
We subjected the models to synthetic physical image distortions to characterize their sensitivity profiles across five axes of acquisition perturbations: additive noise, rotation, optical blur, spatial downsampling, and peripheral occlusion (Table 4). The system exhibited different sensitivity profiles depending on the perturbation type. Performance was relatively preserved under zero-mean additive sensor noise, and angular rotation resulted in only gradual degradation, maintaining accuracies above $82\%$ even at moderate severities. In contrast, the system demonstrated pronounced failure modes when high-frequency spatial structures were obscured or removed. Both optical blur and peripheral occlusion caused substantial degradation, while spatial downsampling produced a severe performance decline, collapsing accuracy to $35.27\%$ at high severity. These degradation trajectories indicate a strong sensitivity to the loss of high-frequency spatial information, highlighting the specific vulnerability profile of the evaluated LeakSafe-CGN architecture rather than generalized robustness.

Severity 0 denotes the lowest perturbation setting used in the corresponding robustness test and should not necessarily be interpreted as an identical reproduction of the primary benchmark.

**Table 4: Physical Perturbation Robustness Trajectory**
| Perturbation Type | Severity 0 | Severity 1 | Severity 2 |
| :--- | :---: | :---: | :---: |
| Additive Noise | 91.55% | 90.75% | 89.50% |
| Angular Rotation | 91.10% | 88.58% | 82.53% |
| Optical Blur | 89.84% | 83.90% | 66.21% |
| Peripheral Occlusion | 87.44% | 85.16% | 75.34% |
| Spatial Downsample | 88.01% | 44.41% | 35.27% |

![Figure 2: Model robustness under synthetic physical perturbations.](figures/fig2_robustness.png)
*Figure 2: Model robustness under synthetic physical perturbations. The line plot illustrates the degradation trajectories of LeakSafe-CGN across varying severities of additive noise, angular rotation, optical blur, peripheral occlusion, and spatial downsampling. Note the severe performance collapse under high-frequency spatial downsampling compared to relative stability under additive noise.*

### 4.4 Calibration and Uncertainty
To evaluate the reliability of the model's confidence estimates, we applied Temperature Scaling on a held-out validation set, deriving a fitted temperature $T = 1.365$. This post-hoc calibration reduced the Expected Calibration Error (ECE) from a raw value of $0.0842$ to $0.0412$, representing a $51.1\%$ reduction. When extending these calibrated probabilities into Split Conformal Prediction sets, the system achieved the target empirical marginal coverage of $90.0\%$ at an empirical non-conformity threshold of $\hat{q} = 0.724$. Furthermore, the framework successfully rejected $100.0\%$ of samples within the evaluated synthetic out-of-distribution (OOD) test set by exceeding the conformal abstention threshold. It must be noted that these metrics strictly demonstrate the mathematical behavior of the system under the evaluated experimental conditions, and do not constitute clinical safety guarantees for real-world deployment. The quantitative calibration outcomes are detailed in Table 5.

**Table 5: Calibration and Uncertainty Metrics**
| Metric | Value |
| :--- | :---: |
| Temperature Scaling ($T$) | 1.365 |
| ECE (Uncalibrated $\to$ Calibrated) | 0.0842 $\to$ 0.0412 |
| Brier Score (Uncalibrated $\to$ Calibrated)| 0.141 $\to$ 0.119 |
| Target Nominal Coverage ($1 - \alpha$) | 90.0% |
| Empirical Coverage | 90.0% |
| Non-Conformity Threshold ($\hat{q}$) | 0.724 |
| Synthetic OOD Abstention Rate | 100.0% |

![Figure 3: Calibration and Conformal Prediction Sets.](figures/fig3_calibration.png)
*Figure 3: (Left) Reliability diagrams demonstrating Expected Calibration Error (ECE) before and after Temperature Scaling ($T=1.365$). (Right) Examples of Split Conformal Prediction sets outputting multiple classes or abstaining (`PREDICTION_WITHHELD`) when non-conformity thresholds are exceeded.*

### 4.5 Statistical Feature Analysis
Independent of the deep learning architecture, five handcrafted biometric texture features were analyzed for feature-vs-class statistical association with the 8-way phenotype class using ANOVA testing (Table 6). The features each yielded $p < 0.001$ significance levels across the flat 8-way target formulation. GLCM Energy exhibited the strongest association ($F = 117.07, \eta^2 = 0.4856$, Mutual Information $= 0.4756$ bits), followed closely by GLCM Homogeneity ($F = 100.06, \eta^2 = 0.4466$, Mutual Information $= 0.4661$ bits) and Intensity Entropy ($F = 96.01, \eta^2 = 0.4364$, Mutual Information $= 0.4578$ bits). LBP Uniformity ($F = 73.84, \eta^2 = 0.3732$, Mutual Information $= 0.3620$ bits) and Ridge Density ($F = 64.55, \eta^2 = 0.3423$, Mutual Information $= 0.3204$ bits) were also highly significant. Crucially, these results establish a statistical association with the class labels exclusively within the evaluated dataset; they do not suggest or prove that these micro-texture features biologically cause or determine blood-group differences.

**Table 6: Statistical Feature Validation (ANOVA $F$-Tests & Effect Sizes)**
| Feature Name | 8-Way $F$ | $p$-value | 8-Way $\eta^2$ | Mutual Info (bits) |
| :--- | :---: | :---: | :---: | :---: |
| GLCM Energy | 117.07 | $< 0.001$ | 0.4856 | 0.4756 |
| GLCM Homogeneity | 100.06 | $< 0.001$ | 0.4466 | 0.4661 |
| Intensity Entropy | 96.01 | $< 0.001$ | 0.4364 | 0.4578 |
| LBP Uniformity | 73.84 | $< 0.001$ | 0.3732 | 0.3620 |
| Ridge Density | 64.55 | $< 0.001$ | 0.3423 | 0.3204 |

### 4.6 Explainability Results
The interpretability of the network's internal representations was evaluated using three mechanisms: Grad-CAM++ for deep spatial attribution, the Orientation-Attention Alignment Score (OAAS) for structural correlation, and Minutiae-based Counterfactual Attribution (MCA) for counterfactual occlusion testing. The analyses evaluated whether model attribution and confidence changes were spatially aligned with fingerprint ridge structures and detected minutiae rather than predominantly occurring in background regions. Representative evaluation demonstrated a significant OAAS alignment correlation ($r = 0.428$, $p_{\text{null}} < 0.05$ under spatial permutation) and MCA showed that targeted occlusion of detected minutiae regions altered predictive confidence. However, this spatial alignment serves solely as an audit of the computational mechanism; it does not establish that these targeted ridge topologies biologically determine the ABO or Rh phenotype.

![Figure 4: Multi-modal attribution analysis.](figures/fig4_explainability.png)
*Figure 4: Multi-modal attribution analysis. (A) Original fingerprint impression. (B) Grad-CAM++ saliency map highlighting class-discriminative regions. (C) Orientation-Attention Alignment Score (OAAS) overlay mapping deep attention against classical ridge orientation fields. (D) Minutiae-based Counterfactual Attribution (MCA) showing prediction confidence decay upon targeted occlusion of detected minutiae regions.*

---

## 5. Discussion

### 5.1 Principal Findings
Our experiments demonstrate that the LeakSafe-CGN architecture achieved an overall accuracy of $91.10\% \pm 0.42\%$ (Macro-F1 $= 0.909$) in the evaluated benchmark. While it achieved the highest point estimate among the evaluated models, the margin over transfer-learning baselines was exceptionally narrow; for instance, MobileNetV2 achieved a closely comparable $91.07\% \pm 0.31\%$ accuracy. The verified ablation experiments indicate that several components of the multimodal architecture contributed differently to the observed performance. Post-hoc temperature scaling reduced the Expected Calibration Error (ECE) from $0.0842$ to $0.0412$, and the integration of Split Conformal Prediction successfully met the $90.0\%$ empirical marginal coverage target. Furthermore, robustness testing identified distinct degradation patterns under specific physical perturbations, and our explainability suite (Grad-CAM++, OAAS, MCA) evaluated the relationship between model attribution and underlying fingerprint structures. It is critical to distinguish these **observed computational results** from definitive clinical interpretation.

### 5.2 Comparison With Previous Studies
Rather than simply claiming an accuracy advantage over previous dermatoglyphic studies, the primary contribution of this work lies in its rigorous experimental methodology. Table 7 outlines the methodological differences of RidgeVision compared to standard paradigms found in recent literature.

**Table 7: Methodological Comparison with Prior Literature**
| Dimension | Reviewed Studies | LeakSafe-CGN (RidgeVision) |
| :--- | :--- | :--- |
| **Target Formulation** | Flat 8-class formulations reported in reviewed studies | Biologically informed: ABO + Rh + joint 8-class |
| **Data Split** | Image-level/random splitting reported in reviewed studies | 64-bit pHash-based near-duplicate grouping |
| **Calibration** | Not reported in reviewed studies | Temperature scaling ($T=1.365$) |
| **Uncertainty Rejection** | Not reported in reviewed studies | Conformal prediction / abstention withholding |
| **Explainability (XAI)** | Qualitative visualization reported in reviewed studies | Grad-CAM++ + OAAS + MCA (null-controlled) |
| **Robustness** | Not systematically evaluated in reviewed studies | Systematic physical perturbation testing |

### 5.3 Leakage Mitigation
Identity leakage is a known risk in the audited dataset, where highly correlated impressions from the same finger (or donor) may inadvertently span both training and test partitions. To address this, the 64-bit pHash protocol was introduced to cluster visually identical or near-duplicate impressions. However, we explicitly acknowledge a foundational limitation: because the public dataset does not provide verified donor identifiers, pHash grouping must be interpreted strictly as **near-duplicate leakage mitigation**, rather than mathematical proof of true, multi-session donor-level independence. 

### 5.4 Calibration and Abstention
Standard deep neural networks are frequently overconfident on ambiguous or out-of-distribution inputs. Our framework mitigates this by moving beyond forced point predictions. By applying temperature scaling ($T = 1.365$), the model's predictive probabilities became better calibrated under the evaluated calibration procedure, as evidenced by the ECE reduction from $0.0842$ to $0.0412$. When paired with Split Conformal Prediction, the system enforces prediction withholding under defined uncertainty criteria. The correct interpretation of these metrics is that the model's confidence estimates are mathematically calibrated to the validation distribution; this does **not** indicate that the model is "clinically safe" for deployment without human oversight.

### 5.5 Explainability and Biological Grounding
To interpret the network's spatial reasoning, three mechanisms were deployed:
* **Grad-CAM++**: Identified influential, class-discriminative image regions via higher-order gradients.
* **Orientation-Attention Alignment Score (OAAS)**: Tested the alignment between learned attribution and underlying ridge-orientation information using a permutation-based null comparison.
* **Minutiae-based Counterfactual Attribution (MCA)**: Measured prediction confidence changes after targeted occlusion of detected anatomical minutiae regions.

While these experiments provide evidence that model attribution overlaps with analyzed fingerprint structures, they do not establish that the model is free from sensor or acquisition confounding. Furthermore, it is imperative to enforce a strict scientific boundary: **attribution to fingerprint structures does not establish biological causation between those structures and the ABO/Rh phenotype.** As noted in our literature audit, there is only weak, mixed clinical evidence for deterministic fingerprint–blood-group relationships. 

### 5.6 Sensitivity Profiles and Failure Modes
Evaluation across physical perturbations identified different sensitivity profiles rather than uniform robustness (Table 4). While performance was relatively preserved under zero-mean additive noise, other perturbations caused distinct degradation trajectories: angular rotation led to gradual degradation, whereas optical blur and peripheral occlusion resulted in substantial degradation. Most critically, spatial downsampling induced severe performance decline, highlighting a pronounced failure mode when high-frequency spatial frequencies (micro-textures) are lost. These findings demonstrate the implications for acquisition quality: the model is sensitive to acquisition degradation, particularly loss of high-frequency spatial information.

### 5.7 Biological and Clinical Interpretation
The results of this study explicitly demonstrate **computational classification performance on the evaluated public dataset**, not clinical diagnostic validity. Our audit acknowledges the weak population-level clinical and dermatoglyphic evidence, the lack of clinical provenance, and the absence of multi-center/multi-scanner validation. Therefore, LeakSafe-CGN must be appropriately positioned as **an exploratory biometric/AI research framework rather than a replacement for serological blood-group testing.** 

### 5.8 Limitations
The integrity of this research is bounded by several specific limitations:
1. **Dataset Provenance**: The public Kaggle dataset lacks verified clinical or serological provenance.
2. **Clinical Validation**: There is an absence of clinical/serological validation information for the source images.
3. **Multi-Center Validation**: The lack of multi-center or multi-scanner validation means findings are tied to a single, uncharacterized acquisition source.
4. **Sensor Confounding**: Potential sensor or acquisition confounding cannot be ruled out as drivers of model performance.
5. **Pseudo-Subject Grouping**: Data splitting relied on pseudo-subject grouping using pHash rather than true donor IDs, limiting absolute guarantees against leakage.
6. **Grouped Evaluation**: The 80/20 grouped evaluation provides only a single view of performance, which may be optimistic compared to out-of-cohort generalization.
7. **Synthetic OOD**: The 100% OOD rejection rate is strictly bounded to the synthetic nature of the evaluated OOD benchmark, not a general claim of real-world OOD safety.
8. **Population Generalization**: There remains significant uncertainty about population-level generalization beyond the demographics implicitly captured in the dataset.
9. **Biological Associations**: There is only weak and mixed biological evidence in the medical literature for deterministic fingerprint–ABO/Rh associations.
10. **Diagnostic Validity**: The high measured accuracy on this dataset does **not** establish clinical diagnostic validity, nor does it justify replacing serological blood-group testing.
11. **Deployment Fallback**: The repository contains a deterministic hash-seeded `"research_mode"` fallback mechanism that activates when trained model weights fail; this is strictly a software fallback and not a scientific prediction mechanism.

### 5.9 Future Work
To transition this framework from computational exploration to scientific validation, future work must incorporate:
* Clinically verified ABO/Rh labels paired with cryptographically verified donor-level identifiers.
* External validation across a multi-center acquisition pipeline utilizing multiple fingerprint sensors (e.g., capacitive, optical, ultrasound).
* Broader demographic diversity and larger cohorts to validate performance on rare blood-group phenotypes.
* Prospective, real-world clinical evaluation.
* Complete removal and replacement of any non-model deployment fallbacks prior to any production testing.


## 6. Conclusion

In this paper, we presented **LeakSafe-CGN**, a rigorous framework for fingerprint-based phenotype prediction that addresses key methodological, architectural, and interpretability limitations of prior baselines. By integrating leakage-audited grouped splits, biologically informed multi-task heads, true Grad-CAM++ saliency, null-model alignment checks, and conformal abstention, LeakSafe-CGN seeks to better align deep learning capability with scientific defensibility. The framework was evaluated through systematic architectural ablations, physical perturbation benchmarks, calibration analysis, and attribution audits, providing a transparent basis for further dermatoglyphic biometric research.

---

## References

[1] T. P. Susmiarsih, S. Mustofa, and M. Mirfat, "A dermatoglyphic study: Association of fingerprint patterns among ABO blood groups," *Biosaintifika: J. Biol. & Biol. Educ.*, vol. 8, no. 3, pp. 294-300, 2016.
[2] A. Bharadwaja, P. K. Saraswat, S. K. Aggarwal, P. Banerji, and S. Bharadwaja, "Pattern of finger-prints in different ABO blood groups," *J. Indian Acad. Forensic Med.*, vol. 26, no. 1, pp. 6-9, 2004.
[3] R. R. Selvaraju, M. Cogswell, A. Das, A. Vedaldi, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618-626.
[4] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," in *Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)*, 2018, pp. 839-847.
[5] S. Woo, J. Park, J. Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 3-19.
[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321-1330.
[7] V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. New York, NY, USA: Springer Science & Business Media, 2005.
[8] A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," *arXiv preprint arXiv:2107.07511*, 2021.
[9] M. Phadke, A. Raut, C. Sawant, G. Ramekar, and S. Badhan, "Fingerprint based blood group detection using CNN," in *Lecture Notes in Electrical Engineering*, 2025.
[10] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105-6114.
[11] P. Swathi, K. Sushmita, and K. V. Horadi, "Efficient deep learning approach for blood group detection using fingerprint images," *Int. J. Adv. Res. Sci. Commun. Technol. (IJARSCT)*, 2024.
[12] H. S. Vineela, A. Kanki, B. Pranathi, and R. Neha, "Machine learning-based blood group detection: A review," *Int. Adv. Res. J. Sci. Eng. Technol. (IARJSET)*, vol. 12, no. 2, pp. 211-216, 2025, doi: 10.17148/IARJSET.2025.12226.
[13] G. Daniels, *Human Blood Groups*, 3rd ed. Oxford, U.K.: John Wiley & Sons, 2013.
[14] S. Kaufman, S. Rosset, C. Perlich, and O. Arik, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data (TKDD)*, vol. 6, no. 4, pp. 1-21, 2012.
[15] W. J. Babler, "Embryologic development of epidermal ridges and their configurations," *Birth Defects Orig. Artic. Ser.*, vol. 27, no. 2, pp. 95-112, 1991.
