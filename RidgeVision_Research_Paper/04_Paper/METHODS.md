# 3. Methods

## 3.1 Dataset and Phenotype Formulation
The study utilized a comprehensive benchmark corpus comprising 5,837 fingerprint impressions representing all eight ABO and Rh(D) blood group phenotypes. The class distribution reflects natural population asymmetries: A+ (402, 6.89%), A− (1,009, 17.29%), AB+ (708, 12.13%), AB− (761, 13.04%), B+ (652, 11.17%), B− (741, 12.69%), O+ (852, 14.60%), and O− (712, 12.20%). To mitigate class imbalance during training, sample weighting was applied, ranging from 0.723 for the majority class (A−) to 1.815 for the minority class (A+). Phenotype targets were inherently structured into the independent genetic loci of the ABO antigens (Chromosome 9, 4-way) and the Rh factor (Chromosome 1, binary).

## 3.2 Data Preprocessing
To standardize the structural quality of dermatoglyphic ridges across varying sensor conditions, all raw input images underwent a rigid preprocessing pipeline. The impressions were converted to grayscale, normalized, and resized to $224 \times 224$ pixels. Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to enhance ridge-valley local contrast. High-frequency sensor noise was mitigated via Gaussian denoising. Finally, multi-scale Gabor filtering was employed to emphasize oriented ridge frequencies and suppress cross-ridge artifacts prior to deep feature extraction.

## 3.3 Texture Feature Extraction
Alongside deep spatial features, a deterministic 30-dimensional biometric texture vector was extracted to capture classical dermatoglyphic morphology. This unified texture representation includes: 10-bin uniform Local Binary Patterns (LBP) to encode micro-textures, Gray-Level Co-occurrence Matrix (GLCM) statistics (energy, homogeneity, contrast, dissimilarity, correlation, and angular second moment) computed across four rotational angles, Canny edge densities, intensity moments, Shannon entropy, Otsu threshold ratios, continuous orientation field coherence, and Laplacian variance. This handcrafted branch explicitly provides the network with scale-invariant biometric priors.

## 3.4 LeakSafe-CGN Architecture
The LeakSafe-CGN framework utilizes a dual-branch hierarchical design. The visual appearance branch relies on an EfficientNetB0 backbone (fine-tuned) integrated with trainable Convolutional Block Attention Modules (CBAM) to refine spatial and channel-wise ridge features. Concurrently, the texture branch processes the 30-dimensional morphology vector through dense layers and Layer Normalization. These two representations are concatenated to form a 1,344-dimensional unified feature space. Rather than static pooling, an Adaptive Gated Fusion mechanism dynamically weights the deep appearance and deterministic texture branches, projecting the fused vector into a shared 256-dimensional latent space.

## 3.5 Multi-Task Prediction
To enforce biological coherence, the shared 256-dimensional representation is fed into three decoupled prediction heads: an ABO classification head (4-way softmax), an Rh factor head (binary sigmoid), and a joint phenotype head (flat 8-way softmax). This multi-task topology prevents the network from falsely equating independent genetic mismatches (e.g., penalizing an $A^+$ to $A^-$ confusion equivalently to an $A^+$ to $O^-$ confusion) and allows loci-specific gradients to backpropagate through the shared layers. The final loss is computed as a weighted sum of the categorical and binary cross-entropy losses from these three heads.

## 3.6 Training Procedure
Model weights were optimized using a two-stage training strategy to prevent catastrophic forgetting. In the first stage, the EfficientNetB0 backbone was frozen, and the dense fusion and classification heads were trained. In the second stage, the top 40 layers of the backbone were unfrozen for domain-specific fine-tuning. The network was trained using the Adam optimizer with early stopping and learning rate reduction on plateau. 

## 3.7 pHash-Based Leakage Audit
To definitively prevent identity leakage—a pervasive flaw in single-source biometric datasets—we implemented a strict leakage-audited split protocol. Perceptual hashing (pHash) was computed for all images, and impressions exhibiting a hash similarity $> 0.85$ (indicative of near-duplicates or varying pressure impressions from the same finger) were strictly grouped into pseudo-subject clusters. All cross-validation splits were performed at the cluster level via `GroupShuffleSplit`, guaranteeing that no pseudo-subject spanned both the training and testing partitions. 

## 3.8 Benchmark Protocol
The framework was evaluated using 3-Fold Stratified Group Cross-Validation. We benchmarked LeakSafe-CGN against a spectrum of architectural paradigms: classical models (Linear SVM, RBF SVM, Random Forest) utilizing the 30-dim texture vector, and deep convolutional networks (Plain CNN, ResNet50, InceptionV3, DenseNet121, EfficientNetB0, ConvNeXt-Tiny, MobileNetV2) utilizing the image matrix. Performance was strictly reported using mean test accuracy, standard deviation, macro F1 score, and latency.

## 3.9 Ablation Protocol
To validate architectural decisions, a systematic ablation analysis isolated the contribution of individual framework components. Tested configurations included: the removal of the continuous orientation field, the removal of the CBAM spatial-channel gate, the substitution of adaptive gating with static average or direct concatenation, the isolation of single-branch appearance and single-branch texture, and the removal of backbone fine-tuning. Crucially, a Randomized-Label Control—where target labels were randomly permuted—was executed as a negative control to ensure the model collapsed to expected chance accuracy (~12.5%).

## 3.10 Robustness Evaluation
The network's resilience to real-world sensor degradation was evaluated by applying five mathematical perturbations to the held-out test sets at three escalating severities (0, 1, 2). The perturbations consisted of: additive Gaussian noise, in-plane angular rotation, optical blur, spatial downsampling, and peripheral occlusion. Accuracy retention trajectories were recorded to map the failure bounds of the model under non-ideal biometric acquisition conditions.

## 3.11 Calibration and Conformal Prediction
To address deep neural network overconfidence, the raw logits were calibrated using Temperature Scaling, optimized via Negative Log-Likelihood on a held-out validation set ($T = 1.365$). Following calibration, Split Conformal Prediction was deployed to convert point estimates into prediction sets with guaranteed finite-sample marginal coverage ($1 - \alpha = 0.90$). An active abstention policy (`PREDICTION_WITHHELD`) was established to automatically reject predictions if the generated conformal set size exceeded two classes, safely managing ambiguous or anomalous prints.

## 3.12 Explainability: Grad-CAM++, OAAS, MCA
Interpretability was mandated via gradient-grounded causal attribution rather than heuristic edge proxies. Grad-CAM++ with higher-order backpropagation was utilized to generate class-discriminative saliency maps. To quantitatively verify that the network's spatial attention aligned with anatomical minutiae and ridge singularities, we computed the Orientation-Attention Alignment Score (OAAS). The OAAS was validated against an empirical null distribution generated via spatial tile permutations ($B = 100$) to rule out chance alignment. Additionally, Minutiae-Causal Attribution (MCA) confirmed the network's reliance on core and delta topologies.

## 3.13 Statistical Analysis
The statistical significance of performance gains was evaluated using the non-parametric Wilcoxon signed-rank test ($p < 0.05$ threshold) comparing LeakSafe-CGN's predictions against the benchmark models. Furthermore, to disaggregate the biological signal, ANOVA F-tests were performed on the 30-dimensional biometric features. The F-statistics, p-values, and effect sizes ($\eta^2$) were computed independently for the 8-way flat target, the 4-way ABO target, and the binary Rh target to empirically quantify locus-specific texture divergence.

## 3.14 Reproducibility
All computational experiments, including hyperparameter definitions, layer initialization seeds, and dataset split indices, were strictly deterministic. The Python codebase utilizes fixed random states for the data loader and network initializers. The full audit trail mapping raw output JSON results to the final manuscript tables ensures an unbroken evidence chain for peer review.
