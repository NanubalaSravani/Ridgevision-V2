import os

new_content = """## 4. Results

### 4.1 Benchmark Performance
The primary evaluation of the proposed framework was conducted using a 3-fold stratified cross-validation benchmark governed by the pHash grouping protocol. The overall classification performance demonstrated that LeakSafe-CGN achieved the highest measured accuracy across the evaluated model configurations. Specifically, LeakSafe-CGN achieved an accuracy of $91.10\% \pm 0.42\%$ with a Macro-F1 score of $0.909$. Among the evaluated transfer-learning backbones, MobileNetV2 achieved closely comparable performance at $91.07\% \pm 0.31\%$ (Macro-F1 $= 0.908$). Although LeakSafe-CGN numerically outperformed the standard architectures, the margin over MobileNetV2 was only $0.03$ percentage points, indicating parity in raw discriminative accuracy rather than a substantial predictive advantage.

Other competitive architectures included our preceding RidgeVisionNet baseline ($90.0\% \pm 0.64\%$, Macro-F1 $= 0.897$), ConvNeXt-Tiny ($89.9\% \pm 0.49\%$, Macro-F1 $= 0.895$), and EfficientNetB0 ($89.8\% \pm 0.88\%$, Macro-F1 $= 0.894$). Classical models relying purely on extracted texture features performed substantially lower, confirming the necessity of deep spatial feature extraction for this task.

### 4.2 Architectural Ablation
A systematic ablation analysis isolated the contributions of individual network components to ascertain which mechanisms materially affected performance (Table 4). Removing the continuous orientation field resulted in a measurable accuracy drop ($-1.49\%$). Altering the fusion strategy from adaptive gating to direct concatenation ($-1.15\%$) or unweighted mean pooling ($-0.12\%$) similarly degraded accuracy. Removal of the spatial-channel attention gate induced a negligible change ($+0.11\%$). Dissecting the dual-branch topology revealed that isolating the texture branch alone severely degraded accuracy ($-11.31\%$), whereas removing the texture branch in favor of single-branch appearance yielded a relatively minor decrease ($-0.92\%$). Furthermore, freezing the backbone weights instead of fine-tuning the top layers caused a $4.46\%$ decline. 

Finally, a randomized-label control was executed, yielding a catastrophic collapse to $12.61\%$ accuracy (approximating random chance for eight classes). This serves strictly as a negative-control experiment demonstrating that the network does not learn dataset-wide ordering artifacts; however, it does not constitute absolute proof that all forms of subtle identity leakage are impossible.

### 4.3 Robustness Analysis
We subjected the models to synthetic physical image distortions to identify performance stability and characterize bounds of failure across five axes: additive noise, rotation, optical blur, spatial downsampling, and peripheral occlusion (Table 5). The system exhibited relative robustness to zero-mean additive sensor noise and in-plane angular rotation, maintaining accuracies above $88\%$ even at moderate perturbation severities. In contrast, the system demonstrated pronounced failure modes when high-frequency spatial structures were destroyed. Optical blur caused rapid accuracy deterioration, while spatial downsampling produced the most severe performance decline, collapsing accuracy to $35.27\%$ at high severity. These degradation trajectories represent the observed sensitivities of the evaluated LeakSafe-CGN architecture, not generalized claims about all fingerprint analysis systems. 

### 4.4 Calibration and Uncertainty
To evaluate the reliability of the model's confidence estimates, we applied Temperature Scaling on a held-out validation set, deriving an optimal scalar $T = 1.365$. This post-hoc calibration reduced the Expected Calibration Error (ECE) from a raw value of $0.0842$ to $0.0412$, representing a $51.1\%$ reduction. When extending these calibrated probabilities into Split Conformal Prediction sets, the system achieved the target empirical marginal coverage of $90.0\%$. Furthermore, the framework successfully rejected $100.0\%$ of samples within the evaluated synthetic out-of-distribution (OOD) test set by exceeding the conformal abstention threshold. It must be noted that these metrics strictly demonstrate the mathematical behavior of the system under the evaluated experimental conditions, and do not constitute clinical safety guarantees for real-world deployment.

### 4.5 Statistical Feature Analysis
Independent of the deep learning architecture, five handcrafted biometric texture features were analyzed for locus-specific statistical association with the ABO and Rh targets using ANOVA testing (Table 7). The features each yielded $p < 0.001$ significance levels across the flat 8-way target formulation. GLCM Energy exhibited the strongest association ($F = 117.07, \eta^2 = 0.4856$, Mutual Information $= 0.4756$ bits), followed closely by GLCM Homogeneity ($F = 100.06, \eta^2 = 0.4466$, Mutual Information $= 0.4661$ bits) and Intensity Entropy ($F = 96.01, \eta^2 = 0.4364$, Mutual Information $= 0.4578$ bits). LBP Uniformity ($F = 73.84, \eta^2 = 0.3732$, Mutual Information $= 0.3620$ bits) and Ridge Density ($F = 64.55, \eta^2 = 0.3423$, Mutual Information $= 0.3204$ bits) were also highly significant. Crucially, these results establish a statistical association with the class labels exclusively within the evaluated dataset; they do not suggest or prove that these micro-texture features biologically cause or determine blood-group differences.

### 4.6 Explainability Results
The interpretability of the network's internal representations was evaluated using three mechanisms: Grad-CAM++ for deep spatial attribution, the Orientation-Attention Alignment Score (OAAS) for structural correlation, and Minutiae-Causal Attribution (MCA) for counterfactual occlusion testing. These analyses successfully demonstrated that the model's visual attention and predictive confidence are spatially associated with genuine fingerprint ridge structures and anatomical minutiae, rather than background artifacts. However, this spatial alignment serves solely as an audit of the computational mechanism; it does not establish that these targeted ridge topologies biologically determine the ABO or Rh phenotype.

---

## 5. Discussion

### 5.1 Principal Findings
The experiments demonstrated that deep convolutional networks, specifically the LeakSafe-CGN architecture, can achieve $>90\%$ classification accuracy on a large-scale fingerprint corpus while maintaining calibrated uncertainty bounds. The multi-task network effectively utilized high-frequency epidermal micro-texture rather than classical macroscopic loop and whorl counts. Furthermore, spatial attribution and statistical feature analyses confirmed that the network's discriminative capability was strongly associated with biologically valid ridge topologies.

### 5.2 Comparison with Previous Literature
While previous studies have reported comparable nominal accuracies ($85\% - 91\%$), the critical distinction of this work lies in its evaluation methodology. Prior research frequently relied on randomized image-level splitting protocols, uncalibrated flat 8-class softmax outputs, and heuristic spatial proxies. By contrast, LeakSafe-CGN introduces a mathematically constrained evaluation framework. The integration of pseudo-subject dataset auditing, decoupled genetic classification targets, conformal abstention policies, and null-controlled spatial attribution elevates the standard of evidence required to evaluate dermatoglyphic biometric systems. 

### 5.3 Leakage and Reproducibility
Identity leakage is the most pervasive flaw in single-source biometric datasets. If impressions from the same individual appear in both training and test partitions, the network will memorize identity artifacts rather than generalized phenotypic markers. Our implementation of the 64-bit pHash near-duplicate clustering protocol proactively mitigated this risk by enforcing strict fold separation. While this perceptual hashing is a necessary methodological safeguard, it remains an algorithmic approximation; it is not equivalent to, and cannot replace, verified donor-level identity separation.

### 5.4 Uncertainty and Abstention
Standard deep learning architectures often project high softmax confidences on degraded, ambiguous, or out-of-distribution inputs, rendering them dangerous in medical or forensic screening contexts. The incorporation of Temperature Scaling and Split Conformal Prediction establishes a vital safety perimeter. The system's ability to trigger a `PREDICTION_WITHHELD` abstention state provides a functional mechanism for managing epistemic uncertainty. However, the $90.0\%$ empirical coverage bound is derived mathematically from the validation cohort distribution; its reliability under substantial external domain shifts remains bounded by the assumptions of exchangeability. 

### 5.5 Explainability
The deployment of Grad-CAM++, OAAS, and MCA successfully shifted the interpretability paradigm from qualitative visualizations to quantitative, counterfactual attribution. These mechanisms verify that the network is looking at the ridges and minutiae rather than sensor noise. Importantly, these are strictly attribution analyses of the computational model. They elucidate how the network arrived at its prediction, but they do not prove that the highlighted fingerprint structures biologically encode the hematological phenotype.

### 5.6 Biological Interpretation
A central objective of this research was to reconcile the high accuracies reported by deep learning models with the weak population-level correlations documented in anthropological literature. Our statistical analysis highlights a strong computational association between localized micro-texture features (e.g., GLCM homogeneity) and blood group labels in the dataset. However, we explicitly distinguish this computational association from established biological causation. The current empirical evidence does not establish a deterministic biological mechanism linking embryonic volar pad development to the *ABO* and *RHD* genetic loci.

### 5.7 Limitations
This study contains three fundamental limitations that heavily restrict its real-world applicability:
1. **Lack of Clinical Provenance**: The public dataset lacks verified clinical serology records and granular demographic metadata (e.g., age, sex, ethnicity, finger indices). 
2. **Scanner and Acquisition Confounding**: The impressions were sourced from a unified repository. It is impossible to retroactively guarantee that the data does not contain hidden confounding artifacts related to the specific biometric scanner hardware or acquisition session. 
3. **Lack of Multi-Center Validation**: The system has not been externally validated on geographically isolated, multi-scanner cohorts, meaning its generalization to true real-world variance is entirely unknown.

Additionally, we disclose an important implementation issue identified during the code audit: the deployment architecture includes a deterministic SHA-256 hash-based `"research_mode"` fallback mechanism that activates when trained model weights fail to load. This engineering fallback artificially inflates prediction confidence to $99.9\%$ by mapping hashes directly to class labels. We emphasize that this artifact is strictly an engineering fallback for continuous integration testing; it must not be presented or utilized as a legitimate predictive mechanism in any clinical or scientific context.

### 5.8 Future Validation
The transition of dermatoglyphic phenotyping from an exploratory computational task to a scientifically verified biometric tool requires extensive future validation. The natural next experiments must incorporate clinically validated blood-group serology labels and cryptographically verified donor-level identity information. Prospective evaluations must span multiple optical and capacitive scanners, ensuring wide demographic diversity. Only through a large-scale, geographically distinct external validation cohort can the true clinical and forensic efficacy of dermatoglyphic phenotyping be definitively established."""

file_path = "RidgeVision_v2_Manuscript.md"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("## 4. Experimental Evaluation and Results"):
        start_idx = i
        break

for i, line in enumerate(lines[start_idx:], start=start_idx):
    if line.startswith("## 6. Conclusion"):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + [new_content + "\n\n"] + lines[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Successfully replaced content between line {start_idx} and {end_idx}.")
else:
    print("Failed to find start or end index.")
