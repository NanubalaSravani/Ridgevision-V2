import os

new_content = """## 5. Discussion

### 5.1 Principal Findings
Our experiments demonstrate that the LeakSafe-CGN architecture achieved an overall accuracy of $91.10\% \pm 0.42\%$ (Macro-F1 $= 0.909$) in the evaluated benchmark. While it numerically outperformed transfer-learning baselines, the margin was exceptionally narrow; for instance, MobileNetV2 achieved a closely comparable $91.07\% \pm 0.31\%$ accuracy. The verified ablation experiments indicate that several components of the multimodal architecture contributed differently to the observed performance. Post-hoc temperature scaling reduced the Expected Calibration Error (ECE) from $0.0842$ to $0.0412$, and the integration of Split Conformal Prediction successfully met the $90.0\%$ empirical marginal coverage target. Furthermore, robustness testing identified distinct degradation patterns under specific physical perturbations, and our explainability suite (Grad-CAM++, OAAS, MCA) quantified the relationship between model attribution and underlying fingerprint structures. It is critical to distinguish these **observed computational results** from definitive clinical interpretation.

### 5.2 Comparison With Previous Studies
Rather than simply claiming an accuracy advantage over previous dermatoglyphic studies, the primary contribution of this work lies in its rigorous experimental methodology. Table 8 outlines the methodological differences of RidgeVision compared to standard paradigms found in recent literature.

**Table 8: Methodological Comparison with Prior Literature**
| Dimension | Earlier Studies | LeakSafe-CGN (RidgeVision) |
| :--- | :--- | :--- |
| **Target Formulation** | Often flat 8-class softmax | Biologically decoupled: ABO + Rh + joint 8-class |
| **Data Split** | Often image-level random split | 64-bit pHash-based near-duplicate grouping |
| **Calibration** | Often absent | Temperature scaling ($T=1.365$) |
| **Uncertainty Rejection** | Usually forced prediction | Conformal prediction / abstention withholding |
| **Explainability (XAI)** | Often qualitative edge-proxy heatmaps | Grad-CAM++ + OAAS + MCA (null-controlled) |
| **Robustness** | Limited to pristine images | Systematic physical perturbation testing |

### 5.3 Leakage Mitigation
A prevalent flaw in single-source public biometric datasets is identity leakage, where highly correlated impressions from the same finger (or donor) inadvertently span both training and test partitions. To address this, the 64-bit pHash protocol was introduced to cluster visually identical or near-duplicate impressions. However, we explicitly acknowledge a foundational limitation: because the public dataset does not provide verified donor identifiers, pHash grouping must be interpreted strictly as **near-duplicate leakage mitigation**, rather than mathematical proof of true, multi-session donor-level independence. 

### 5.4 Calibration and Abstention
Standard deep neural networks are frequently overconfident on ambiguous or out-of-distribution inputs. Our framework mitigates this by moving beyond forced point predictions. By applying temperature scaling ($T = 1.365$), the model's predictive probabilities became significantly better calibrated under the evaluated calibration procedure, as evidenced by the ECE reduction from $0.0842$ to $0.0412$. When paired with Split Conformal Prediction, the system enforces prediction withholding under defined uncertainty criteria. The correct interpretation of these metrics is that the model's confidence estimates are mathematically calibrated to the validation distribution; this does **not** indicate that the model is "clinically safe" for deployment without human oversight.

### 5.5 Explainability and Biological Grounding
To interpret the network's spatial reasoning, three mechanisms were deployed:
* **Grad-CAM++**: Identified influential, class-discriminative image regions via higher-order gradients.
* **Orientation-Attention Alignment Score (OAAS)**: Tested the alignment between learned attribution and underlying ridge-orientation information using a permutation-based null comparison.
* **Minutiae-Causal Attribution (MCA)**: Measured prediction confidence changes after targeted occlusion of detected anatomical minutiae regions.

While these experiments verify that the model attends to valid fingerprint structures rather than sensor artifacts, it is imperative to enforce a strict scientific boundary: **attribution to fingerprint structures does not establish biological causation between those structures and the ABO/Rh phenotype.** As noted in our literature audit, there is only weak, mixed clinical evidence for deterministic fingerprint–blood-group relationships. 

### 5.6 Robustness and Failure Modes
Robustness evaluation identified specific degradation bounds for the architecture (Table 5). The system remained relatively stable under zero-mean additive noise and in-plane angular rotation. Conversely, it exhibited substantial degradation and pronounced failure modes when subjected to optical blur and spatial downsampling, indicating a strong reliance on high-frequency spatial frequencies (micro-textures). These findings demonstrate the implications for acquisition quality: the model requires pristine, high-resolution impressions to function correctly, and sensor degradation will rapidly compromise its predictive integrity.

### 5.7 Biological and Clinical Interpretation
The results of this study explicitly demonstrate **computational classification performance on the evaluated public dataset**, not clinical diagnostic validity. Our audit acknowledges the weak population-level clinical and dermatoglyphic evidence, the lack of clinical provenance, and the absence of multi-center/multi-scanner validation. Therefore, LeakSafe-CGN must be appropriately positioned as **an exploratory biometric/AI research framework rather than a replacement for serological blood-group testing.** 

### 5.8 Limitations
The integrity of this research is bounded by several specific limitations:
1. **Dataset Provenance**: The public Kaggle dataset lacks verified clinical serology provenance.
2. **Donor Identity**: The absence of verified donor IDs means absolute donor-level separation cannot be guaranteed.
3. **Sensor Confounding**: Without multi-scanner or multi-center validation, hidden acquisition artifacts cannot be ruled out.
4. **Biological Validity**: High computational performance does not establish a biological or developmental mechanism linking ridges to blood types.
5. **External Validation**: The results are confined to this specific cohort and are not yet validated on an independent clinical cohort.
6. **Rare/Imbalanced Phenotypes**: The dataset contains severe imbalances; for example, the A+ phenotype is represented by only 402 images, limiting robust inference for minority classes.
7. **Deployment Fallback Artifact**: The project repository contains a deterministic hash-seeded `"research_mode"` fallback mechanism that activates when trained model weights fail. This must be treated strictly as a software development fallback and **not as a scientific prediction mechanism**.

### 5.9 Future Work
To transition this framework from computational exploration to scientific validation, future work must incorporate:
* Clinically verified ABO/Rh labels paired with cryptographically verified donor-level identifiers.
* External validation across a multi-center acquisition pipeline utilizing multiple fingerprint sensors (e.g., capacitive, optical, ultrasound).
* Broader demographic diversity and larger cohorts to validate performance on rare blood-group phenotypes.
* Prospective, real-world clinical evaluation.
* Complete removal and replacement of any non-model deployment fallbacks prior to any production testing.
"""

file_path = "RidgeVision_v2_Manuscript.md"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("## 5. Discussion"):
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
    print(f"Successfully replaced Section 5 between line {start_idx} and {end_idx}.")
else:
    print("Failed to find start or end index.")
