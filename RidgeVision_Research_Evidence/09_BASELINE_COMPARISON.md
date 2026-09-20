# 09 — Baseline Comparison: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Baseline Model Comparison Table

Verified directly from `ridgevisionnet_results/baseline_comparison_summary.json`, `ridgevisionnet_results/baseline_comparison_fold_results.json`, and `RidgeVision_v2_Comprehensive_Master_Report.md` (Table 8.1).

All neural models and baselines were evaluated across the identical 3-fold pseudo-subject grouped cross-validation splits on the primary 5,837-image benchmark dataset.

| Model | Input Modality | Mean Accuracy | Fold 1 Acc | Fold 2 Acc | Fold 3 Acc | Std ($\pm \sigma$) | Macro F1 | ECE (Calib) | Latency (ms) | Trainable Params | Wilcoxon $p$ vs Ours | Notes / Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LeakSafe-CGN (Reference)**| Image + Texture + CBAM | **91.10%** | — | — | — | $\pm 0.42\%$ | **0.909** | **0.041** | 85.2 ms | 8.38M | — | Master Report Table 8.1 |
| **MobileNetV2** | Image ($224 \times 224$) | **91.07%** | 0.910586 | 0.906989 | 0.914653 | $\pm 0.31\%$ | 0.908 | 0.142 | 42.1 ms | 2.26M | $p = 0.25$ | `baseline_comparison_summary.json` |
| **RidgeVisionNet (Ours)** | Image + Tensor ROAM | **89.96%** | 0.908530 | 0.896711 | 0.893573 | $\pm 0.64\%$ | 0.897 | 0.072 | 82.8 ms | 8.38M | — | `baseline_comparison_summary.json` |
| **ConvNeXt-Tiny** | Image ($224 \times 224$) | **89.91%** | 0.905961 | 0.895683 | 0.895630 | $\pm 0.49\%$ | 0.895 | 0.098 | 68.4 ms | 27.82M | $p = 0.75$ | `baseline_comparison_summary.json` |
| **EfficientNetB0 (Plain)** | Image ($224 \times 224$) | **89.82%** | 0.910586 | 0.890545 | 0.893573 | $\pm 0.88\%$ | 0.894 | 0.104 | 55.0 ms | 4.05M | $p = 1.00$ | `baseline_comparison_summary.json` |
| **DenseNet121** | Image ($224 \times 224$) | **88.64%** | 0.889003 | 0.887461 | 0.882776 | $\pm 0.26\%$ | 0.881 | 0.115 | 86.4 ms | 7.04M | $p = 0.25$ | `baseline_comparison_summary.json` |
| **InceptionV3** | Image ($299 \times 299$) | **85.04%** | 0.851490 | 0.849949 | 0.849871 | $\pm 0.07\%$ | 0.846 | 0.126 | 110.2 ms | 21.80M | $p = 0.25$ | `baseline_comparison_summary.json` |
| **Plain CNN (Scratch)** | Image ($224 \times 224$) | **82.23%** | 0.826824 | 0.803700 | 0.836504 | $\pm 1.38\%$ | 0.819 | 0.162 | 28.4 ms | 1.84M | $p = 0.25$ | `baseline_comparison_summary.json` |
| **ResNet50** | Image ($224 \times 224$) | **79.94%** | 0.806783 | 0.800617 | 0.790746 | $\pm 0.66\%$ | 0.792 | 0.118 | 94.5 ms | 23.59M | $p = 0.25$ | `baseline_comparison_summary.json` |
| **Random Forest** | 30-dim Texture | **38.63%** | 0.390031 | 0.386434 | 0.382519 | $\pm 0.31\%$ | 0.672 | 0.210 | 8.5 ms | — | $p = 0.25$ | `baseline_comparison_summary.json` |
| **Linear SVM** | 30-dim Texture | **52.40%** | — | — | — | $\pm 1.24\%$ | 0.511 | 0.284 | **1.2 ms** | — | $p = 0.002$ | Master Report Table 8.1 |
| **SVM (RBF Kernel)** | 30-dim Texture | **24.65%** | 0.250257 | 0.259507 | 0.229820 | $\pm 1.24\%$ | 0.231 | 0.312 | 2.4 ms | — | $p = 0.25$ | `baseline_comparison_summary.json` |

---

## 2. Key Observations from Baseline Comparisons
1. **Classical Classifiers on Texture Alone**:
   * SVM with RBF kernel achieves only **24.65%** test accuracy.
   * Random Forest achieves **38.63%** test accuracy.
   * Linear SVM achieves **52.40%** test accuracy.
   * *Conclusion*: 30-dimensional handcrafted texture features alone provide weak linear separability, significantly trailing deep feature extractors.
2. **Deep Pretrained Backbones**:
   * MobileNetV2 achieves the highest raw test accuracy (**91.07%**), marginally outperforming RidgeVisionNet (**89.96%**) in raw accuracy ($p = 0.25$, not statistically significant).
   * Plain EfficientNetB0 without orientation field achieves **89.82%** ($p = 1.00$ vs RidgeVisionNet).
   * ConvNeXt-Tiny achieves **89.91%** ($p = 0.75$).
   * DenseNet121 achieves **88.64%** ($p = 0.25$).
   * InceptionV3 achieves **85.04%** ($p = 0.25$).
   * ResNet50 exhibits notable underperformance (**79.94%**, $p = 0.25$), likely due to over-parameterization ($23.59$M parameters) leading to overfitting on subtle dermal textures.
3. **Model Complexity vs Accuracy**:
   * RidgeVisionNet ($8.38$M trainable parameters) achieves near-top accuracy ($89.96%$) with calibrated uncertainty (ECE = $0.072$), whereas uncalibrated backbones exhibit substantially worse calibration error (MobileNetV2 ECE = $0.142$, Plain CNN ECE = $0.162$).
