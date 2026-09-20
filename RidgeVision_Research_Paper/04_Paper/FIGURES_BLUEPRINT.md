# Manuscript Figures Blueprint

## Figure 1: Overall LeakSafe-CGN Framework Architecture

\\\mermaid
graph TD
    A[Fingerprint Image] --> B[Preprocessing]
    B --> B1(Grayscale)
    B --> B2(Resize)
    B --> B3(CLAHE)
    B --> B4(Gaussian Denoising)
    B --> B5(Gabor Filtering)
    B5 --> C[EfficientNetB0]
    B5 --> D[Texture Features]
    D --> D1(LBP)
    D --> D2(GLCM)
    D --> D3(Morphology)
    C --> E[CBAM]
    D3 --> F[Dense + LN]
    E --> G[Feature Concatenation <br> 1344-D]
    F --> G
    G --> H[Adaptive Gated Fusion]
    H --> I[Dense 256]
    I --> J[ABO 4-class]
    I --> K[Rh Binary]
    I --> L[Joint 8-class]
\\\

## Figure 2: Dataset Distribution
![Figure 2](figures/fig2_dataset.png)

## Figure 3: Comparative Performance of Benchmark Models
![Figure 3](figures/fig3_benchmark.png)

## Figure 4: Architectural Ablation Analysis of LeakSafe-CGN
![Figure 4](figures/fig4_ablation.png)

## Figure 5: Robustness Under Physical Image Perturbations
![Figure 5](figures/fig5_robustness.png)

## Figure 6: Calibration and Uncertainty Analysis
![Figure 6](figures/fig6_calibration.png)
