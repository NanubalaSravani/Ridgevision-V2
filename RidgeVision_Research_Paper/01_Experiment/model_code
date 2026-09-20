# 05 — Model Architecture: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Implemented Architectures

The project defines two core architectures in `backend/ml/models/architecture.py`, alongside the legacy v1 models in `notebooks/90-accuracy.ipynb`:

1. **`LeakSafe-CGN` (`build_leaksafe_cgn_model`)**: The primary multi-task publication architecture with trainable CBAM attention, unified texture vector projection, learned adaptive gated fusion, and biologically decoupled heads.
2. **`RidgeVisionNet` (`build_ridgevision_net`)**: An end-to-end vision model featuring a deterministic on-device ridge orientation tensor field, ROAM attention, and Adaptive Gated Fusion.
3. **v1 Model 88 & Model 91 (`build_fusion_model`)**: EfficientNetB0 (224x224) and EfficientNetB3 (300x300) dual-branch networks combined via soft voting.

---

## 2. LeakSafe-CGN Layer-by-Layer Architectural Table

**Source Code**: `backend/ml/models/architecture.py`, Lines 182–257

| Layer Name | Layer Class / Type | Input Shape | Output Shape | Parameters / Operations |
| :--- | :--- | :---: | :---: | :--- |
| `image_input` | `InputLayer` | — | $(224, 224, 3)$ | Normalized RGB fingerprint image tensor |
| `texture_input` | `InputLayer` | — | $(30,)$ | Canonical 30-dim biometric texture vector |
| `efficientnetb0` | `EfficientNetB0` (Backbone) | $(224, 224, 3)$ | $(7, 7, 1280)$ | ImageNet pretrained, top 40 layers trainable |
| `cbam_block` | `CBAM` (Custom Layer) | $(7, 7, 1280)$ | $(7, 7, 1280)$ | Channel Attention ($r=8$) + Spatial Attention ($7 \times 7$ conv) |
| `cnn_gap` | `GlobalAveragePooling2D` | $(7, 7, 1280)$ | $(1280,)$ | Squeezes spatial dimensions across feature map |
| `texture_projection`| `Dense` | $(30,)$ | $(64,)$ | 64 units, ReLU activation |
| `texture_norm` | `LayerNormalization` | $(64,)$ | $(64,)$ | Normalizes projected texture activations |
| `multimodal_fusion`| `Concatenate` | $(1280,), (64,)$ | $(1344,)$ | Merges CNN features and texture projections |
| `fusion_gate` | `Dense` | $(1344,)$ | $(1344,)$ | 1344 units, Sigmoid activation (learned gating vector $\mathbf{g}$) |
| `gated_fusion` | `Multiply` | $(1344,), (1344,)$ | $(1344,)$ | Element-wise product: $\mathbf{x}_{\text{fused}} \odot \mathbf{g}$ |
| `shared_representation`| `Dense` | $(1344,)$ | $(256,)$ | 256 units, ReLU activation |
| `shared_dropout` | `Dropout` | $(256,)$ | $(256,)$ | Dropout rate $p = 0.35$ |
| `abo_group` | `Dense` (Softmax) | $(256,)$ | $(4,)$ | 4 units: classes $\{A, B, AB, O\}$ (Chromosome 9) |
| `rh_factor` | `Dense` (Sigmoid) | $(256,)$ | $(1,)$ | 1 unit: binary probability for Rh factor (Chromosome 1) |
| `blood_group` | `Dense` (Softmax) | $(256,)$ | $(8,)$ | 8 units: joint classes $\{A^+, A^-, AB^+, AB^-, B^+, B^-, O^+, O^-\}$ |

---

## 3. Mathematical Formulations of Key Custom Layers

### 3.1 Convolutional Block Attention Module (CBAM)
Implemented in `backend/ml/models/architecture.py`, Lines 96–179:
1. **Channel Attention**:
   $$\mathbf{M}_c(\mathbf{F}) = \sigma\left( \mathbf{W}_1 \left( \mathbf{W}_0 (\text{AvgPool}(\mathbf{F})) \right) + \mathbf{W}_1 \left( \mathbf{W}_0 (\text{MaxPool}(\mathbf{F})) \right) \right)$$
   Where $\mathbf{W}_0 \in \mathbb{R}^{\frac{C}{r} \times C}$, $\mathbf{W}_1 \in \mathbb{R}^{C \times \frac{C}{r}}$, reduction ratio $r=8$.
2. **Spatial Attention**:
   $$\mathbf{M}_s(\mathbf{F}') = \sigma\left( f^{7 \times 7}([\text{AvgPool}(\mathbf{F}'); \text{MaxPool}(\mathbf{F}')]) \right)$$
   Where inter-channel average and max slices are concatenated and convolved with a $7 \times 7$ kernel.

### 3.2 Ridge Orientation Field (Deterministic On-Device Layer in RidgeVisionNet)
Implemented in `backend/ml/models/architecture.py`, Lines 10–37:
* Non-trainable layer computing continuous directional fields directly on the GPU.
* Convolves single-channel grayscale print with fixed $3 \times 3$ Sobel kernels $S_x, S_y$.
* Calculates squared gradients $G_{xx} = G_x^2, G_{yy} = G_y^2, G_{xy} = G_x G_y$.
* Applies block-wise average pooling ($k=8$):
  $$\theta_2 = \text{atan2}(2 V_{xy}, V_{xx} - V_{yy})$$
  $$\text{Coherence} = \text{clip}\left(\frac{\sqrt{(2 V_{xy})^2 + (V_{xx} - V_{yy})^2}}{V_{xx} + V_{yy} + 10^{-6}}, 0.0, 1.0\right)$$
* Emits a 3-channel continuous orientation tensor: $[\cos(2\theta), \sin(2\theta), \text{Coherence}]$.

### 3.3 Ridge Orientation Attention Module (ROAM)
Implemented in `backend/ml/models/architecture.py`, Lines 40–75:
* Gates intermediate CNN feature maps from `block6a_expand_activation` using the continuous orientation tensor.
* Computes spatial projection via $3 \times 3$ Conv2D (ReLU) and $1 \times 1$ Conv2D (Sigmoid).
* Optionally excites channels via shared squeeze-and-excitation MLP.

### 3.4 Adaptive Gated Fusion
Implemented in `backend/ml/models/architecture.py`, Lines 78–93:
* Dynamically combines two feature vectors $\mathbf{a}, \mathbf{b}$ via learned sigmoid gate:
  $$\mathbf{g} = \sigma(\mathbf{W}_g [\text{proj}(\mathbf{a}); \text{proj}(\mathbf{b})])$$
  $$\mathbf{z}_{\text{fused}} = \mathbf{g} \odot \text{proj}(\mathbf{a}) + (1 - \mathbf{g}) \odot \text{proj}(\mathbf{b})$$

---

## 4. Parameter Count & Computational Complexity
Verified directly from `ridgevisionnet_results/compute_cost.json`:
* **Trainable Parameters**: **8,378,217**
* **Non-Trainable Parameters**: **2,012,071**
* **Total Parameters**: **10,390,288**
* **Single-Image Inference Latency**: **82.78 ms** (Batch size = 1, CPU/GPU mixed execution)
* **Model File Sizes on Disk**:
  * `models/ridgevision_model.keras`: **51,857,728 bytes** (~51.8 MB)
  * `ridgevisionnet_results/ridgevision_full_model.weights.h5`: **176,322,992 bytes** (~176.3 MB)

---

## 5. Multi-Task Loss Formulation
Implemented in `backend/ml/models/architecture.py`, Lines 278–289:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ABO}} \mathcal{L}_{\text{CE}}(y_{\text{ABO}}, \hat{y}_{\text{ABO}}) + \lambda_{\text{Rh}} \mathcal{L}_{\text{BCE}}(y_{\text{Rh}}, \hat{y}_{\text{Rh}}) + \lambda_{\text{flat}} \mathcal{L}_{\text{CE}}(y_{\text{flat}}, \hat{y}_{\text{flat}})$$
* **Loss Weights**:
  * $\lambda_{\text{ABO}} = 0.50$ (Categorical Cross-Entropy across 4 ABO classes)
  * $\lambda_{\text{Rh}} = 0.30$ (Binary Cross-Entropy for Rh factor)
  * $\lambda_{\text{flat}} = 0.20$ (Categorical Cross-Entropy across 8 flat classes)
