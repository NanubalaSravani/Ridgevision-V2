# 03 — Preprocessing Audit: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. End-to-End Image Preprocessing Pipeline

The actual preprocessing pipeline implemented in `backend/ml/preprocessing/fingerprint.py` and `backend/ml/feature_engineering/texture.py` consists of the following sequential operations:

```
[Raw Fingerprint Image Stream (bytes)]
                 ↓
Operation 1: decode_image()
                 ↓
Operation 2: Color Space Conversion (BGR to Grayscale)
                 ↓
Operation 3: Spatial Resampling (INTER_AREA to 224x224)
                 ↓
Operation 4: Contrast-Limited Adaptive Histogram Equalization (CLAHE)
                 ↓
Operation 5: Gaussian Denoising Smoothing
                 ↓
Operation 6: Multi-Orientation Gabor Filter Bank Convolutions (8 Orientations)
                 ↓
Operation 7: Max-Response Orientation Pooling
                 ↓
Operation 8: Min-Max Normalization (uint8 [0, 255] and float32 [0.0, 1.0])
                 ↓
        ┌────────┴──────────────────────────────────────────┐
        ▼                                                   ▼
Spatial Tensor Normalization                    Unified Biometric Texture Extraction
(224×224×3 float32)                             (30-dim Canonical Feature Vector)
```

---

## 2. Operation-by-Operation Audit

### Operation 1: Image Decoding
* **Function**: `decode_image(image_bytes: bytes) -> np.ndarray`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Lines 7–12
* **Operation**: Converts binary buffer into a NumPy uint8 array using `cv2.imdecode(buffer, cv2.IMREAD_COLOR)`.
* **Parameters**: `flags = cv2.IMREAD_COLOR` (loads as 3-channel BGR).
* **Validation**: Raises `ValueError("Could not decode image. Use a valid PNG, JPG, or WEBP file.")` if decoding returns `None`.

### Operation 2: Grayscale Conversion
* **Function**: `enhance_fingerprint(image: np.ndarray) -> dict[str, np.ndarray]`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Line 16
* **Operation**: Converts 3-channel BGR to single-channel 8-bit luminance using standard Rec.601 coefficients:
  $$I_{\text{gray}} = 0.299 R + 0.587 G + 0.114 B$$
* **Method**: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.

### Operation 3: Spatial Resizing
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Line 17
* **Operation**: Resamples image to target dimensions (224 x 224 pixels) specified in `settings.image_size`.
* **Interpolation**: `cv2.INTER_AREA` (pixel area relation interpolation, optimal for avoiding moire artifacts when downsampling high-frequency epidermal ridges).

### Operation 4: Contrast Enhancement (CLAHE)
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Lines 19–20
* **Operation**: Applies Contrast-Limited Adaptive Histogram Equalization on local contextual tiles:
* **Parameters**:
  * `clipLimit = 2.6`: Limits maximum histogram bin height to prevent over-amplifying sensor background noise.
  * `tileGridSize = (8, 8)`: Partitions image into an 8 x 8 grid of local contextual regions (28 x 28 pixels per tile).
* **Method**: `clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)); contrast = clahe.apply(resized)`.

### Operation 5: Gaussian Denoising
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Line 21
* **Operation**: Smooths sensor salt-and-pepper noise while preserving continuous epidermal ridge edges.
* **Parameters**:
  * Kernel size: `(3, 3)`
  * Standard deviation: `sigmaX = 0` (computed from kernel size)
* **Method**: `denoised = cv2.GaussianBlur(contrast, (3, 3), 0)`.

### Operation 6: Multi-Scale Gabor Filter Bank
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Lines 23–27
* **Operation**: Convolves denoised image with a bank of 8 Gabor kernels across canonical orientation angles.
* **Parameters**:
  * Kernel size: `ksize = (17, 17)`
  * Spatial envelope: `sigma = 4.0`
  * Orientation angles: 8 orientations generated via `np.linspace(0, np.pi, 8, endpoint=False)`: 0, pi/8, 2pi/8, 3pi/8, 4pi/8, 5pi/8, 6pi/8, 7pi/8
  * Wavelength: `lambd = 10.0` pixels (tuned to average human inter-ridge distance at 224x224)
  * Spatial aspect ratio (ellipticity): `gamma = 0.55`
  * Phase offset: `psi = 0`
  * Output data type: `ktype = cv2.CV_32F`
* **Method**: `cv2.getGaborKernel((17, 17), 4.0, theta, 10.0, 0.55, 0, ktype=cv2.CV_32F)` followed by `cv2.filter2D(denoised, cv2.CV_32F, kernel)`.

### Operation 7: Max-Response Orientation Pooling
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Line 29
* **Operation**: Takes element-wise maximum across all 8 orientation filter responses:
  $$I_{\text{enhanced}}(x, y) = \max_{k \in \{0,\dots,7\}} \left( I_{\text{denoise}} * G_{\theta_k} \right)(x, y)$$
* **Method**: `enhanced = np.max(np.stack(gabor_responses, axis=0), axis=0)`.

### Operation 8: Dynamic Range Normalization
* **Function**: `enhance_fingerprint(image: np.ndarray)`
* **Source Location**: `backend/ml/preprocessing/fingerprint.py`, Lines 30–31
* **Operation**:
  * Maps response to uint8: `cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)`
  * Maps to float32 range [0.0, 1.0]: `normalized = enhanced.astype(np.float32) / 255.0`

---

## 3. Canonical 30-Dimensional Feature Extraction Pipeline
* **Class**: `UnifiedTextureExtractor`
* **Source Location**: `backend/ml/feature_engineering/texture.py`, Lines 7–109
* **Components**:
  1. **LBP Uniform Histogram (10 dimensions)**:
     * `local_binary_pattern(enhanced_224, P=8, R=1, method="uniform")`
     * 10 bins: bins 0–8 for uniform patterns, bin 9 for non-uniform patterns. Normalized via `density=True`.
  2. **GLCM Descriptors (12 dimensions)**:
     * `graycomatrix(enhanced_224, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)`
     * Computes mean and standard deviation across 4 angles for: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM (6 x 2 = 12 features).
  3. **Biometric Ridge & Quality Metrics (8 dimensions)**:
     * Canny ridge density: `np.mean(cv2.Canny(enhanced_224, 50, 150) > 0)`
     * Mean Sobel gradient magnitude: `np.mean(np.sqrt(gx**2 + gy**2)) / 255.0`
     * Mean intensity: `np.mean(enhanced_224) / 255.0`
     * Intensity standard deviation: `np.std(enhanced_224) / 255.0`
     * Shannon entropy: `shannon_entropy(enhanced_224) / 8.0`
     * Otsu foreground binarization ratio: `np.mean(cv2.threshold(..., cv2.THRESH_OTSU)[1] > 0)`
     * Orientation field coherence: `np.abs(np.mean(np.exp(1j * np.arctan2(gy, gx))))`
     * Laplacian focus variance: `cv2.Laplacian(enhanced_224, cv2.CV_64F).var() / 10000.0`

---

## 4. Data Augmentation Pipeline (Applied During Training)
Documented in `notebooks/90-accuracy.ipynb` (Cell 6) and `RidgeVision_v2_Comprehensive_Master_Report.md` (Section 1.4.4):
* **In-plane rotation**: +/- 10 deg with border reflection (`cv2.BORDER_REFLECT`).
* **Photometric intensity scaling**: I' = alpha * I + beta, where alpha in [0.85, 1.15] and beta in [-0.05, 0.05] (or [-15, 15] in v1).
* **Additive sensor noise**: Gaussian noise N(0, 0.02^2) applied with probability p=0.30.
* **Spatial translation**: dx, dy in [-10, +10] pixels.
* **Affine zoom**: 0.85x to 1.15x.

---

## 5. Operations NOT Implemented
* **Fingerprint Segmentation**: **NOT IMPLEMENTED** (The pipeline does not compute convex hull or morphological mask segmentation to crop out empty scanner background; raw rectangular borders are retained).
* **Core/Delta Alignment**: **NOT IMPLEMENTED** (Images are not rotated or aligned to canonical core/delta anchor points prior to feeding into CNN).
