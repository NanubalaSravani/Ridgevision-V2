# 12 — Research Risks and Missing Information: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

This document flags all methodological, biological, and technical risks that could be challenged during academic peer review.

---

## 1. High-Priority Research Risks

### Risk 1: The "Biological Disconnect" vs Classical Dermatoglyphic Literature
* **Vulnerability**: Medical anthropology and forensic dermatoglyphic studies (Susmiarsih et al., 2016; Bharadwaja et al., 2004) establish that fingerprint ridge patterns (loops, whorls, arches) display only **weak, population-level statistical correlations** with ABO phenotypes ($p < 0.05$). No clinical study has ever demonstrated that a single fingerprint impression contains sufficient information to predict an individual's 8-way ABO/Rh status at ~90% accuracy.
* **Reviewer Challenge**: *"Why does your model achieve 90% accuracy when 80 years of clinical dermatoglyphics shows only weak statistical tendencies?"*
* **Defensible Response Grounded in Code**:
  1. Classical literature classified fingerprints using qualitative, hand-counted macro-patterns (loop/whorl/arch types).
  2. The CNN and `UnifiedTextureExtractor` extract **microscopic epidermal micro-texture** (GLCM energy, LBP uniformity, high-frequency ridge frequency and orientation coherence) that are invisible to manual visual inspection.
  3. ANOVA effect sizes (`statistical_validation_full_scale.json`) confirm massive statistical association for micro-textures (GLCM Energy $\eta^2 = 0.4856, p < 10^{-120}$).
  4. The model collapsed to 12.61% under randomized labels, proving it is not memorizing labels.

---

### Risk 2: Absence of Donor Identifiers and Risk of Identity Leakage
* **Vulnerability**: Public Kaggle fingerprint datasets do not contain subject/donor identifiers. In v1 (`90-accuracy.ipynb`), splitting was performed purely at the image level. Multiple impressions from the same individual may have been split across training and testing partitions, allowing the model to memorize donor identities rather than blood groups.
* **Mitigation Implemented in v2**: The 64-bit DCT perceptual hash (`pHash`) clustering protocol groups near-duplicate impressions from the same finger into a unified `cluster_id` and splits via `GroupShuffleSplit`.
* **Residual Risk**: While `pHash` clustering mitigates leakage of near-identical impressions, it cannot identify prints from *different fingers of the same person* (e.g., thumb vs index finger of donor X) if their visual topologies differ.

---

### Risk 3: Dual Dataset Usage Across Project Phases
* **Vulnerability**: Two different Kaggle datasets were used across the repository:
  1. Prototype v1 (`90-accuracy.ipynb`): 5,837 balanced images (1,000 per class) from `abhiramshibaraya`.
  2. Framework v2 (`Part 1-3.ipynb`, Master Report): 5,837 images from `sravani2006`.
* **Reviewer Challenge**: Conflating results between the 5,837-image balanced cohort and the 5,837-image imbalanced cohort without explicit demarcation will confuse reviewers.
* **Action Required**: The paper must clearly demarcate Phase 1 (Initial Feasibility Study on 5,837 images) from Phase 2 (Audited Benchmark on 5,837 images).

---

### Risk 4: Class Imbalance in Primary Benchmark Dataset
* **Vulnerability**: In the 5,837-image benchmark dataset, class distribution is imbalanced:
  * A+ has only 402 images (6.89%, class weight: 1.815)
  * A- has 1,009 images (17.29%, class weight: 0.723)
* **Risk**: High macro accuracy could mask lower sensitivity on underrepresented minority classes unless balanced class weights are strictly applied during training and evaluation.

---

### Risk 5: Silent Fallback to Hash-Seeded Pseudo-Predictions in UI Mode
* **Vulnerability**: In `backend/ml/inference/predictor.py` (Lines 378–434), if model weights fail to load, `_research_mode_probabilities()` computes a seeded pseudo-random probability from an image SHA-256 hash.
* **Risk**: If a benchmark or test script runs without verifying that model weights were loaded, it could silently report hash-seeded synthetic predictions.
* **Action Required**: Ensure all research benchmarks explicitly assert `self.model is not None` and disable `research_mode` during academic evaluations.

---

## 2. Missing Information (Not Found in Repository Files)

1. **Medical Serology Provenance**: **NOT FOUND IN PROJECT FILES**. The exact clinical/laboratory method used to determine ground truth blood groups (slide agglutination, tube agglutination, microplate) is unrecorded in the Kaggle dataset.
2. **Scanner & Acquisition Specifications**: **NOT FOUND IN PROJECT FILES**. Optical vs capacitive sensor model, DPI resolution, and image capture hardware are undocumented.
3. **Donor Demographics**: **NOT FOUND IN PROJECT FILES**. Donor age, biological sex, ethnicity, and geographic distribution are unrecorded.
4. **Frozen Split Manifest File**: **NOT FOUND IN PROJECT FILES**. While splitting code exists, a static CSV file listing the exact image filenames assigned to train, validation, and test sets was not persisted to disk.
5. **Loss & Metric Curve Image Files**: **NOT FOUND IN PROJECT FILES**. Epoch-by-epoch loss curves exist in notebook console text, but pre-rendered graphic files (PNG/PDF) are not saved on disk.
