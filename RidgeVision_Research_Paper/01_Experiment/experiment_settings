# 06 — Training Configuration: RidgeVision AI

**Date of Audit**: 2026-09-10  
**Audit Protocol**: Strict Code-Grounding & Non-Invention Policy  
**Status**: VERIFIED FROM PROJECT CODEBASE  

---

## 1. Verified Hyperparameters Across Training Regimes

### Regime A: v2 Journal Framework / Benchmark Training (`LeakSafe-CGN` & `RidgeVisionNet`)
Verified in `backend/ml/models/architecture.py`, `notebooks/Part 1.ipynb`, and `RidgeVision_v2_Comprehensive_Master_Report.md`:

| Hyperparameter | Stage 1: Head Warmup | Stage 2: Backbone Fine-Tuning | Notes & Source |
| :--- | :--- | :--- | :--- |
| **Trainable Layers** | Top Heads & Attention (Backbone Frozen) | Top 40 Backbone Layers + Heads | `architecture.py:311` (`trainable_backbone_layers=40`) |
| **Batch Normalization** | Frozen (`layer.trainable = False`) | Frozen (`layer.trainable = False`) | `architecture.py:312` |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$) | Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$) | Standard Adam implementation |
| **Base Learning Rate** | $\eta_1 = 1.0 \times 10^{-3}$ ($1\text{e-}3$) | $\eta_2 = 1.0 \times 10^{-5}$ ($1\text{e-}5$) | Master Report Table 5.1 |
| **Batch Size ($B$)** | 32 | 32 | Configured in training generator |
| **Maximum Epochs** | 8 epochs | 30 epochs (or up to 60 in v1) | Master Report Table 5.1 |
| **Early Stopping** | Monitored: `val_loss`, Patience: 7 epochs | Monitored: `val_loss`, Patience: 7 epochs | `restore_best_weights = True` |
| **Learning Rate Schedule** | `ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7)` | `ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-7)` | Monitored on `val_loss` |
| **Dropout Rates** | 0.35 on shared latent | 0.35 on shared latent | `architecture.py:231, 356` |
| **Mixed Precision** | `mixed_float16` enabled | `mixed_float16` enabled | Notebook log: `set_global_policy('mixed_float16')` |
| **XLA Compilation** | `jit_compile = True` | `jit_compile = True` | TensorFlow Accelerated Linear Algebra |

### Per-Baseline Learning Rate Overrides:
To prevent baseline collapse due to inappropriate learning rates, specialized overrides were logged in `Part 1.ipynb` and Master Report Section 5.3:
* `mobilenet_v2`: Warmup LR = $3 \times 10^{-4}$, Fine-tune LR = $3 \times 10^{-6}$
* `resnet50`: Warmup LR = $3 \times 10^{-5}$, Fine-tune LR = $3 \times 10^{-7}$
* All other baselines: Default $1 \times 10^{-3} / 1 \times 10^{-5}$ schedule

---

### Regime B: v1 Prototype Ensemble Training (`notebooks/90-accuracy.ipynb`)
Verified directly from cell execution logs in `notebooks/90-accuracy.ipynb`:
* **Dataset**: 8,000 images (`abhiramshibaraya`), 1,000 per class, 5600 train / 1200 val / 1200 test.
* **Model 88 (`EfficientNetB0`)**:
  * Input: $224 \times 224 \times 3$, Batch size = 16.
  * Stage 1 (Warmup): 5 epochs, Adam $\text{LR} = 1 \times 10^{-3}$.
  * Stage 2 (Fine-tuning): 15 epochs, top 40 layers trainable, Adam $\text{LR} = 1 \times 10^{-5}$.
  * Best validation accuracy achieved: **88.75%** at epoch 12. Test accuracy: **86.00%**.
* **Model 91 (`EfficientNetB3`)**:
  * Input: $300 \times 300 \times 3$, Batch size = 16.
  * Stage 1 (Warmup): 15 epochs, Adam $\text{LR} = 1 \times 10^{-3}$.
  * Stage 2 (Fine-tuning): 60 epochs, top 80 layers trainable, Adam $\text{LR} = 1 \times 10^{-5}$, EarlyStopping patience = 20.
  * Best validation accuracy achieved: **89.25%** at epoch 47 (LR reduced to $8.999 \times 10^{-7}$). Test accuracy: **89.42%**.
* **Soft-Voting Ensemble**:
  * Grid search found optimal weights: $w_{88} = 0.15, w_{91} = 0.85$.
  * Best ensemble validation accuracy: **89.42%**. Test accuracy: **89.50%** (~90%).

---

## 2. Hardware Environment
Verified from cell output logs (`notebooks/90-accuracy.ipynb` Cell 2):
* **Compute Platform**: Kaggle Cloud GPU Environment
* **GPUs Detected**: **2x NVIDIA Tesla T4** (`/physical_device:GPU:0`, `/physical_device:GPU:1`)
* **VRAM**: 14,757 MB (16GB nominal) per GPU
* **Driver / Compute Capability**: CUDA compute capability 7.5
* **System RAM**: **NOT FOUND IN LOGS** (standard Kaggle GPU instance provides 30GB host RAM)

---

## 3. Software Environment & Dependencies
Verified from `requirements.txt`, `backend/requirements.txt`, `.python-version`, and runtime logs:
* **Python**: **3.11.9** (`.python-version`, `runtime.txt`)
* **TensorFlow**: **2.19.0** (verified in notebook execution log; requirement specified as `>=2.16,<3.0`)
* **OpenCV**: `opencv-python-headless >= 4.10, < 5.0`
* **scikit-learn**: `>= 1.4, < 2.0`
* **scikit-image**: `>= 0.25, < 1.0`
* **SciPy**: `>= 1.11, < 2.0`
* **FastAPI**: `>= 0.115, < 1.0`
* **Uvicorn**: `>= 0.34, < 1.0`
* **Random Seed**: `SEED = 42` pinned across all notebooks and split functions.
