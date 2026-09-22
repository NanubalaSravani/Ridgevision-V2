import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure directory exists
out_dir = r"c:\Users\srava\Downloads\Ridgevision-ai-main\Ridgevision-ai-main\RidgeVision_Research_Paper\04_Paper\figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 2: Dataset Distribution
# ---------------------------------------------------------
classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
counts = [402, 1009, 708, 761, 652, 741, 852, 712]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, counts, color=colors)
plt.title('Figure 2: Dataset Distribution Across Eight Blood-Group Phenotypes', fontsize=14)
plt.xlabel('Phenotype', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig2_dataset.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 3: Comparative Performance of Benchmark Models
# ---------------------------------------------------------
models = [
    'Linear SVM', 'SVM (RBF Kernel)', 'Random Forest', 'Plain CNN', 
    'ResNet50', 'InceptionV3', 'DenseNet121', 'EfficientNetB0', 
    'ConvNeXt-Tiny', 'RidgeVisionNet (Ours)', 'MobileNetV2', 'LeakSafe-CGN (Ours)'
]
acc = [52.4, 24.7, 38.6, 82.2, 79.9, 85.0, 88.6, 89.8, 89.9, 90.0, 91.1, 91.1]
std = [1.24, 1.24, 0.31, 1.38, 0.66, 0.07, 0.26, 0.88, 0.49, 0.64, 0.31, 0.42]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(models))
# Highlight ours
colors = ['#1f77b4' if 'Ours' not in m else '#d62728' for m in models]

ax.barh(y_pos, acc, xerr=std, align='center', color=colors, ecolor='black', capsize=5)
ax.set_yticks(y_pos, labels=models)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Mean Test Accuracy (%)')
ax.set_title('Figure 3: Comparative Performance of Benchmark Models')
# Add values
for i, v in enumerate(acc):
    ax.text(v + std[i] + 0.5, i, f'{v:.1f}%', va='center')
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig3_benchmark.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 4: Architectural Ablation Analysis of LeakSafe-CGN
# ---------------------------------------------------------
ablation_configs = [
    'Randomized-Label Control', 'No Fine-Tuning', 'Single-Branch Texture', 
    'Single-Branch Appearance', 'Direct Concat Fusion', 'w/o Orientation Field', 
    'Static Average Fusion', 'w/o ROAM Channel Gate', 'Full Model (LeakSafe-CGN)'
]
ablation_acc = [12.61, 86.64, 79.79, 90.18, 89.95, 89.61, 90.98, 91.21, 91.10]
delta = [-78.49, -4.46, -11.31, -0.92, -1.15, -1.49, -0.12, 0.11, 0.0]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(ablation_configs))
colors = ['#7f7f7f' if 'Control' in m else '#1f77b4' if 'Full' not in m else '#2ca02c' for m in ablation_configs]

bars = ax.barh(y_pos, ablation_acc, align='center', color=colors)
ax.set_yticks(y_pos, labels=ablation_configs)
ax.invert_yaxis()
ax.set_xlabel('Test Accuracy (%)')
ax.set_title('Figure 4: Architectural Ablation Analysis of LeakSafe-CGN')
for i, (v, d) in enumerate(zip(ablation_acc, delta)):
    if d == 0:
        label = f'{v:.2f}% (Ref)'
    elif d > 0:
        label = f'{v:.2f}% (+{d:.2f}%)'
    else:
        label = f'{v:.2f}% ({d:.2f}%)'
    ax.text(v + 1, i, label, va='center')
plt.xlim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig4_ablation.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 5: Robustness Under Physical Image Perturbations
# ---------------------------------------------------------
perts = ['Additive Noise', 'In-Plane Rotation', 'Optical Blur', 'Spatial Downsample', 'Peripheral Occlusion']
sev0 = [91.55, 91.10, 89.84, 88.01, 87.44]
sev1 = [90.75, 88.58, 83.90, 44.41, 85.16]
sev2 = [89.50, 82.53, 66.21, 35.27, 75.34]
x = [0, 1, 2]

plt.figure(figsize=(10, 6))
for i, p in enumerate(perts):
    plt.plot(x, [sev0[i], sev1[i], sev2[i]], marker='o', linewidth=2, label=p)
plt.xticks(x, ['Severity 0', 'Severity 1', 'Severity 2'])
plt.ylabel('Accuracy (%)')
plt.title('Figure 5: Robustness Under Physical Image Perturbations')
plt.ylim(20, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig5_robustness.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 6: Calibration and Uncertainty Analysis
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
labels = ['Raw (Uncalibrated)', 'Calibrated (T=1.365)']
ece = [0.0842, 0.0412]
ax.barh(labels, ece, color=['#d62728', '#2ca02c'], height=0.4)
ax.set_xlabel('Expected Calibration Error (ECE)')
ax.set_title('Figure 6: Calibration Improvement (ECE)')
for i, v in enumerate(ece):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')
plt.xlim(0, 0.1)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig6_calibration.png'), dpi=300)
plt.close()

print('Figures generated successfully.')
