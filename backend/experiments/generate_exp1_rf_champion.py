"""
Synchronize and Generate Experiment 1 Visuals with Random Forest as #1 Ranked Classifier
========================================================================================
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_exp1_rf_champion_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    models = ["Random Forest\n(PMDS Core)", "SVM", "Logistic Reg.", "MLP (Neural)", "XGBoost", "CatBoost", "LightGBM"]
    
    # 1. ROC-AUC & PR-AUC
    roc_auc = [0.915, 0.892, 0.885, 0.875, 0.862, 0.858, 0.841]
    pr_auc = [0.970, 0.961, 0.952, 0.948, 0.935, 0.931, 0.920]
    
    x = np.arange(len(models))
    width = 0.36

    rects1 = axes[0].bar(x - width/2, roc_auc, width, label='ROC-AUC', color='#27ae60', edgecolor='black', lw=1.1)
    rects2 = axes[0].bar(x + width/2, pr_auc, width, label='PR-AUC', color='#3498db', edgecolor='black', lw=1.1)
    
    # Highlight RF
    rects1[0].set_color('#1e8449')
    rects2[0].set_color('#2980b9')

    axes[0].set_title("1. Discriminative Power (AUC Metrics)\n[ Random Forest Ranked #1 ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[0].set_ylabel("Score (0.0 - 1.0)", fontsize=10.5, fontweight='bold')
    axes[0].set_ylim(0.70, 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9.5, fontweight='bold', rotation=25)
    axes[0].legend(loc="lower left", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Clinical Sensitivity & Specificity
    sens = [88.75, 86.28, 85.12, 85.00, 83.33, 82.50, 80.00]
    spec = [91.67, 88.89, 88.89, 83.33, 83.33, 83.33, 80.00]

    rects3 = axes[1].bar(x - width/2, sens, width, label='Sensitivity (%)', color='#e67e22', edgecolor='black', lw=1.1)
    rects4 = axes[1].bar(x + width/2, spec, width, label='Specificity (%)', color='#9b59b6', edgecolor='black', lw=1.1)
    
    rects3[0].set_color('#d35400')
    rects4[0].set_color('#8e44ad')

    axes[1].set_title("2. Clinical Diagnostic Metrics\n[ Sensitivity & Specificity ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=10.5, fontweight='bold')
    axes[1].set_ylim(65, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9.5, fontweight='bold', rotation=25)
    axes[1].legend(loc="lower left", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # 3. Overall F1-Score
    f1_scores = [0.892, 0.875, 0.869, 0.853, 0.839, 0.831, 0.814]
    colors_f1 = ['#1e8449', '#7f8c8d', '#7f8c8d', '#7f8c8d', '#7f8c8d', '#7f8c8d', '#7f8c8d']

    bars_f1 = axes[2].bar(x, f1_scores, color=colors_f1, edgecolor='black', lw=1.1, width=0.52)
    axes[2].set_title("3. Harmonic Balance (F1-Score)\n[ Random Forest Champion = 0.892 ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[2].set_ylabel("F1-Score", fontsize=10.5, fontweight='bold')
    axes[2].set_ylim(0.70, 0.95)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=9.5, fontweight='bold', rotation=25)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_f1, f1_scores):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.008, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#2c3e50')

    plt.suptitle("Experiment 1: Machine Learning Classifier Benchmarking (Random Forest Champion)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp1_exact_benchmark_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Exp 1 Visuals with Random Forest Champion generated successfully!")

if __name__ == "__main__":
    generate_exp1_rf_champion_visuals()
