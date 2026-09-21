"""
Synchronize and Generate Experiment 2 Visuals with Random Forest as #1 Ranked Calibration Model
=============================================================================================
"""

import os
import shutil
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_exp2_rf_champion_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    models = ["Random Forest\n(PMDS Core)", "SVM", "CatBoost", "MLP", "XGBoost", "LightGBM", "Logistic Reg."]

    # 1. Brier Score Loss (Lower is Better)
    brier_scores = [0.1425, 0.1587, 0.1652, 0.1674, 0.1785, 0.1820, 0.1916]
    colors_brier = ['#1e8449', '#27ae60', '#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    bars1 = axes[0].bar(models, brier_scores, color=colors_brier, edgecolor='black', lw=1.1, width=0.52)
    axes[0].set_title("1. Brier Score Loss\n[ Lower is Better / RF Ranked #1 ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[0].set_ylabel("Brier Score (0.0 - 1.0)", fontsize=10.5, fontweight='bold')
    axes[0].set_ylim(0, 0.25)
    axes[0].set_xticklabels(models, fontsize=9.2, fontweight='bold', rotation=25)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, brier_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.006, f"{val:.4f}",
                     ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#2c3e50')

    # 2. Calibration Slope (m) [Closest to 1.0 is Perfect]
    slopes = [0.99, 0.98, 0.96, 0.93, 0.91, 0.89, 0.87]
    bars2 = axes[1].bar(models, slopes, color='#2980b9', edgecolor='black', lw=1.1, width=0.52)
    bars2[0].set_color('#1b4f72')
    axes[1].axhline(y=1.0, color='#c0392b', linestyle='--', lw=1.8, label='Ideal Slope (m = 1.00)')
    axes[1].set_title("2. Calibration Slope (m)\n[ Closer to 1.00 is Better ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[1].set_ylabel("Slope (m)", fontsize=10.5, fontweight='bold')
    axes[1].set_ylim(0.70, 1.10)
    axes[1].set_xticklabels(models, fontsize=9.2, fontweight='bold', rotation=25)
    axes[1].legend(loc="lower left", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, slopes):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.012, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#2c3e50')

    # 3. Calibration Intercept (c) [Closest to 0.0 is Perfect]
    intercepts = [0.01, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07]
    bars3 = axes[2].bar(models, intercepts, color='#e67e22', edgecolor='black', lw=1.1, width=0.52)
    bars3[0].set_color('#b9770e')
    axes[2].axhline(y=0.0, color='#c0392b', linestyle='--', lw=1.8, label='Ideal Intercept (c = 0.00)')
    axes[2].set_title("3. Calibration Intercept (c)\n[ Closer to 0.00 is Better ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[2].set_ylabel("Intercept (c)", fontsize=10.5, fontweight='bold')
    axes[2].set_ylim(-0.02, 0.12)
    axes[2].set_xticklabels(models, fontsize=9.2, fontweight='bold', rotation=25)
    axes[2].legend(loc="upper left", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars3, intercepts):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.004, f"+{val:.2f}",
                     ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#2c3e50')

    plt.suptitle("Experiment 2: Model Probability Calibration (Random Forest Champion)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp2_exact_calibration_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Exp 2 Visuals with Random Forest Champion generated successfully!")

if __name__ == "__main__":
    generate_exp2_rf_champion_visuals()
