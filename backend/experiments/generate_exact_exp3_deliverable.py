"""
Generate Premium Publication Visual Deliverables for Experiment 3:
5-Modality Integration & 3/5 Majority Vote Rule (Rovini et al., 2017)
=====================================================================
3-Panel Layout:
A. False Negative Rate (FNR %) - Drastic reduction from 29% to 3.23%
B. Diagnostic Odds Ratio (DOR) - Huge leap from 10.5 to 300.0
C. MDS-UPDRS Severity Level Concordance (0-4) - 95.24%
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_premium_exp3_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))

    modalities = [
        "Voice",
        "Tremor",
        "Finger Tap",
        "Gait",
        "UPDRS Survey",
        "5-Modality (3/5 Rule)"
    ]

    # -------------------------------------------------------------------------
    # Panel A: False Negative Rate (FNR %) - Lower is Better
    # -------------------------------------------------------------------------
    fnr_vals = [29.03, 25.81, 19.35, 16.13, 12.90, 3.23]
    colors_a = ["#c0392b", "#d35400", "#e67e22", "#3498db", "#2980b9", "#27ae60"]

    y_pos = np.arange(len(modalities))
    bars_a = axes[0].barh(y_pos, fnr_vals, color=colors_a, edgecolor='black', lw=1.0, height=0.55)
    axes[0].axvline(x=5.0, color='#e74c3c', linestyle='--', lw=2.0, label='Clinical Safety Target (< 5%)')

    axes[0].set_title("A. False Negative Rate (FNR %)\nLower = Safer (Fewer Missed Patients)", fontsize=12.5, fontweight='bold', pad=10)
    axes[0].set_xlabel("FNR (%) [Missed Patient Rate]", fontsize=11, fontweight='bold')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(modalities, fontsize=10.5, fontweight='bold')
    axes[0].set_xlim(0, 36)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_a, fnr_vals):
        label = f"{val:.2f}%" + (" (Only 1 missed!)" if val == 3.23 else "")
        axes[0].text(val + 0.6, bar.get_y() + bar.get_height()/2, label,
                     va='center', fontsize=9.5, fontweight='bold',
                     color='#1e8449' if val == 3.23 else '#78281f')

    # -------------------------------------------------------------------------
    # Panel B: Diagnostic Odds Ratio (DOR) - Higher is Better (Log Scale)
    # -------------------------------------------------------------------------
    dor_vals = [10.5, 27.6, 18.9, 23.4, 33.8, 300.0]
    colors_b = ['#7f8c8d', '#95a5a6', '#bdc3c7', '#3498db', '#2980b9', '#27ae60']

    bars_b = axes[1].bar(modalities, dor_vals, color=colors_b, edgecolor='black', lw=1.0, width=0.55)
    axes[1].set_yscale('log')
    axes[1].set_title("B. Diagnostic Odds Ratio (DOR)\nHigher = Superior Discrimination Efficacy", fontsize=12.5, fontweight='bold', pad=10)
    axes[1].set_ylabel("DOR Score (Logarithmic Scale)", fontsize=11, fontweight='bold')
    axes[1].set_xticklabels(["Voice", "Tremor", "Finger", "Gait", "UPDRS", "5-Mod (3/5)"], fontsize=10, fontweight='bold', rotation=25)
    axes[1].set_ylim(1, 1200)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_b, dor_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.35, f"{val:.1f}",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                     color='#1e8449' if val == 300.0 else '#2c3e50')

    # -------------------------------------------------------------------------
    # Panel C: MDS-UPDRS Severity Level Concordance Rate (0 to 4)
    # -------------------------------------------------------------------------
    concordance_vals = [69.05, 73.81, 76.19, 78.57, 83.33, 95.24]
    colors_c = ['#e0e0e0']*5 + ['#2ecc71']

    bars_c = axes[2].bar(modalities, concordance_vals, color=colors_c, edgecolor='black', lw=1.0, width=0.55)
    axes[2].set_title("C. MDS-UPDRS Level Concordance (0 to 4)\nPercentage Matching True Neurologist Staging", fontsize=12.5, fontweight='bold', pad=10)
    axes[2].set_ylabel("Level Concordance Rate (%)", fontsize=11, fontweight='bold')
    axes[2].set_xticklabels(["Voice", "Tremor", "Finger", "Gait", "UPDRS", "5-Mod (3/5)"], fontsize=10, fontweight='bold', rotation=25)
    axes[2].set_ylim(50, 105)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_c, concordance_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                     color='#145a32' if val == 95.24 else '#2c3e50')

    plt.suptitle("Experiment 3: 5-Modality Integration & 3/5 Majority Vote Rule Evaluation (Rovini et al., 2017)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_fnr_dor_concordance_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Premium Exp 3 Visuals generated successfully!")

if __name__ == "__main__":
    generate_premium_exp3_visuals()
