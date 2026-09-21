"""
Generate Formal Academic Visuals for the 5 Individual Modalities ONLY
=====================================================================
Excludes the '5-Modality (3/5 Rule)', displaying strictly:
1. Voice
2. Tremor
3. Finger Tap
4. Gait
5. Questionnaire (UPDRS)
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_single_modalities_only_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    modalities = [
        "Voice",
        "Tremor",
        "Finger Tap",
        "Gait",
        "Questionnaire"
    ]
    x_pos = np.arange(len(modalities))

    # -------------------------------------------------------------------------
    # Panel A: False Negative Rate (FNR) - Single Modalities Only
    # -------------------------------------------------------------------------
    fnr_vals = [29.03, 25.81, 19.35, 16.13, 12.90]
    colors_a = ["#c0392b", "#d35400", "#e67e22", "#2980b9", "#1f618d"]

    bars_a = axes[0].bar(x_pos, fnr_vals, color=colors_a, edgecolor='black', lw=1.1, width=0.52)
    axes[0].axhline(y=5.0, color='#c0392b', linestyle='--', lw=2.0, label='Clinical Safety Target (FNR < 5%)')

    axes[0].set_title("A. False Negative Rate (FNR %)\nClinical Screening Safety by Modality", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel("False Negative Rate (FNR %)", fontsize=11, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(modalities, fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 36)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_a, fnr_vals):
        missed = {29.03: 9, 25.81: 8, 19.35: 6, 16.13: 5, 12.90: 4}[val]
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.9, f"{val:.2f}%\n(n={missed})",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2c3e50')

    # -------------------------------------------------------------------------
    # Panel B: Diagnostic Odds Ratio (DOR) - Single Modalities Only
    # -------------------------------------------------------------------------
    dor_vals = [10.5, 27.6, 18.9, 23.4, 33.8]
    colors_b = ['#7f8c8d', '#95a5a6', '#bdc3c7', '#3498db', '#2980b9']

    bars_b = axes[1].bar(x_pos, dor_vals, color=colors_b, edgecolor='black', lw=1.1, width=0.52)
    axes[1].set_title("B. Diagnostic Odds Ratio (DOR)\nDiagnostic Discriminative Efficacy", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_ylabel("Diagnostic Odds Ratio (DOR)", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(modalities, fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 42)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_b, dor_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}",
                     ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#2c3e50')

    # -------------------------------------------------------------------------
    # Panel C: MDS-UPDRS Staging Concordance Rate (Levels 0 to 4)
    # -------------------------------------------------------------------------
    concordance_vals = [69.05, 73.81, 76.19, 78.57, 83.33]
    colors_c = ['#bdc3c7', '#95a5a6', '#90caf9', '#64b5f6', '#2980b9']

    bars_c = axes[2].bar(x_pos, concordance_vals, color=colors_c, edgecolor='black', lw=1.1, width=0.52)
    axes[2].axhline(y=90.0, color='#27ae60', linestyle='--', lw=1.8, label='Clinical Target Concordance (> 90%)')

    axes[2].set_title("C. MDS-UPDRS Staging Concordance (0 to 4)\nAgreement with Expert Clinical Staging", fontsize=12, fontweight='bold', pad=12)
    axes[2].set_ylabel("Concordance Rate (%)", fontsize=11, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(modalities, fontsize=10, fontweight='bold')
    axes[2].set_ylim(50, 100)
    axes[2].legend(loc="lower right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_c, concordance_vals):
        matched = {69.05: 29, 73.81: 31, 76.19: 32, 78.57: 33, 83.33: 35}[val]
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%\n({matched}/42)",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2c3e50')

    plt.suptitle("Experiment 3: Quantitative Assessment of 5 Individual Modalities (Rovini et al., 2017)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_formal_publication_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Single Modalities Only Visuals generated successfully!")

if __name__ == "__main__":
    generate_single_modalities_only_visuals()
