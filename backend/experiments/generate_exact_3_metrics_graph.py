"""
Generate Exact 3-Metric Publication Graphic for Experiment 3
============================================================
Exclusively contains the exact 3 metrics requested by the user:
1. อัตราหลุดตรวจ FNR (%) [ยิ่งต่ำยิ่งดี]
2. ดัชนีวินิจฉัย DOR [ยิ่งสูงยิ่งดี]
3. ความสอดคล้องระดับ 0-4 (Concordance %) [ยิ่งสูงยิ่งดี]
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

def generate_exact_3_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

    modalities = [
        "Voice",
        "Tremor",
        "Finger Tap",
        "Gait",
        "Questionnaire",
        "5-Modality (3/5)"
    ]

    # Invert so 5-Modality (3/5) is at the top
    modalities_rev = list(reversed(modalities))
    y_pos = np.arange(len(modalities_rev))

    # -------------------------------------------------------------------------
    # Panel 1: อัตราหลุดตรวจ FNR (%) [ยิ่งต่ำยิ่งดี]
    # -------------------------------------------------------------------------
    fnr_vals = [29.03, 25.81, 19.35, 16.13, 12.90, 3.23]
    fnr_rev = list(reversed(fnr_vals))
    colors_1 = ["#27ae60", "#2980b9", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

    bars1 = axes[0].barh(y_pos, fnr_rev, color=colors_1, edgecolor='black', lw=1.1, height=0.52)
    axes[0].axvline(x=5.0, color='#c0392b', linestyle='--', lw=2.0, label='Safety Limit (< 5%)')

    axes[0].set_title("1. False Negative Rate (FNR %)\n[ Lower is Safer / Target < 5% ]", fontsize=12.5, fontweight='bold', pad=10)
    axes[0].set_xlabel("False Negative Rate (%)", fontsize=10.5, fontweight='bold')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(modalities_rev, fontsize=10.5, fontweight='bold')
    axes[0].set_xlim(0, 36)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    axes[0].get_yticklabels()[0].set_color('#1e8449')

    for bar, val in zip(bars1, fnr_rev):
        text_label = f" {val:.1f}%" + (" (Passed!)" if val == 3.23 else "")
        axes[0].text(val + 0.6, bar.get_y() + bar.get_height()/2, text_label,
                     va='center', fontsize=9.8, fontweight='bold',
                     color='#1e8449' if val == 3.23 else '#2c3e50')

    # -------------------------------------------------------------------------
    # Panel 2: ดัชนีวินิจฉัย DOR [ยิ่งสูงยิ่งดี]
    # -------------------------------------------------------------------------
    dor_vals = [10.5, 27.6, 18.9, 23.4, 33.8, 300.0]
    dor_rev = list(reversed(dor_vals))
    colors_2 = ["#27ae60", "#2980b9", "#3498db", "#bdc3c7", "#bdc3c7", "#bdc3c7"]

    bars2 = axes[1].barh(y_pos, dor_rev, color=colors_2, edgecolor='black', lw=1.1, height=0.52)
    axes[1].set_xscale('log')
    axes[1].set_title("2. Diagnostic Odds Ratio (DOR)\n[ Higher is Better - Log Scale ]", fontsize=12.5, fontweight='bold', pad=10)
    axes[1].set_xlabel("Diagnostic Odds Ratio (DOR)", fontsize=10.5, fontweight='bold')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(modalities_rev, fontsize=10.5, fontweight='bold')
    axes[1].set_xlim(5, 700)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].get_yticklabels()[0].set_color('#1e8449')

    for bar, val in zip(bars2, dor_rev):
        text_label = f" {val:.1f}" + (" (Highest!)" if val == 300.0 else "")
        axes[1].text(val * 1.15, bar.get_y() + bar.get_height()/2, text_label,
                     va='center', fontsize=9.8, fontweight='bold',
                     color='#1e8449' if val == 300.0 else '#2c3e50')

    # -------------------------------------------------------------------------
    # Panel 3: ความสอดคล้องระดับ 0-4 Concordance (%) [ยิ่งสูงยิ่งดี]
    # -------------------------------------------------------------------------
    concordance_vals = [69.05, 73.81, 76.19, 78.57, 83.33, 95.24]
    concordance_rev = list(reversed(concordance_vals))
    colors_3 = ["#27ae60", "#64b5f6", "#90caf9", "#bdc3c7", "#bdc3c7", "#bdc3c7"]

    bars3 = axes[2].barh(y_pos, concordance_rev, color=colors_3, edgecolor='black', lw=1.1, height=0.52)
    axes[2].set_title("3. Level 0-4 Concordance Rate (%)\n[ Higher is Better / Matches Neurologist ]", fontsize=12.5, fontweight='bold', pad=10)
    axes[2].set_xlabel("MDS-UPDRS Level Concordance (%)", fontsize=10.5, fontweight='bold')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(modalities_rev, fontsize=10.5, fontweight='bold')
    axes[2].set_xlim(55, 108)
    axes[2].grid(axis='x', linestyle='--', alpha=0.5)
    axes[2].get_yticklabels()[0].set_color('#1e8449')

    for bar, val in zip(bars3, concordance_rev):
        matched = {69.05: 29, 73.81: 31, 76.19: 32, 78.57: 33, 83.33: 35, 95.24: 40}[val]
        text_label = f" {val:.1f}% ({matched}/42)"
        axes[2].text(val + 0.8, bar.get_y() + bar.get_height()/2, text_label,
                     va='center', fontsize=9.8, fontweight='bold',
                     color='#1e8449' if val == 95.24 else '#2c3e50')

    plt.suptitle("Experiment 3: 3 Core Clinical Metrics Evaluation (FNR, DOR, Level 0-4 Concordance)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_exact_3_metrics_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Exact 3-Metric Visuals generated successfully!")

if __name__ == "__main__":
    generate_exact_3_metrics()
