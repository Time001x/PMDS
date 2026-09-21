"""
Generate Ultra-Clean, High-Legibility, Modern 2-Panel Visuals for Experiment 3
=============================================================================
Designed specifically for extreme clarity, readability, and immediate impact:
- Horizontal bars (no rotated text, instant reading)
- Clean typography and large readable numbers
- Panel 1 (Left): อัตราการหลุดตรวจ FNR % (แท่งสีแดงลดเหลือแท่งสีเขียว 3.2%)
- Panel 2 (Right): ความแม่นยำในการจัดระดับ 0-4 % (แท่งสีเขียวพุ่งแตะ 95.2%)
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_ultra_clean_visuals():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

    modalities = [
        "Voice",
        "Tremor",
        "Finger Tap",
        "Gait",
        "Questionnaire",
        "5-Modality (3/5 Rule)"
    ]

    # Invert so 5-Modality is at the top
    modalities_rev = list(reversed(modalities))
    y_pos = np.arange(len(modalities_rev))

    # -------------------------------------------------------------------------
    # Panel 1 (Left): FNR % (Lower is Better)
    # -------------------------------------------------------------------------
    fnr_vals = [29.03, 25.81, 19.35, 16.13, 12.90, 3.23]
    fnr_rev = list(reversed(fnr_vals))
    colors_fnr = ["#27ae60", "#2980b9", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

    bars1 = axes[0].barh(y_pos, fnr_rev, color=colors_fnr, edgecolor='black', lw=1.2, height=0.55)
    axes[0].axvline(x=5.0, color='#c0392b', linestyle='--', lw=2.0, label='Clinical Safety Limit (< 5%)')

    axes[0].set_title("1. False Negative Rate (FNR %)\n[ Shorter is Safer / Lowest Missed Cases ]", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("False Negative Rate (%)", fontsize=11, fontweight='bold')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(modalities_rev, fontsize=11, fontweight='bold')
    axes[0].set_xlim(0, 35)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    # Highlight 5-modality label
    axes[0].get_yticklabels()[0].set_color('#196f3d')

    for bar, val in zip(bars1, fnr_rev):
        miss = {29.03: 9, 25.81: 8, 19.35: 6, 16.13: 5, 12.90: 4, 3.23: 1}[val]
        text_label = f" {val:.1f}% (Miss {miss})" + (" [Best: Safety Passed!]" if val == 3.23 else "")
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height()/2, text_label,
                     va='center', fontsize=10, fontweight='bold',
                     color='#196f3d' if val == 3.23 else '#2c3e50')

    # -------------------------------------------------------------------------
    # Panel 2 (Right): Staging Accuracy & Concordance % (Higher is Better)
    # -------------------------------------------------------------------------
    acc_vals = [73.81, 78.57, 80.95, 83.33, 85.71, 95.24]
    acc_rev = list(reversed(acc_vals))
    colors_acc = ["#27ae60", "#64b5f6", "#90caf9", "#bdc3c7", "#bdc3c7", "#bdc3c7"]

    bars2 = axes[1].barh(y_pos, acc_rev, color=colors_acc, edgecolor='black', lw=1.2, height=0.55)

    axes[1].set_title("2. Staging Accuracy & Concordance (%)\n[ Longer is Better / Matches Neurologist ]", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Accuracy & Level Concordance Rate (%)", fontsize=11, fontweight='bold')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(modalities_rev, fontsize=11, fontweight='bold')
    axes[1].set_xlim(60, 108)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    # Highlight 5-modality label
    axes[1].get_yticklabels()[0].set_color('#196f3d')

    for bar, val in zip(bars2, acc_rev):
        matched = {73.81: 31, 78.57: 33, 80.95: 34, 83.33: 35, 85.71: 36, 95.24: 40}[val]
        text_label = f" {val:.1f}% ({matched}/42 Matched)" + (" [Highest Accuracy!]" if val == 95.24 else "")
        axes[1].text(val + 0.6, bar.get_y() + bar.get_height()/2, text_label,
                     va='center', fontsize=10, fontweight='bold',
                     color='#196f3d' if val == 95.24 else '#2c3e50')

    plt.suptitle("Experiment 3 Summary: Single Modalities vs 5-Modality Integration (3/5 Majority Rule)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_ultra_clean_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Ultra Clean Visuals generated successfully!")

if __name__ == "__main__":
    generate_ultra_clean_visuals()
