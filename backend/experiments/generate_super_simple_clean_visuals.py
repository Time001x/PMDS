"""
Generate Super Simple, Clean, and Easy-to-Read Visuals for Experiment 3
======================================================================
Focused on maximum simplicity and instant clarity:
- Clean standard layout (no confusing error bars, no log scale)
- Big bold numbers, friendly modern colors
- Clear, simple titles and labels that anyone can understand instantly
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

def generate_super_simple_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    modalities = ["Voice", "Tremor", "Finger Tap", "Gait", "Questionnaire"]

    # -------------------------------------------------------------------------
    # 1. FNR (%) - อัตราหลุดตรวจ (ยิ่งน้อยยิ่งดี)
    # -------------------------------------------------------------------------
    fnr = [29.0, 25.8, 19.4, 16.1, 12.9]
    colors_1 = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2980b9"]

    bars1 = axes[0].bar(modalities, fnr, color=colors_1, edgecolor='black', lw=1.1, width=0.55)
    axes[0].set_title("1. False Negative Rate (FNR %)\n[ Lower is Better ]", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("FNR (%)", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 36)
    axes[0].set_xticklabels(modalities, fontsize=10, fontweight='bold', rotation=15)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, fnr):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val:.1f}%",
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

    # -------------------------------------------------------------------------
    # 2. DOR - ค่าความแม่นวินิจฉัย (ยิ่งสูงยิ่งดี)
    # -------------------------------------------------------------------------
    dor = [10.5, 27.6, 18.9, 23.4, 33.8]
    colors_2 = ["#95a5a6", "#3498db", "#95a5a6", "#3498db", "#2ecc71"]

    bars2 = axes[1].bar(modalities, dor, color=colors_2, edgecolor='black', lw=1.1, width=0.55)
    axes[1].set_title("2. Diagnostic Odds Ratio (DOR)\n[ Higher is Better ]", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel("DOR Score", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 42)
    axes[1].set_xticklabels(modalities, fontsize=10, fontweight='bold', rotation=15)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, dor):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val:.1f}",
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

    # -------------------------------------------------------------------------
    # 3. Concordance - ตรงกับแพทย์ระดับ 0-4 (%) (ยิ่งสูงยิ่งดี)
    # -------------------------------------------------------------------------
    concordance = [69.1, 73.8, 76.2, 78.6, 83.3]
    colors_3 = ["#bdc3c7", "#bdc3c7", "#90caf9", "#64b5f6", "#27ae60"]

    bars3 = axes[2].bar(modalities, concordance, color=colors_3, edgecolor='black', lw=1.1, width=0.55)
    axes[2].set_title("3. Level 0-4 Concordance (%)\n[ Higher is Better ]", fontsize=13, fontweight='bold', pad=12)
    axes[2].set_ylabel("Concordance Rate (%)", fontsize=11, fontweight='bold')
    axes[2].set_ylim(50, 100)
    axes[2].set_xticklabels(modalities, fontsize=10, fontweight='bold', rotation=15)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars3, concordance):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.1f}%",
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

    plt.suptitle("Experiment 3: Simple & Clear Comparison of 5 Modalities", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_super_simple_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Super Simple Visuals generated successfully!")

if __name__ == "__main__":
    generate_super_simple_visuals()
