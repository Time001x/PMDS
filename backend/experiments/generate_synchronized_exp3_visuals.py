"""
Generate Perfectly Synchronized Publication Visuals for Experiment 3
====================================================================
Matches the clean table 100% (DOR removed, clear focus on FNR, Accuracy, and Staging Concordance).
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_synchronized_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    modalities = [
        "Voice",
        "Tremor",
        "Finger Tap",
        "Gait",
        "Questionnaire",
        "5-Modality (3/5)"
    ]

    # -------------------------------------------------------------------------
    # Chart 1: อัตราการหลุดตรวจ FNR (%) - ยิ่งต่ำยิ่งดี (สอดคล้องกับคอลัมน์ FNR ในตาราง)
    # -------------------------------------------------------------------------
    fnr_vals = [29.03, 25.81, 19.35, 16.13, 12.90, 3.23]
    colors_fnr = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2980b9", "#27ae60"]

    bars1 = axes[0].bar(modalities, fnr_vals, color=colors_fnr, edgecolor='black', lw=1.2, width=0.55)
    axes[0].axhline(y=5.0, color='#c0392b', linestyle='--', lw=2.0, label='Clinical Safety Limit (< 5%)')

    axes[0].set_title("1. False Negative Rate (FNR %)\n[ Lower is Safer / Target < 5% ]", fontsize=12.5, fontweight='bold', pad=10)
    axes[0].set_ylabel("False Negative Rate (%)", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 35)
    axes[0].set_xticklabels(modalities, fontsize=9.5, fontweight='bold', rotation=20)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, fnr_vals):
        miss_count = {29.03: 9, 25.81: 8, 19.35: 6, 16.13: 5, 12.90: 4, 3.23: 1}[val]
        text_label = f"{val:.1f}%\n(Miss {miss_count})"
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1.0, text_label,
                     ha='center', va='bottom', fontsize=9.2, fontweight='bold',
                     color='#196f3d' if val == 3.23 else '#78281f')

    # -------------------------------------------------------------------------
    # Chart 2: ความแม่นยำรวม Accuracy (%) - ยิ่งสูงยิ่งดี (สอดคล้องกับคอลัมน์ Accuracy ในตาราง)
    # -------------------------------------------------------------------------
    acc_vals = [73.81, 78.57, 80.95, 83.33, 85.71, 95.24]
    colors_acc = ["#bdc3c7", "#bdc3c7", "#bdc3c7", "#90caf9", "#64b5f6", "#27ae60"]

    bars2 = axes[1].bar(modalities, acc_vals, color=colors_acc, edgecolor='black', lw=1.2, width=0.55)
    axes[1].set_title("2. Staging Accuracy & Concordance (%)\n[ Higher is Better / Matches Neurologist ]", fontsize=12.5, fontweight='bold', pad=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    axes[1].set_ylim(60, 106)
    axes[1].set_xticklabels(modalities, fontsize=9.5, fontweight='bold', rotation=20)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, acc_vals):
        matched_count = {73.81: 31, 78.57: 33, 80.95: 34, 83.33: 35, 85.71: 36, 95.24: 40}[val]
        text_label = f"{val:.1f}%\n({matched_count}/42 Matched)"
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 1.0, text_label,
                     ha='center', va='bottom', fontsize=9.2, fontweight='bold',
                     color='#196f3d' if val == 95.24 else '#2c3e50')

    # -------------------------------------------------------------------------
    # Chart 3: สรุปผลคนไข้จริง 42 คน (เกณฑ์ 3/5) Donut Chart
    # -------------------------------------------------------------------------
    labels_pie = [
        "True Parkinson (TP)\n30 Patients (71.4%)",
        "True Healthy (TN)\n10 Patients (23.8%)",
        "False Alarm (FP)\n1 Patient (2.4%)",
        "Missed Case (FN)\n1 Patient (2.4%)"
    ]
    counts_pie = [30, 10, 1, 1]
    colors_pie = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    wedges, texts, autotexts = axes[2].pie(
        counts_pie, labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        textprops={'fontsize': 9.8, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.42, edgecolor='black', lw=1.2)
    )
    axes[2].set_title("3. Clinical Screening Breakdown (N=42)\n[ 40 Correct (95.2%) / 2 Errors (4.8%) ]", fontsize=12.5, fontweight='bold', pad=10)

    plt.suptitle("Experiment 3: Single Modalities vs 5-Modality Integration (3/5 Rule)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp3_synchronized_table_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Synchronized Visuals generated successfully!")

if __name__ == "__main__":
    generate_synchronized_visuals()
