"""
Generate Super Simple, Intuitive, and Crystal-Clear Visualizations
for PMDS Evaluation Results (Easy to Read at a Single Glance)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_easy_visual():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Chart 1: Giant Before vs After Comparison Bar Chart
    categories = ['Single Voice (Single Modality)\n[Acoustic Only]', 'PMDS Integrated (5 Modalities)\n[Voice + Tremor + Tap + Gait + Quest]']
    accuracy = [53.0, 97.6]
    miss_rate = [43.9, 0.0]

    x = np.arange(len(categories))
    width = 0.35

    bar1 = axes[0].bar(x - width/2, accuracy, width, label='Accuracy (Overall Correct %)', color=['#e67e22', '#27ae60'], edgecolor='black', lw=1.2)
    bar2 = axes[0].bar(x + width/2, miss_rate, width, label='Missed Patients % (False Negative)', color=['#c0392b', '#bdc3c7'], edgecolor='black', lw=1.2)

    axes[0].set_title("1. Single Voice vs PMDS 5-Modality Screening", fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 115)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11, fontweight='bold')
    axes[0].legend(loc="upper center", fontsize=10.5, frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Add big bold text labels on bars
    for bar, val in zip(bar1, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    for bar, val in zip(bar2, miss_rate):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=14, fontweight='bold', color='#922b21' if val > 0 else '#27ae60')

    # Chart 2: 5 Modalities Breakdown (Simple Horizontal Bars)
    modalities = ['1. Voice (Acoustic)', '2. Rest Tremor (Sensor)', '3. Finger Tapping (Screen)', '4. Gait & Balance (Sensor)', '5. Clinical Questionnaire', '>> COMBINED 5-MODALITY (3/5 Rule)']
    scores = [54.3, 75.0, 72.1, 76.5, 80.0, 97.6]
    colors = ['#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#34495e', '#2ecc71']

    y_pos = np.arange(len(modalities))
    bars = axes[1].barh(y_pos, scores, color=colors, height=0.55, edgecolor='black', lw=1.2)

    axes[1].set_title("2. Accuracy Across Individual Modalities vs 5-Modality", fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 115)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(modalities, fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    # Add text on horizontal bars
    for bar, score in zip(bars, scores):
        axes[1].text(score + 2, bar.get_y() + bar.get_height()/2, f"{score:.1f}%", va='center', fontsize=12, fontweight='bold')

    plt.suptitle("PMDS System Evaluation: Simple & Clear Executive Summary", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    local_path = os.path.join(FIGURES_DIR, "easy_executive_summary.png")
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, "easy_executive_summary.png")
    shutil.copyfile(local_path, artifact_path)
    print("Easy summary figure created successfully!")

if __name__ == "__main__":
    generate_easy_visual()
