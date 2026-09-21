"""
Generate Publication Visuals for Experiment 5:
API Response Latency & AES-256 Security Overhead Benchmark
==========================================================
Metrics:
1. AES-256 Encryption/Decryption Overhead (< 3.5 ms)
2. Database Save Test Result Latency (< 12.0 ms)
3. Predict API End-to-End Latency (< 350 ms)
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

def generate_exp5_visuals():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. AES-256 Security Overhead
    categories_1 = ["Encryption", "Decryption", "Combined"]
    vals_1 = [0.023, 0.009, 0.032]
    colors_1 = ["#3498db", "#2980b9", "#27ae60"]

    bars1 = axes[0].bar(categories_1, vals_1, color=colors_1, edgecolor='black', lw=1.1, width=0.48)
    axes[0].axhline(y=3.5, color='#c0392b', linestyle='--', lw=1.8, label='Target Limit (< 3.5 ms)')
    axes[0].set_title("1. AES-256 Security Overhead\n[ Target < 3.5 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[0].set_ylabel("Latency (ms) [Lower is Faster]", fontsize=10.5, fontweight='bold')
    axes[0].set_ylim(0, 4.0)
    axes[0].set_xticks(range(len(categories_1)))
    axes[0].set_xticklabels(categories_1, fontsize=10.5, fontweight='bold')
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, vals_1):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.15, f"{val:.3f} ms\n(Passed!)",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e8449')

    # 2. Database Save Test Result Latency
    categories_2 = ["Mean", "Median", "P95"]
    vals_2 = [2.793, 2.652, 3.241]
    colors_2 = ["#e67e22", "#f39c12", "#d35400"]

    bars2 = axes[1].bar(categories_2, vals_2, color=colors_2, edgecolor='black', lw=1.1, width=0.48)
    axes[1].axhline(y=12.0, color='#c0392b', linestyle='--', lw=1.8, label='Target Limit (< 12.0 ms)')
    axes[1].set_title("2. SQLite DB Save Latency\n(pmds_secure.db) [ Target < 12 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[1].set_ylabel("Write Latency (ms)", fontsize=10.5, fontweight='bold')
    axes[1].set_ylim(0, 14.0)
    axes[1].set_xticks(range(len(categories_2)))
    axes[1].set_xticklabels(categories_2, fontsize=10.5, fontweight='bold')
    axes[1].legend(loc="upper right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, vals_2):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.4, f"{val:.2f} ms\n(Passed!)",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e8449')

    # 3. Predict API End-to-End Latency
    categories_3 = ["Mean", "Median", "P95"]
    vals_3 = [9.35, 8.84, 9.51]
    colors_3 = ["#2ecc71", "#27ae60", "#16a085"]

    bars3 = axes[2].bar(categories_3, vals_3, color=colors_3, edgecolor='black', lw=1.1, width=0.48)
    axes[2].axhline(y=350.0, color='#c0392b', linestyle='--', lw=1.8, label='Target Limit (< 350 ms)')
    axes[2].set_title("3. Predict API End-to-End Latency\n[ Target < 350 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[2].set_ylabel("Response Time (ms)", fontsize=10.5, fontweight='bold')
    axes[2].set_ylim(0, 400.0)
    axes[2].set_xticks(range(len(categories_3)))
    axes[2].set_xticklabels(categories_3, fontsize=10.5, fontweight='bold')
    axes[2].legend(loc="upper right", frameon=True, fontsize=9.2, facecolor='white', framealpha=0.95)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars3, vals_3):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 15.0, f"{val:.1f} ms\n(Realtime!)",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e8449')

    plt.suptitle("Experiment 5: AES-256 Encryption Overhead & API Response Latency Benchmark", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp5_latency_performance_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Exp 5 Latency Performance Visuals generated successfully!")

if __name__ == "__main__":
    generate_exp5_visuals()
