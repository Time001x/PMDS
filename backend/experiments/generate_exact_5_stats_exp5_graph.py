"""
Generate Experiment 5 Visuals Exactly Containing the 5 Requested Columns:
- Mean (ms)
- Median (ms)
- Std Dev (ms)
- P95 (ms)
- Target Limit (Red Dashed Line)
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

def generate_exact_5_stats_graph():
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    stat_labels = ["Mean", "Median", "Std Dev", "P95"]
    colors = ["#27ae60", "#2ecc71", "#f39c12", "#2980b9"]

    # =========================================================================
    # Panel 1: AES-256 Encryption/Decryption Overhead
    # =========================================================================
    vals_1 = [0.032, 0.016, 0.147, 0.018]
    target_1 = 3.50

    bars1 = axes[0].bar(stat_labels, vals_1, color=colors, edgecolor='black', lw=1.1, width=0.52)
    axes[0].axhline(y=target_1, color='#c0392b', linestyle='--', lw=2.0, label=f'Target Limit (< {target_1:.1f} ms)')
    
    axes[0].set_title("1. AES-256 Security Overhead\n[ Target Limit < 3.5 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[0].set_ylabel("Time (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 4.2)
    axes[0].set_xticks(range(len(stat_labels)))
    axes[0].set_xticklabels(stat_labels, fontsize=10.5, fontweight='bold')
    axes[0].legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, vals_1):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.12, f"{val:.3f}",
                     ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#2c3e50')

    # =========================================================================
    # Panel 2: Database Save Test Result Latency
    # =========================================================================
    vals_2 = [2.793, 2.652, 0.865, 3.241]
    target_2 = 12.00

    bars2 = axes[1].bar(stat_labels, vals_2, color=colors, edgecolor='black', lw=1.1, width=0.52)
    axes[1].axhline(y=target_2, color='#c0392b', linestyle='--', lw=2.0, label=f'Target Limit (< {target_2:.1f} ms)')
    
    axes[1].set_title("2. Database Save Latency\n(SQLite pmds_secure.db) [ Target < 12 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[1].set_ylabel("Time (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 14.5)
    axes[1].set_xticks(range(len(stat_labels)))
    axes[1].set_xticklabels(stat_labels, fontsize=10.5, fontweight='bold')
    axes[1].legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, vals_2):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.35, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    # =========================================================================
    # Panel 3: Predict API End-to-End Latency
    # =========================================================================
    vals_3 = [9.351, 8.835, 3.457, 9.509]
    target_3 = 350.00

    bars3 = axes[2].bar(stat_labels, vals_3, color=colors, edgecolor='black', lw=1.1, width=0.52)
    axes[2].axhline(y=target_3, color='#c0392b', linestyle='--', lw=2.0, label=f'Target Limit (< {target_3:.0f} ms)')
    
    axes[2].set_title("3. Predict API End-to-End Latency\n[ Real-Time / Target < 350 ms ]", fontsize=12.5, fontweight='bold', pad=12)
    axes[2].set_ylabel("Time (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 420.0)
    axes[2].set_xticks(range(len(stat_labels)))
    axes[2].set_xticklabels(stat_labels, fontsize=10.5, fontweight='bold')
    axes[2].legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='white', framealpha=0.95)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars3, vals_3):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 14.0, f"{val:.2f}",
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    plt.suptitle("Experiment 5: System Latency & Security Overhead (Mean, Median, Std Dev, P95 vs Target Limit)", fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp5_latency_performance_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Exact 5 Stats Exp 5 Visuals generated successfully!")

if __name__ == "__main__":
    generate_exact_5_stats_graph()
