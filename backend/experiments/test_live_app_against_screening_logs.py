"""
PMDS Fast Live App End-to-End Testing against pmds_app_live_screening_logs.csv
=============================================================================
Directly evaluates the live backend logic and models from `backend/main.py`
across all 1,000 screening records in `pmds_app_live_screening_logs.csv`
"""

import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import load_models, PredictPayload, predict, MODELS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"
dataset_path = os.path.join(BASE_DIR, "datasets", "pmds_app_live_screening_logs.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def test_live_app_pipeline():
    print("==========================================================================================")
    print(" PMDS LIVE APP END-TO-END SCREENING BENCHMARK (1,000 CALLS TO /predict LOGIC)")
    print("==========================================================================================")
    
    # 1. Load Live Models in Backend
    load_models()
    
    # 2. Load Dataset
    print(f"Reading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total screening sessions to test: {len(df)}")

    live_app_results = []
    
    for idx, row in df.iterrows():
        payload = PredictPayload(
            uid=str(row["user_id"]),
            Age=float(row["age"]),
            speechScore=float(row["speech_risk_score"]),
            tremorScore=float(row["tremor_risk_score"]),
            fingerScore=float(row["finger_risk_score"]),
            gaitScore=float(row["gait_risk_score"]),
            questionnaireScore=float(row["questionnaire_risk_score"])
        )
        
        # Execute live prediction logic from main.py
        res = predict(payload)
        
        live_app_results.append({
            "test_id": row["test_id"],
            "doctor_diagnosis": int(row["doctor_diagnosis"]),
            "app_level": res.level,
            "app_risk_score": res.riskScore,
            "app_risk_percent": res.riskPercent,
            "app_diagnosis": res.diagnosis,
            "app_confidence": res.confidence,
            "is_app_high_risk": 1 if res.level >= 3 else 0
        })

    df_results = pd.DataFrame(live_app_results)
    
    # Compute Clinical Metrics
    y_true = df_results["doctor_diagnosis"].values
    y_pred = df_results["is_app_high_risk"].values
    
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    fnr = fn / (fn + tp)
    fpr = fp / (tn + fp)

    print("\n==========================================================================================")
    print(" LIVE APP ACCURACY EVALUATION (1,000 SESSIONS EXECUTED ON REAL APP LOGIC)")
    print("==========================================================================================")
    print(f"Accuracy (ความแม่นยำรวมของแอป)       : {acc*100:.2f}% ({tp+tn}/{len(df)} ราย)")
    print(f"Sensitivity (ความไวในการตรวจจับโรค)  : {sens*100:.2f}% (ตรวจพบผู้ป่วย {tp} จาก {tp+fn} คน)")
    print(f"Specificity (ความจำเพาะคนปกติ)       : {spec*100:.2f}% (ระบุคนปกติถูกต้อง {tn} จาก {tn+fp} คน)")
    print(f"Precision (ความแม่นตรงเมื่อแจ้งเตือน)  : {prec*100:.2f}%")
    print(f"F1-Score                           : {f1:.4f}")
    print(f"False Negative Rate (หลุดตรวจ)      : {fnr*100:.2f}% ({fn} คน)")
    print(f"False Positive Rate (ทักมั่ว)       : {fpr*100:.2f}% ({fp} คน)")
    print("==========================================================================================")

    # 1. Confusion Matrix Heatmap
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cm_matrix = np.array([[tn, fp], [fn, tp]])
    cm_labels = np.array([[f"True Healthy (TN)\n{tn} คน ({tn/len(df)*100:.1f}%)", f"False Positive (FP)\n{fp} คน ({fp/len(df)*100:.1f}%)"],
                          [f"False Negative (FN)\n{fn} คน ({fn/len(df)*100:.1f}%)", f"True Parkinson (TP)\n{tp} คน ({tp/len(df)*100:.1f}%)"]])
    
    sns.heatmap(cm_matrix, annot=cm_labels, fmt="", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"fontsize": 11, "fontweight": "bold"},
                xticklabels=["App: Negative (Level 0-2)", "App: High Risk (Level 3-4)"],
                yticklabels=["Doctor: Healthy (0)", "Doctor: Parkinson (1)"])
    axes[0].set_title(f"A. Live PMDS App Confusion Matrix (N=1,000)\nAccuracy = {acc*100:.2f}%, Sensitivity = {sens*100:.2f}%", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Live App Diagnosis Output", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Doctor Clinical Diagnosis", fontsize=11, fontweight='bold')

    # 2. Diagnosis Breakdown by MDS-UPDRS Scale Level
    diag_order = ["ไม่มีอาการ", "เล็กน้อย", "เสี่ยงปานกลาง", "เสี่ยงมาก", "อาการรุนแรง"]
    colors_diag = ["#05C134", "#10B981", "#F59E0B", "#F97316", "#C10508"]
    
    counts_ordered = [df_results[df_results['app_diagnosis'] == d].shape[0] for d in diag_order]
    bars = axes[1].bar(diag_order, counts_ordered, color=colors_diag, edgecolor='black', lw=0.9, width=0.6)
    axes[1].set_title("B. Live App Diagnosis Distribution (1,000 Tests)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("App Screening Classification", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Number of Patients", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, max(counts_ordered) * 1.18)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, counts_ordered):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 15, f"{val} คน\n({val/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.suptitle("Live Application Testing Results on pmds_app_live_screening_logs.csv", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "live_app_screening_benchmark.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print(f"[SUCCESS] Live App Benchmark Figure generated: {artifact_path}")
    return df_results

if __name__ == "__main__":
    test_live_app_pipeline()
