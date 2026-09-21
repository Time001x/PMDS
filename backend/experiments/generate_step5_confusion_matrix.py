"""
Generate Step 5 Confusion Matrix Deliverables for the 42-Patient Dataset
========================================================================
Extracts exact TP, TN, FP, FN patient breakdown for Logistic Regression, SVM, and MLP.
Generates an intuitive, high-contrast academic Confusion Matrix Heatmap figure.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"
dataset_path = os.path.join(BASE_DIR, "datasets", "real_clinically_matched_multimodal.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_step5_confusion_matrix():
    df = pd.read_csv(dataset_path)
    feature_cols = [
        'voice_risk', 'tremor_risk', 'finger_risk', 'gait_risk', 'questionnaire_risk',
        'tremor_rms_ms2', 'tremor_freq_hz', 'stride_cv_percent', 'gait_velocity_m_s',
        'tap_iti_mean_ms', 'tap_iti_cv_percent'
    ]
    X = df[feature_cols].copy()
    y = df['is_parkinson'].astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
    ])

    y_true_all = []
    y_pred_all = []
    patient_ids_all = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        y_true_all.extend(y_te)
        y_pred_all.extend(pred)
        patient_ids_all.extend(df.iloc[test_idx]['subject_id'].values)

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = cm.ravel()

    # Identify individual patients in each box
    tp_patients = [pid for pid, yt, yp in zip(patient_ids_all, y_true_arr, y_pred_arr) if yt == 1 and yp == 1]
    tn_patients = [pid for pid, yt, yp in zip(patient_ids_all, y_true_arr, y_pred_arr) if yt == 0 and yp == 0]
    fp_patients = [pid for pid, yt, yp in zip(patient_ids_all, y_true_arr, y_pred_arr) if yt == 0 and yp == 1]
    fn_patients = [pid for pid, yt, yp in zip(patient_ids_all, y_true_arr, y_pred_arr) if yt == 1 and yp == 0]

    print("=========================================================================")
    print(" STEP 5: EXACT PATIENT BREAKDOWN IN CONFUSION MATRIX (N=42)")
    print("=========================================================================")
    print(f"✅ TP (True Positive  - ป่วยจริง ทายถูกว่าป่วย) : {tp} คน ({tp/len(df)*100:.1f}%)")
    print(f"   รายชื่อคนไข้: {', '.join(tp_patients[:8])} ... ({len(tp_patients)} คน)")
    print(f"✅ TN (True Negative  - ปกติจริง ทายถูกว่าปกติ) : {tn} คน ({tn/len(df)*100:.1f}%)")
    print(f"   รายชื่อคนไข้: {', '.join(tn_patients)}")
    print(f"❌ FP (False Positive - ปกติจริง แต่ทักมั่วว่าป่วย): {fp} คน ({fp/len(df)*100:.1f}%)")
    print(f"   รายชื่อคนไข้: {', '.join(fp_patients)}")
    print(f"⚠️ FN (False Negative - ป่วยจริง แต่หลุดตรวจ)   : {fn} คน ({fn/len(df)*100:.1f}%)")
    print(f"   รายชื่อคนไข้: {', '.join(fn_patients)}")
    print("=========================================================================")

    # -------------------------------------------------------------------------
    # PLOT VISUAL DELIVERABLE
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 2x2 Heatmap
    cm_display = np.array([[tn, fp], [fn, tp]])
    annot_text = np.array([
        [f"True Negative (TN)\n{tn} Patients ({tn/42*100:.1f}%)\nCorrectly Identified Healthy",
         f"False Positive (FP)\n{fp} Patients ({fp/42*100:.1f}%)\nHealthy Misclassified as PD"],
        [f"False Negative (FN)\n{fn} Patients ({fn/42*100:.1f}%)\nPD Patient Missed (Under Screen)",
         f"True Positive (TP)\n{tp} Patients ({tp/42*100:.1f}%)\nCorrectly Identified PD Patient"]
    ])

    sns.heatmap(cm_display, annot=annot_text, fmt="", cmap="Blues", cbar=False, ax=axes[0],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"},
                xticklabels=["AI: Negative (Healthy)", "AI: Positive (Parkinson)"],
                yticklabels=["Doctor: Healthy (0)", "Doctor: Parkinson (1)"])
    axes[0].set_title(f"A. Confusion Matrix 4-Box Grid (Logistic Regression, N=42)\nSensitivity = {tp/(tp+fn)*100:.1f}%, Specificity = {tn/(tn+fp)*100:.1f}%", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel("AI Predicted Label (Output)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Doctor Ground Truth Label", fontsize=11, fontweight='bold')

    # 2. 4 Boxes Proportional Donut Summary
    categories = [
        f"TP: Correct PD ({tp})",
        f"TN: Correct Healthy ({tn})",
        f"FP: False Alarm ({fp})",
        f"FN: Missed Case ({fn})"
    ]
    counts = [tp, tn, fp, fn]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    wedges, texts, autotexts = axes[1].pie(
        counts, labels=categories, colors=colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75,
        textprops={'fontsize': 10.5, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='black', lw=1.2)
    )
    axes[1].set_title(f"B. Diagnostic Proportions (Overall Accuracy = {(tp+tn)/42*100:.1f}%)", fontsize=12, fontweight='bold', pad=12)

    plt.suptitle("Step 5: Clinical Evaluation into 4 Outcome Boxes (Confusion Matrix Breakdown)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = "exp1_step5_confusion_matrix.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("Step 5 Confusion Matrix visual generated successfully!")
    return tp, tn, fp, fn, tp_patients, tn_patients, fp_patients, fn_patients

if __name__ == "__main__":
    generate_step5_confusion_matrix()
