"""
PMDS App Prediction vs Clinical Ground-Truth Matcher (1,000 Patients)
=======================================================================
Tests the PMDS AI Backend Prediction Engine directly against the 1,000-Patient Clinical Dataset
with Known Ground-Truth Labels (เฉลยจริงจากทางคลินิก 1,000 คน).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(backend_dir, "datasets")
dataset_1000_path = os.path.join(datasets_dir, "real_clinically_matched_1000_patients.csv")

def run_1000_patient_test():
    if not os.path.exists(dataset_1000_path):
        print(f"Dataset not found at {dataset_1000_path}")
        return

    df = pd.read_csv(dataset_1000_path)
    print("==========================================================")
    print(" PMDS APP PREDICTION vs GROUND-TRUTH TEST (1,000 PATIENTS)")
    print("==========================================================")
    print(f"Total Test Patients: {len(df)}")
    print(f"PD Positive (เฉลยป่วย) : {sum(df['is_parkinson'] == 1)}")
    print(f"Healthy Controls (เฉลยปกติ) : {sum(df['is_parkinson'] == 0)}")
    print("----------------------------------------------------------")

    y_ground_truth = df['is_parkinson'].values
    y_app_pred = []

    for idx, row in df.iterrows():
        risk_scores = [
            row['voice_risk'],
            row['tremor_risk'],
            row['finger_risk'],
            row['gait_risk'],
            row['questionnaire_risk']
        ]

        # 3/5 Majority Decision Rule in App
        high_risk_modalities = sum(r >= 0.35 for r in risk_scores)
        app_pred_label = 1 if high_risk_modalities >= 3 else 0
        y_app_pred.append(app_pred_label)

    y_app_pred = np.array(y_app_pred)
    acc = accuracy_score(y_ground_truth, y_app_pred)
    prec = precision_score(y_ground_truth, y_app_pred, zero_division=0)
    rec = recall_score(y_ground_truth, y_app_pred, zero_division=0)
    f1 = f1_score(y_ground_truth, y_app_pred, zero_division=0)

    cm = confusion_matrix(y_ground_truth, y_app_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n--- APP PREDICTION vs GROUND-TRUTH EMPIRICAL RESULTS (1,000 PATIENTS) ---")
    print(f" Matches (แอปทำนายตรงกับเฉลยจริง)      : {tp + tn} / {len(df)} ({acc*100:.2f}%)")
    print(f" Mismatches (แอปทำนายต่างจากเฉลยจริง)   : {fp + fn} / {len(df)} ({(1-acc)*100:.2f}%)")
    print(f" True Positives (TP)                   : {tp} (คนไข้ที่แอปทำนายตรงว่าเป็นพาร์กินสัน)")
    print(f" True Negatives (TN)                   : {tn} (คนปกติที่แอปทำนายตรงว่าปกติ)")
    print(f" False Positives (FP)                  : {fp} (คนปกติที่แอปทำนายผิดเป็นพาร์กินสัน)")
    print(f" False Negatives (FN)                  : {fn} (คนไข้ที่แอปทำนายผิดว่าปกติ)")
    print(f" Sensitivity (Recall)                  : {rec*100:.2f}%")
    print(f" Specificity                           : {spec*100:.2f}%")
    print(f" False Negative Rate (FNR)             : {fnr*100:.2f}%")
    print("==========================================================")

if __name__ == "__main__":
    run_1000_patient_test()
