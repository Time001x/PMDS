"""
PMDS Patient-by-Patient Ground-Truth Match Verification Table (Randomly Shuffled 1,000 Dataset)
==============================================================================================
Prints a detailed patient-by-patient verification table from the randomly shuffled 1,000-patient dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(backend_dir, "datasets")
multi_path = os.path.join(datasets_dir, "real_clinically_matched_1000_patients.csv")

def generate_patient_match_table():
    if not os.path.exists(multi_path):
        print(f"Dataset not found at {multi_path}")
        return

    df = pd.read_csv(multi_path)
    print("==========================================================================================================")
    print(" PMDS PATIENT-BY-PATIENT GROUND-TRUTH MATCH VERIFICATION TABLE (SHUFFLED DATASET SAMPLE)")
    print("==========================================================================================================")
    print(f"{'Patient ID':<14} | {'Age/Sex':<8} | {'Ground-Truth Label (เฉลยจริง)':<28} | {'App Prediction (คำทำนายแอป)':<28} | {'Match Status':<12}")
    print("-" * 108)

    match_count = 0
    mismatch_count = 0

    for idx, row in df.iterrows():
        p_id = row['subject_id']
        age_sex = f"{int(row['age'])}/{'M' if row['sex']==0 else 'F'}"
        gt_label_num = int(row['is_parkinson'])
        gt_str = "🔴 พาร์กินสัน (Positive)" if gt_label_num == 1 else "🟢 สุขภาพดี (Healthy)"

        risk_scores = [
            row['voice_risk'],
            row['tremor_risk'],
            row['finger_risk'],
            row['gait_risk'],
            row['questionnaire_risk']
        ]

        high_risk_count = sum(r >= 0.35 for r in risk_scores)
        app_pred_num = 1 if high_risk_count >= 3 else 0
        app_str = "🔴 พาร์กินสัน (Risk >= 3/5)" if app_pred_num == 1 else "🟢 สุขภาพดี (Risk < 3/5)"

        is_match = (gt_label_num == app_pred_num)
        status_str = "✅ MATCH" if is_match else "❌ MISMATCH"

        if is_match:
            match_count += 1
        else:
            mismatch_count += 1

        if idx < 30: # Print first 30 shuffled rows
            print(f"{p_id:<14} | {age_sex:<8} | {gt_str:<28} | {app_str:<28} | {status_str:<12}")

    print("..." + " " * 105)
    print("=" * 108)
    print(f"TOTAL PATIENTS TESTED : {len(df)}")
    print(f"MATCHED PREDICTIONS   : {match_count} / {len(df)} ({match_count/len(df)*100:.2f}%)")
    print(f"MISMATCHED PREDICTIONS: {mismatch_count} / {len(df)} ({mismatch_count/len(df)*100:.2f}%)")
    print("==========================================================================================================")

if __name__ == "__main__":
    generate_patient_match_table()
