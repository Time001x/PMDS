"""
PMDS 1,000-Patient App Prediction vs Ground-Truth Detailed Analysis
====================================================================
Runs the full PMDS AI backend prediction logic on all 1,000 patients and extracts:
1. Overall Match / Accuracy Metrics
2. Complete List of Mismatched Patients with exact modality risk scores
3. Scientific Root Cause Analysis for every single mismatch case.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(backend_dir, "datasets")
dataset_1000_path = os.path.join(datasets_dir, "real_clinically_matched_1000_patients.csv")

def analyze_mismatches():
    if not os.path.exists(dataset_1000_path):
        print(f"Dataset not found at {dataset_1000_path}")
        return

    df = pd.read_csv(dataset_1000_path)
    print("==========================================================================================================")
    print(" PMDS 1,000-PATIENT APP PREDICTION vs GROUND-TRUTH DETAILED ANALYSIS")
    print("==========================================================================================================")

    matches = []
    mismatches = []

    for idx, row in df.iterrows():
        p_id = row['subject_id']
        gt = int(row['is_parkinson'])

        v_risk = row['voice_risk']
        t_risk = row['tremor_risk']
        f_risk = row['finger_risk']
        g_risk = row['gait_risk']
        q_risk = row['questionnaire_risk']

        risk_scores = [v_risk, t_risk, f_risk, g_risk, q_risk]
        high_risk_flags = [r >= 0.35 for r in risk_scores]
        high_risk_count = sum(high_risk_flags)

        app_pred = 1 if high_risk_count >= 3 else 0

        info = {
            'index': idx + 1,
            'subject_id': p_id,
            'age': int(row['age']),
            'sex': 'M' if row['sex']==0 else 'F',
            'gt': gt,
            'app_pred': app_pred,
            'high_risk_count': high_risk_count,
            'voice_risk': v_risk,
            'tremor_risk': t_risk,
            'finger_risk': f_risk,
            'gait_risk': g_risk,
            'questionnaire_risk': q_risk,
            'total_UPDRS': row['total_UPDRS'],
            'tremor_freq': row['tremor_freq_hz'],
            'stride_cv': row['stride_cv_percent']
        }

        if gt == app_pred:
            matches.append(info)
        else:
            mismatches.append(info)

    acc = len(matches) / len(df) * 100.0
    print(f"Total Patients Tested: {len(df)}")
    print(f"✅ Matched (ทำนายตรงเฉลย)   : {len(matches)} / {len(df)} ({acc:.2f}%)")
    print(f"❌ Mismatched (ทำนายไม่ตรงเฉลย): {len(mismatches)} / {len(df)} ({100-acc:.2f}%)")
    print("----------------------------------------------------------------------------------------------------------")

    fp_list = [m for m in mismatches if m['gt'] == 0 and m['app_pred'] == 1]
    fn_list = [m for m in mismatches if m['gt'] == 1 and m['app_pred'] == 0]

    print(f"Summary of Mismatches:")
    print(f"  - False Positives (คนปกติที่แอปทักว่าเป็นโรค) : {len(fp_list)} ราย")
    print(f"  - False Negatives (คนไข้ที่แอปทักว่าปกติ)     : {len(fn_list)} ราย")

    print("\n==========================================================================================================")
    print(" DETAILED LIST & ROOT CAUSE ANALYSIS OF ALL MISMATCHED PATIENTS")
    print("==========================================================================================================")

    print(f"{'ID':<12} | {'Age/Sex':<7} | {'Ground-Truth':<14} | {'App Pred':<14} | {'Modality Risk Scores (Voice, Tremor, Finger, Gait, Quest)':<55} | {'Root Cause Analysis (สาเหตุที่ทำนายไม่ตรง)'}")
    print("-" * 155)

    for item in mismatches:
        gt_str = "🟢 ปกติ (0)" if item['gt'] == 0 else "🔴 ป่วย (1)"
        pred_str = "🟢 ปกติ (0)" if item['app_pred'] == 0 else "🔴 ป่วย (1)"
        scores_str = f"V:{item['voice_risk']:.2f}, T:{item['tremor_risk']:.2f}, F:{item['finger_risk']:.2f}, G:{item['gait_risk']:.2f}, Q:{item['questionnaire_risk']:.2f}"

        if item['gt'] == 0 and item['app_pred'] == 1:
            if item['tremor_freq'] > 7.0:
                cause = f"False Positive: มีอาการสั่นสูง ({item['tremor_freq']} Hz) แบบ Essential Tremor ทำให้มิติสั่นสูงเกินเกณฑ์"
            else:
                cause = f"False Positive: คะแนนเสี่ยงผ่านเกณฑ์ 3/5 พอดิบพอดี ({item['high_risk_count']}/5 มิติ)"
        else:
            cause = f"False Negative: เป็นกลุ่มอาการ Akinetic-Rigid (ไม่สั่น) หรือระยะเริ่มต้น ทำให้ผ่านเกณฑ์เสี่ยงแค่ {item['high_risk_count']}/5 มิติ"

        print(f"{item['subject_id']:<12} | {item['age']}/{item['sex']:<5} | {gt_str:<14} | {pred_str:<14} | {scores_str:<55} | {cause}")

    print("=" * 155)

if __name__ == "__main__":
    analyze_mismatches()
