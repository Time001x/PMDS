"""
PMDS Row-by-Row App Test vs Ground-Truth Feature Comparison Generator
====================================================================
Generates a detailed table showing:
- Patient ID
- Real Clinical Ground-Truth Status (is_parkinson)
- Key Modality Feature Measurements (Tremor Freq, Gait CV, Finger ITI, UPDRS Score)
- PMDS App Output (Diagnosis, Risk %, Level 0-4)
- Match Status (✅ MATCH / ❌ MISMATCH)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(backend_dir, "datasets")
dataset_path = os.path.join(datasets_dir, "real_clinically_matched_1000_patients.csv")

def generate_row_comparison():
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    print("=============================================================================================================================================")
    print(" PMDS APP DETAILED ROW-BY-ROW TEST COMPARISON TABLE (DATASET FEATURES vs APP DIAGNOSIS vs GROUND-TRUTH)")
    print("=============================================================================================================================================")
    print(f"{'Patient ID':<12} | {'Ground-Truth (เฉลย)':<18} | {'สั่น (Tremor Hz)':<16} | {'เดิน (Gait CV %)':<16} | {'เคาะนิ้ว (Tap ITI)':<16} | {'UPDRS Score':<12} | {'แอปทำนาย (App Output)':<22} | {'Match'}")
    print("-" * 145)

    for idx, row in df.head(30).iterrows():
        p_id = row['subject_id']
        gt_num = int(row['is_parkinson'])
        gt_str = "🔴 พาร์กินสัน (1)" if gt_num == 1 else "🟢 ปกติ (0)"

        tremor_hz = f"{row['tremor_freq_hz']:.2f} Hz"
        gait_cv = f"{row['stride_cv_percent']:.1f} %"
        tap_iti = f"{row['tap_iti_mean_ms']:.1f} ms"
        updrs = f"{row['total_UPDRS']:.1f}"

        risk_scores = [
            row['voice_risk'],
            row['tremor_risk'],
            row['finger_risk'],
            row['gait_risk'],
            row['questionnaire_risk']
        ]

        high_risk_count = sum(r >= 0.35 for r in risk_scores)
        app_pred_num = 1 if high_risk_count >= 3 else 0
        avg_risk_percent = int(np.mean(risk_scores) * 100)

        if avg_risk_percent < 25:
            level = 0
            diag = "🟢 ไม่มีอาการ (Normal)"
        elif avg_risk_percent < 45:
            level = 1
            diag = "🟢 เล็กน้อย (Slight)"
        elif avg_risk_percent < 65:
            level = 2
            diag = "🟡 เสี่ยงปานกลาง (Mild)"
        elif avg_risk_percent < 80:
            level = 3
            diag = "🟠 เสี่ยงมาก (Moderate)"
        else:
            level = 4
            diag = "🔴 อาการรุนแรง (Severe)"

        app_str = f"{diag} ({avg_risk_percent}%)"
        is_match = (gt_num == app_pred_num)
        status = "✅ MATCH" if is_match else "❌ MISMATCH"

        print(f"{p_id:<12} | {gt_str:<18} | {tremor_hz:<16} | {gait_cv:<16} | {tap_iti:<16} | {updrs:<12} | {app_str:<22} | {status}")

    print("=" * 145)

if __name__ == "__main__":
    generate_row_comparison()
