"""
PMDS EXPERIMENT 3: 5-Modality Integration & Majority Rule Evaluation
=====================================================================
Evaluates 5-Modality Integrated Screening (Voice, Tremor, Finger Tap, Gait, Questionnaire)
using 3/5 Majority Decision Rule on Clinically-Matched Multimodal Dataset.

Computes Full Confusion Matrix Metrics:
- True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
- Accuracy, Sensitivity (Recall), Specificity, F1-Score
- False Negative Rate (FNR = FN / (TP + FN))
- Diagnostic Odds Ratio (DOR = (TP * TN) / (FP * FN))
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def compute_diagnostic_odds_ratio(tp, tn, fp, fn):
    if fp * fn == 0:
        return float('nan')
    return float((tp * tn) / (fp * fn))

def run_experiment_3():
    print("==========================================================")
    print(" PMDS EXPERIMENT 3: MULTIMODAL 3/5 MAJORITY RULE EVALUATION")
    print("==========================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets")
    dataset_path = os.path.join(datasets_dir, "real_clinically_matched_multimodal.csv")

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(datasets_dir, "real_multimodal_clinical_composite.csv")

    df = pd.read_csv(dataset_path)

    print(f"Dataset Loaded : {dataset_path}")
    print(f"Total Subjects : {len(df)}")
    print(f"PD Positive    : {sum(df['is_parkinson'] == 1)}")
    print(f"Healthy Control: {sum(df['is_parkinson'] == 0)}")
    print("----------------------------------------------------------")

    y_true = df['is_parkinson'].astype(int).values

    # Extract 5 modality risk scores (0.0 to 1.0)
    risk_cols = ['voice_risk', 'tremor_risk', 'finger_risk', 'gait_risk', 'questionnaire_risk']

    modal_predictions = []
    for col in risk_cols:
        vals = df[col].values
        # Decision threshold per modality (Risk >= 0.35 is High Risk)
        modal_predictions.append(vals >= 0.35)

    modal_matrix = np.column_stack(modal_predictions)
    high_risk_counts = np.sum(modal_matrix, axis=1)

    # 3 out of 5 majority decision rule
    y_pred = (high_risk_counts >= 3).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    dor = compute_diagnostic_odds_ratio(tp, tn, fp, fn)

    results = {
        "Dataset_Path": dataset_path,
        "True_Positives (TP)": int(tp),
        "True_Negatives (TN)": int(tn),
        "False_Positives (FP)": int(fp),
        "False_Negatives (FN)": int(fn),
        "Accuracy": f"{acc:.4f}",
        "Sensitivity (Recall)": f"{rec:.4f}",
        "Specificity": f"{spec:.4f}",
        "F1_Score": f"{f1:.4f}",
        "False_Negative_Rate (FNR)": f"{fnr:.4f}",
        "Diagnostic_Odds_Ratio (DOR)": f"{dor:.4f}" if not np.isnan(dor) else "N/A"
    }

    print("\n==========================================================")
    print(" EXPERIMENT 3 MULTIMODAL 3/5 MAJORITY RULE EMPIRICAL RESULTS")
    print("==========================================================")
    for k, v in results.items():
        print(f" {k:<30} : {v}")

    return results

if __name__ == "__main__":
    run_experiment_3()
