"""
PMDS EXPERIMENT 2: Model Probability Calibration Audit
======================================================
Strict scientific evaluation of Model Probability Calibration:
- Calibration Curve / Reliability Diagrams
- Brier Score Loss (Overall Calibration Loss)
- Expected Calibration Error (ECE)
- Predicted Probability vs Actual Outcome Distribution

Evaluated on REAL CLINICAL DATASET:
UCI Parkinson Telemonitoring Dataset (5,875 real patient recordings, 42 patients)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTE = None

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
BASE_SEED = 42

def compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece

def run_experiment_2():
    print("==========================================================")
    print(" PMDS EXPERIMENT 2: MODEL CALIBRATION AUDIT")
    print("==========================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets")
    dataset_path = os.path.join(datasets_dir, "real_clinical_telemonitoring_processed.csv")

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parkinsons.data")

    df = pd.read_csv(dataset_path)

    if 'subject#' in df.columns or 'subject_id' in df.columns:
        df['subject_id'] = df['subject_id'] if 'subject_id' in df.columns else df['subject#'].apply(lambda x: f"pat_{x}")
        target_col = 'is_parkinson' if 'is_parkinson' in df.columns else 'status'
        exclude_cols = ['subject#', 'subject_id', 'is_parkinson', 'status', 'test_time', 'motor_UPDRS', 'total_UPDRS']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        is_synthetic = False
    else:
        df['subject_id'] = [f"sub_{i}" for i in range(len(df))]
        target_col = df.columns[-1]
        feature_cols = df.columns[:-1].tolist()
        is_synthetic = True

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values
    groups = df['subject_id'].values

    print(f"Dataset      : {dataset_path}")
    print(f"Dataset Scope: {'REAL CLINICAL DATASET (UCI Telemonitoring 5,875 recordings)' if not is_synthetic else 'SYNTHETIC'}")
    print(f"Total Samples: {len(X)}")
    print(f"Subjects     : {len(np.unique(groups))}")
    print("----------------------------------------------------------")

    models = {
        "CatBoost": ImbPipeline([("scaler", StandardScaler()), ("classifier", CatBoostClassifier(auto_class_weights="Balanced", random_seed=BASE_SEED, verbose=0))]),
        "LightGBM": ImbPipeline([("scaler", StandardScaler()), ("classifier", LGBMClassifier(class_weight="balanced", random_state=BASE_SEED, verbose=-1))]),
        "Random Forest": ImbPipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(class_weight="balanced", random_state=BASE_SEED))]),
        "Multi-Layer Perceptron (MLP)": ImbPipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(random_state=BASE_SEED, max_iter=500))]),
        "XGBoost": ImbPipeline([("scaler", StandardScaler()), ("classifier", XGBClassifier(random_state=BASE_SEED, eval_metric="logloss"))]),
        "Support Vector Machine (SVM)": ImbPipeline([("scaler", StandardScaler()), ("classifier", SVC(class_weight="balanced", probability=True, random_state=BASE_SEED))]),
        "Logistic Regression": ImbPipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", random_state=BASE_SEED, max_iter=1000))])
    }

    cv = StratifiedGroupKFold(n_splits=5)
    calibration_summary = []

    for name, pipeline in models.items():
        y_true_all = []
        y_prob_all = []

        for train_idx, test_idx in cv.split(X, y, groups=groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipeline.fit(X_tr, y_tr)
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_te)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                y_prob = pipeline.decision_function(X_te)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
            else:
                y_prob = pipeline.predict(X_te).astype(float)

            y_true_all.extend(y_te)
            y_prob_all.extend(y_prob)

        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)

        brier = brier_score_loss(y_true_arr, y_prob_arr)
        ece = compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=10)
        prob_pred, prob_true = calibration_curve(y_true_arr, y_prob_arr, n_bins=5)
        curve_dev = np.mean(np.abs(prob_pred - prob_true))

        calibration_summary.append({
            "Model": name,
            "Brier_Score_Loss": f"{brier:.4f}",
            "Expected_Calibration_Error (ECE)": f"{ece:.4f}",
            "Calibration_Curve_Mean_Dev": f"{curve_dev:.4f}",
            "Mean_Predicted_Probability": f"{y_prob_arr.mean():.4f}",
            "Actual_Positive_Ratio": f"{y_true_arr.mean():.4f}",
            "Brier_Raw": brier
        })

    df_cal = pd.DataFrame(calibration_summary).sort_values(by="Brier_Raw")
    df_cal.drop(columns=["Brier_Raw"], inplace=True)
    print("\n==========================================================")
    print(" EXPERIMENT 2 EMPIRICAL CALIBRATION SUMMARY (REAL DATASET)")
    print("==========================================================")
    print(df_cal.to_string(index=False))
    return df_cal

if __name__ == "__main__":
    run_experiment_2()
