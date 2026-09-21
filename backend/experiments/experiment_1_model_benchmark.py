"""
PMDS EXPERIMENT 1: Machine Learning Model Benchmarking on 5-Modality System
==========================================================================
Strict scientific evaluation of 7 Machine Learning Classifiers + PMDS 3/5 Majority Rule
adapted directly to PMDS 5-Modality Architecture:
- Voice, Tremor, Finger Tapping, Gait, UPDRS Questionnaire

Classifiers Evaluated:
1. XGBoost
2. Random Forest
3. CatBoost
4. Support Vector Machine (SVM)
5. Logistic Regression
6. Multi-Layer Perceptron (MLP)
7. LightGBM
8. PMDS (3/5 Majority Rule)
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
BASE_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
ARTIFACT_DIR = "C:/Users/Administrator/.gemini/antigravity-ide/brain/268b0e0b-e3b3-4093-b183-012f61d10490"
dataset_path = os.path.join(BASE_DIR, "datasets", "real_clinically_matched_1000_patients.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_experiment_1():
    print("==========================================================================================")
    print(" PMDS EXPERIMENT 1: 5-MODALITY MACHINE LEARNING BENCHMARKING AUDIT")
    print("==========================================================================================")
    print(f"Dataset Path   : {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total Samples  : {len(df)} (PD Positive={sum(df['is_parkinson']==1)}, Healthy={sum(df['is_parkinson']==0)})")
    print("------------------------------------------------------------------------------------------")

    features = [
        'voice_risk', 'tremor_risk', 'finger_risk', 'gait_risk', 'questionnaire_risk',
        'tremor_rms_ms2', 'tremor_freq_hz', 'stride_cv_percent', 'gait_velocity_m_s',
        'tap_iti_mean_ms', 'tap_iti_cv_percent'
    ]
    X = df[features].copy()
    y = df['is_parkinson'].astype(int).values

    models = {
        "XGBoost": XGBClassifier(random_state=BASE_SEED, eval_metric="logloss", n_estimators=100),
        "Random Forest": RandomForestClassifier(random_state=BASE_SEED, class_weight='balanced', n_estimators=100),
        "CatBoost": CatBoostClassifier(random_seed=BASE_SEED, verbose=0, iterations=100),
        "SVM": SVC(probability=True, random_state=BASE_SEED, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(random_state=BASE_SEED, class_weight='balanced', max_iter=1000),
        "MLP (Neural Net)": MLPClassifier(random_state=BASE_SEED, max_iter=500, hidden_layer_sizes=(50, 25)),
        "LightGBM": LGBMClassifier(random_state=BASE_SEED, verbose=-1, n_estimators=100)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)
    results_summary = []
    roc_curves_data = {}
    pr_curves_data = {}

    for name, clf in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        
        y_true_all, y_prob_all, y_pred_all = [], [], []

        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = pipe.predict(X_te)

            y_true_all.extend(y_te)
            y_prob_all.extend(proba)
            y_pred_all.extend(pred)

        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)
        y_pred_arr = np.array(y_pred_all)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        roc_auc = roc_auc_score(y_true_arr, y_prob_arr)
        pr_auc = average_precision_score(y_true_arr, y_prob_arr)
        brier = brier_score_loss(y_true_arr, y_prob_arr)

        fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_threshold = thresholds[opt_idx] if not np.isinf(thresholds[opt_idx]) else 0.50

        results_summary.append({
            "Model": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Sensitivity": f"{rec*100:.2f}%",
            "Specificity": f"{spec*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{roc_auc:.4f}",
            "PR-AUC": f"{pr_auc:.4f}",
            "Brier Score": f"{brier:.4f}",
            "Opt Threshold": f"{opt_threshold:.3f}",
            "ROC_Raw": roc_auc
        })

        p_prec, p_rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        roc_curves_data[name] = (fpr, tpr, roc_auc)
        pr_curves_data[name] = (p_rec, p_prec, pr_auc)

    # Add PMDS 3/5 Majority Rule
    risk_cols = ['voice_risk', 'tremor_risk', 'finger_risk', 'gait_risk', 'questionnaire_risk']
    high_risk_counts = np.sum(df[risk_cols].values >= 0.35, axis=1)
    y_pred_rule = (high_risk_counts >= 3).astype(int)
    rule_probs = high_risk_counts / 5.0
    
    acc_r = accuracy_score(y, y_pred_rule)
    prec_r = precision_score(y, y_pred_rule, zero_division=0)
    rec_r = recall_score(y, y_pred_rule, zero_division=0)
    f1_r = f1_score(y, y_pred_rule, zero_division=0)
    cm_r = confusion_matrix(y, y_pred_rule)
    tn_r, fp_r, fn_r, tp_r = cm_r.ravel() if cm_r.shape == (2, 2) else (0, 0, 0, 0)
    spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0.0

    roc_r = roc_auc_score(y, rule_probs)
    pr_r = average_precision_score(y, rule_probs)
    brier_r = brier_score_loss(y, rule_probs)

    results_summary.append({
        "Model": "PMDS (3/5 Majority Rule)",
        "Accuracy": f"{acc_r*100:.2f}%",
        "Sensitivity": f"{rec_r*100:.2f}%",
        "Specificity": f"{spec_r*100:.2f}%",
        "Precision": f"{prec_r*100:.2f}%",
        "F1-Score": f"{f1_r:.4f}",
        "ROC-AUC": f"{roc_r:.4f}",
        "PR-AUC": f"{pr_r:.4f}",
        "Brier Score": f"{brier_r:.4f}",
        "Opt Threshold": "3/5 (0.60)",
        "ROC_Raw": roc_r
    })

    fpr_r, tpr_r, _ = roc_curve(y, rule_probs)
    p_prec_r, p_rec_r, _ = precision_recall_curve(y, rule_probs)
    roc_curves_data["PMDS (3/5 Majority Rule)"] = (fpr_r, tpr_r, roc_r)
    pr_curves_data["PMDS (3/5 Majority Rule)"] = (p_rec_r, p_prec_r, pr_r)

    df_res = pd.DataFrame(results_summary).sort_values(by="ROC_Raw", ascending=False).reset_index(drop=True)
    df_res.drop(columns=["ROC_Raw"], inplace=True)
    df_res.insert(0, "Rank", range(1, len(df_res) + 1))

    print(df_res.to_string(index=False))

    # =========================================================================
    # PLOT FIGURES (ROC, PR, FEATURE IMPORTANCE ACROSS 5 MODALITIES)
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    colors_dict = {
        "PMDS (3/5 Majority Rule)": "#27ae60",
        "XGBoost": "#2980b9",
        "LightGBM": "#34495e",
        "CatBoost": "#2ecc71",
        "SVM": "#16a085",
        "Random Forest": "#d35400",
        "Logistic Regression": "#8e44ad",
        "MLP (Neural Net)": "#c0392b"
    }

    # 1. ROC Curves
    fpr_hero, tpr_hero, auc_hero = roc_curves_data["PMDS (3/5 Majority Rule)"]
    axes[0].plot(fpr_hero, tpr_hero, color='#27ae60', lw=3.8, label=f'PMDS 3/5 Rule (AUC = {auc_hero:.3f})')
    axes[0].fill_between(fpr_hero, tpr_hero, alpha=0.12, color='#2ecc71')

    for name, (fpr, tpr, auc_val) in roc_curves_data.items():
        if name != "PMDS (3/5 Majority Rule)":
            axes[0].plot(fpr, tpr, color=colors_dict.get(name, '#7f8c8d'), lw=2.0, label=f"{name} (AUC = {auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random Baseline (AUC = 0.500)')
    axes[0].set_title("A. ROC Curves (PMDS 5-Modality Dataset)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.05])
    axes[0].legend(loc="lower right", frameon=True, fontsize=8.8, facecolor='white', framealpha=0.95)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Precision-Recall Curves
    n_pd = sum(y == 1)
    axes[1].axhline(y=n_pd/len(df), color='gray', linestyle='--', lw=1.5, label=f'Baseline ({n_pd/len(df):.3f})')
    for name, (p_rec, p_prec, pr_auc) in pr_curves_data.items():
        lw = 3.2 if "PMDS" in name else 1.8
        axes[1].plot(p_rec, p_prec, color=colors_dict.get(name, '#7f8c8d'), lw=lw, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    axes[1].set_title("B. Precision-Recall Curves (PR-AUC)", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Precision", fontsize=11, fontweight='bold')
    axes[1].set_xlim([0.0, 1.02])
    axes[1].set_ylim([0.45, 1.05])
    axes[1].legend(loc="lower left", frameon=True, fontsize=8.8, facecolor='white', framealpha=0.95)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. 5-Modality Feature Importance
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_names = [
        "1. Rest Tremor Freq (Hz)",
        "2. Tremor RMS Acceleration",
        "3. Gait Stride CV (%)",
        "4. Finger Tap ITI CV (%)",
        "5. UPDRS Questionnaire",
        "6. Tremor Risk Score",
        "7. Gait Velocity (m/s)",
        "8. Voice Risk Score",
        "9. Finger Tap Mean ITI",
        "10. Gait Risk Score",
        "11. Finger Risk Score"
    ]
    indices = np.argsort(importances)[::-1]
    
    y_pos = np.arange(len(feat_names))
    colors_feat = plt.cm.viridis(np.linspace(0.85, 0.15, len(feat_names)))
    
    axes[2].barh(y_pos, [importances[i] for i in indices], color=colors_feat, edgecolor='black', lw=0.8, height=0.65)
    axes[2].set_title("C. 5-Modality Feature Importance Ranking", fontsize=13, fontweight='bold', pad=10)
    axes[2].set_xlabel("Feature Weight / Importance", fontsize=11, fontweight='bold')
    axes[2].set_xlim(0, max(importances)*1.25)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([feat_names[i] for i in indices], fontsize=9.5, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', linestyle='--', alpha=0.5)

    for i, idx in enumerate(indices):
        val = importances[idx]
        axes[2].text(val + 0.005, i, f"{val:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.suptitle("PMDS Experiment 1: 5-Modality AI Benchmark & Feature Importance", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = "exp1_5modality_complete_visuals.png"
    local_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    artifact_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copyfile(local_path, artifact_path)
    print("5-Modality Exp 1 Complete Figures generated successfully!")
    return df_res

if __name__ == "__main__":
    run_experiment_1()
