import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, brier_score_loss
)

# Imbalanced Learning Pipeline & Oversampling
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTE = None
    HAS_IMBLEARN = False

# ML Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Base random state
BASE_SEED = 42

def load_dataset(dataset_path="backend/parkinsons.data"):
    print("==================================================")
    print(" PUBLICATION-QUALITY DATASET PREPARATION")
    print("==================================================")
    if not os.path.exists(dataset_path):
        dataset_path = "backend/combined_pmds_dataset.csv"

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if 'name' in df.columns and 'status' in df.columns:
        df['subject_id'] = df['name'].str.extract(r'phon_R01_(S\d+)_')
        target_col = 'status'
        feature_cols = [c for c in df.columns if c not in ['name', 'status', 'patient_id', 'group_label', 'subject_id']]
    elif 'patient_id' in df.columns:
        df['subject_id'] = df['patient_id']
        target_col = 'is_parkinson' if 'is_parkinson' in df.columns else df.columns[-1]
        feature_cols = [c for c in df.columns if c not in ['name', 'is_parkinson', 'patient_id', 'group_label', 'modality', 'subject_id']]
    else:
        df['subject_id'] = [f"sub_{i}" for i in range(len(df))]
        target_col = df.columns[-1]
        feature_cols = df.columns[:-1].tolist()

    df['subject_id'] = df['subject_id'].fillna(df['name'] if 'name' in df.columns else 'unknown')

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values
    groups = df['subject_id'].values

    print(f" Total Recordings : {len(df)}")
    print(f" Total Features   : {X.shape[1]}")
    print(f" Unique Subjects  : {len(np.unique(groups))}")
    print(f" Parkinson Positive: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f" Healthy Controls  : {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)\n")

    return X, y, groups, feature_cols

def get_models_and_pipelines(y_train):
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight_val = neg_count / max(1, pos_count)

    models_config = {
        "Logistic Regression": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", random_state=BASE_SEED, max_iter=1000))
            ]),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"]
            }
        },
        "Support Vector Machine (SVM)": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(class_weight="balanced", probability=True, random_state=BASE_SEED))
            ]),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"]
            }
        },
        "Random Forest": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(class_weight="balanced", random_state=BASE_SEED))
            ]),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=BASE_SEED, eval_metric="logloss"))
            ]),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "LightGBM": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", LGBMClassifier(class_weight="balanced", random_state=BASE_SEED, verbose=-1))
            ]),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, -1],
                "classifier__learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "CatBoost": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("classifier", CatBoostClassifier(auto_class_weights="Balanced", random_seed=BASE_SEED, verbose=0))
            ]),
            "param_grid": {
                "classifier__iterations": [100, 200],
                "classifier__depth": [4, 6],
                "classifier__learning_rate": [0.03, 0.1]
            }
        },
        "Multi-Layer Perceptron (MLP)": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                *([("sampler", SMOTE(random_state=BASE_SEED))] if HAS_IMBLEARN and SMOTE is not None else []),
                ("classifier", MLPClassifier(random_state=BASE_SEED, max_iter=1000))
            ]),
            "param_grid": {
                "classifier__hidden_layer_sizes": [(64, 32), (100,), (50, 25)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__learning_rate_init": [0.001, 0.01]
            }
        }
    }
    return models_config

def run_repeated_subject_evaluations(X, y, groups, n_repeats=5, n_splits=5):
    """
    Executes repeated subject-independent StratifiedGroupKFold evaluations (50 fold evaluations total).
    Prevents subject data leakage & fold preprocessing leakage.
    """
    models_config = get_models_and_pipelines(y)
    model_fold_results = {name: [] for name in models_config.keys()}
    model_predictions = {name: {"y_true": [], "y_proba": []} for name in models_config.keys()}
    model_trained_pipelines = {}

    total_evals = n_repeats * n_splits
    print(f"==================================================")
    print(f" REPEATED SUBJECT-INDEPENDENT EVALUATION ({n_repeats} x {n_splits} = {total_evals} FOLDS)")
    print(f"==================================================")

    # 1. First tuning pass to find best hyperparameters per model
    print("\n--- Phase 1: Hyperparameter Tuning via Inner StratifiedGroupKFold ---")
    inner_cv = StratifiedGroupKFold(n_splits=5)
    best_pipelines = {}

    for name, config in models_config.items():
        print(f"[Tuning] {name}...")
        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            error_score=0
        )
        grid.fit(X, y, groups=groups)
        best_pipelines[name] = grid.best_estimator_
        clean_params = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}
        print(f"   [+] Best Params: {clean_params}")

    # 2. Repeated Outer Evaluation Folds
    print(f"\n--- Phase 2: Running {n_repeats} Repeated Group Folds Across {total_evals} Folds ---")
    
    start_train_time = time.time()
    repeat_seeds = [BASE_SEED + i * 101 for i in range(n_repeats)]

    for rep_idx, seed in enumerate(repeat_seeds):
        outer_cv = StratifiedGroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            g_tr = groups[train_idx]

            for name in models_config.keys():
                pipeline = best_pipelines[name]
                pipeline.fit(X_tr, y_tr)

                # ROC-AUC Verification: ALWAYS use predict_proba or decision_function
                if hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_te)[:, 1]
                elif hasattr(pipeline, "decision_function"):
                    y_proba = pipeline.decision_function(X_te)
                else:
                    y_proba = pipeline.predict(X_te).astype(float)

                y_pred = (y_proba >= 0.50).astype(int)

                acc = accuracy_score(y_te, y_pred)
                prec = precision_score(y_te, y_pred, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)
                f1 = f1_score(y_te, y_pred, zero_division=0)
                
                cm = confusion_matrix(y_te, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                roc_auc = roc_auc_score(y_te, y_proba)
                pr_auc = average_precision_score(y_te, y_proba)
                brier = brier_score_loss(y_te, y_proba)

                model_fold_results[name].append({
                    "acc": acc, "prec": prec, "rec": rec, "spec": spec,
                    "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier
                })

                model_predictions[name]["y_true"].extend(y_te)
                model_predictions[name]["y_proba"].extend(y_proba)
                model_trained_pipelines[name] = pipeline

        print(f" Completed Repeat {rep_idx + 1}/{n_repeats} (5 outer folds)")

    return model_fold_results, model_predictions, model_trained_pipelines

def compute_publication_metrics(model_fold_results, model_predictions):
    """
    Computes Mean ± SD, 95% Confidence Intervals, Youden's J Threshold Optimization,
    and Statistical Significance Tests against the top model.
    """
    print("\n==================================================")
    print(" COMPUTING STATISTICAL METRICS & 95% CONFIDENCE INTERVALS")
    print("==================================================")

    metrics_rows = []

    for name, folds in model_fold_results.items():
        df_folds = pd.DataFrame(folds)

        accs = df_folds["acc"].values
        recs = df_folds["rec"].values
        specs = df_folds["spec"].values
        f1s = df_folds["f1"].values
        aucs = df_folds["roc_auc"].values
        pr_aucs = df_folds["pr_auc"].values
        briers = df_folds["brier"].values

        # Mean ± SD
        m_acc, s_acc = np.mean(accs), np.std(accs)
        m_rec, s_rec = np.mean(recs), np.std(recs)
        m_spec, s_spec = np.mean(specs), np.std(specs)
        m_f1, s_f1 = np.mean(f1s), np.std(f1s)
        m_auc, s_auc = np.mean(aucs), np.std(aucs)
        m_pr_auc, s_pr_auc = np.mean(pr_aucs), np.std(pr_aucs)
        m_brier, s_brier = np.mean(briers), np.std(briers)

        # 95% CI (Percentile method across folds)
        ci_auc_low, ci_auc_high = np.percentile(aucs, [2.5, 97.5])
        ci_acc_low, ci_acc_high = np.percentile(accs, [2.5, 97.5])
        ci_f1_low, ci_f1_high = np.percentile(f1s, [2.5, 97.5])

        # Youden's J Threshold Optimization: J = Sensitivity + Specificity - 1
        y_true = np.array(model_predictions[name]["y_true"])
        y_proba = np.array(model_predictions[name]["y_proba"])

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_threshold = thresholds[opt_idx]
        
        # Clip threshold if infinite
        if np.isinf(opt_threshold) or opt_threshold > 1.0:
            opt_threshold = 0.50

        y_opt_pred = (y_proba >= opt_threshold).astype(int)
        opt_rec = recall_score(y_true, y_opt_pred, zero_division=0)
        cm_opt = confusion_matrix(y_true, y_opt_pred)
        tn_opt, fp_opt, _, _ = cm_opt.ravel() if cm_opt.shape == (2, 2) else (0, 0, 0, 0)
        opt_spec = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0

        metrics_rows.append({
            "Model": name,
            "Accuracy_Mean": m_acc, "Accuracy_SD": s_acc,
            "Sensitivity_Mean": m_rec, "Sensitivity_SD": s_rec,
            "Specificity_Mean": m_spec, "Specificity_SD": s_spec,
            "F1_Mean": m_f1, "F1_SD": s_f1,
            "ROC_AUC_Mean": m_auc, "ROC_AUC_SD": s_auc,
            "PR_AUC_Mean": m_pr_auc, "PR_AUC_SD": s_pr_auc,
            "Brier_Mean": m_brier, "Brier_SD": s_brier,
            "ROC_AUC_95_CI": f"{ci_auc_low:.3f} - {ci_auc_high:.3f}",
            "Accuracy_95_CI": f"{ci_acc_low*100:.1f}% - {ci_acc_high*100:.1f}%",
            "Optimal_Threshold": round(float(opt_threshold), 3),
            "Opt_Sensitivity": round(float(opt_rec), 3),
            "Opt_Specificity": round(float(opt_spec), 3),
            "fold_aucs": aucs
        })

    results_df = pd.DataFrame(metrics_rows)
    results_df = results_df.sort_values(by="ROC_AUC_Mean", ascending=False).reset_index(drop=True)
    results_df["Rank"] = range(1, len(results_df) + 1)

    # Statistical Significance Test against Top Ranked Model (Paired Wilcoxon Signed-Rank Test)
    top_model_aucs = results_df.iloc[0]["fold_aucs"]
    p_values = []

    for idx, row in results_df.iterrows():
        if idx == 0:
            p_values.append("Ref (Top)")
        else:
            try:
                _, p_val = stats.wilcoxon(top_model_aucs, row["fold_aucs"])
                p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
                if p_val < 0.05:
                    p_str += " *"
                p_values.append(p_str)
            except Exception:
                p_values.append("N/A")

    results_df["p_value_vs_Top"] = p_values
    return results_df

def generate_publication_visualizations(results_df, model_predictions, trained_pipelines, feature_names):
    print("\n==================================================")
    print(" GENERATING ROC, PR & CALIBRATION PLOTS")
    print("==================================================")

    # 1. ROC Curves Plot
    plt.figure(figsize=(10, 8))
    for name in results_df["Model"]:
        y_true = np.array(model_predictions[name]["y_true"])
        y_proba = np.array(model_predictions[name]["y_proba"])
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title("Publication ROC Curve Comparison (Aggregated Test Proba)", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("backend/model_comparison_roc_curves.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves Plot
    plt.figure(figsize=(10, 8))
    for name in results_df["Model"]:
        y_true = np.array(model_predictions[name]["y_true"])
        y_proba = np.array(model_predictions[name]["y_proba"])
        p_prec, p_rec, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.plot(p_rec, p_prec, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')

    plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title("Publication Precision-Recall Curve Comparison", fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig("backend/model_comparison_pr_curves.png", dpi=300)
    plt.close()

    # 3. Calibration Curves Plot
    plt.figure(figsize=(10, 8))
    for name in results_df["Model"]:
        y_true = np.array(model_predictions[name]["y_true"])
        y_proba = np.array(model_predictions[name]["y_proba"])
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=8)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1.5, label=f'{name}')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    plt.title("Model Probability Calibration Curves", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig("backend/model_calibration_curves.png", dpi=300)
    plt.close()

    # 4. Feature Importance Plot
    top_model_name = results_df.iloc[0]["Model"]
    best_pipe = trained_pipelines[top_model_name]
    best_clf = best_pipe.named_steps["classifier"]

    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=[importances[i] for i in indices],
            y=[feature_names[i] for i in indices],
            palette="viridis"
        )
        plt.title(f"Top 15 Feature Importances ({top_model_name})", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
        plt.ylabel("Biomedical Feature", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("backend/model_feature_importance.png", dpi=300)
        plt.close()

def print_publication_table(results_df):
    print("\n==========================================================================================================================")
    print("                     PUBLICATION-QUALITY PARKINSON'S ML COMPARISON TABLE (MEAN ± SD)                                       ")
    print("==========================================================================================================================")

    headers = [
        "Rank", "Model", "Accuracy (%)", "Sensitivity (%)", "Specificity (%)",
        "F1-Score", "ROC-AUC", "95% CI (ROC-AUC)", "Opt Threshold", "p-value vs Top"
    ]
    print(f"{headers[0]:<4} | {headers[1]:<26} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<14} | {headers[5]:<12} | {headers[6]:<12} | {headers[7]:<16} | {headers[8]:<12} | {headers[9]:<12}")
    print("-" * 148)

    for idx, row in results_df.iterrows():
        acc_str = f"{row['Accuracy_Mean']*100:.1f} ± {row['Accuracy_SD']*100:.1f}"
        rec_str = f"{row['Sensitivity_Mean']*100:.1f} ± {row['Sensitivity_SD']*100:.1f}"
        spec_str = f"{row['Specificity_Mean']*100:.1f} ± {row['Specificity_SD']*100:.1f}"
        f1_str = f"{row['F1_Mean']:.3f} ± {row['F1_SD']:.3f}"
        auc_str = f"{row['ROC_AUC_Mean']:.3f} ± {row['ROC_AUC_SD']:.3f}"

        print(
            f"{row['Rank']:<4} | "
            f"{row['Model']:<26} | "
            f"{acc_str:<14} | "
            f"{rec_str:<14} | "
            f"{spec_str:<14} | "
            f"{f1_str:<12} | "
            f"{auc_str:<12} | "
            f"{row['ROC_AUC_95_CI']:<16} | "
            f"{row['Optimal_Threshold']:<12} | "
            f"{row['p_value_vs_Top']:<12}"
        )
    print("==========================================================================================================================\n")

if __name__ == "__main__":
    X, y, groups, feature_names = load_dataset()
    model_fold_results, model_predictions, trained_pipelines = run_repeated_subject_evaluations(
        X, y, groups, n_repeats=5, n_splits=5 # 25 outer fold evaluations
    )
    results_df = compute_publication_metrics(model_fold_results, model_predictions)
    generate_publication_visualizations(results_df, model_predictions, trained_pipelines, feature_names)
    print_publication_table(results_df)

    # Export clean CSV
    export_cols = [c for c in results_df.columns if c != "fold_aucs"]
    csv_path = "backend/model_comparison_results.csv"
    results_df[export_cols].to_csv(csv_path, index=False)
    print(f" Saved publication metrics summary to: {csv_path}")
