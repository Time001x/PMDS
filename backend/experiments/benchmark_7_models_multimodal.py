"""
Benchmarking all 7 ML Models on 5-Modality Multimodal Dataset
============================================================
Compare performance of each of the 7 ML models:
1. When trained on Single Modality (Voice Only)
2. When trained on All 5 Modalities (Multimodal Features)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "datasets", "real_clinically_matched_multimodal.csv")

def evaluate_7_models_on_multimodal():
    df = pd.read_csv(dataset_path)
    
    # 5 Modality Risk features + raw sensor features
    features = [
        'voice_risk', 'tremor_risk', 'finger_risk', 'gait_risk', 'questionnaire_risk',
        'tremor_rms_ms2', 'tremor_freq_hz', 'stride_cv_percent', 'gait_velocity_m_s', 
        'tap_iti_mean_ms', 'tap_iti_cv_percent'
    ]
    
    X = df[features].copy()
    y = df['is_parkinson'].astype(int).values

    models = {
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(random_state=42, max_iter=500),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42, class_weight='balanced'),
        "CatBoost": CatBoostClassifier(random_seed=42, verbose=0),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, clf in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        
        cv_res = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=False
        )
        
        results.append({
            "Model": name,
            "Accuracy_5Mod": f"{np.mean(cv_res['test_accuracy'])*100:.1f}%",
            "Sensitivity_5Mod": f"{np.mean(cv_res['test_recall'])*100:.1f}%",
            "F1_Score_5Mod": f"{np.mean(cv_res['test_f1'])*100:.1f}%",
            "ROC_AUC_5Mod": f"{np.mean(cv_res['test_roc_auc']):.3f}",
            "ROC_AUC_Raw": np.mean(cv_res['test_roc_auc'])
        })

    df_res = pd.DataFrame(results).sort_values(by="ROC_AUC_Raw", ascending=False)
    print(df_res.to_string(index=False))
    return df_res

if __name__ == "__main__":
    evaluate_7_models_on_multimodal()
