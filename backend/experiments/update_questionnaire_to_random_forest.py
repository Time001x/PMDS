"""
Fix Questionnaire RandomForest Model to Exactly Match 15 Selected Features
==========================================================================
"""

import os
import sys
import types
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

p = types.ModuleType('preprocessing')
fs = types.ModuleType('preprocessing.feature_selection')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if hasattr(self, 'selected_indices_') and self.selected_indices_ is not None:
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_indices_]
            elif hasattr(X, 'shape'):
                return X[:, self.selected_indices_]
        return X

FeatureSelector.__module__ = 'preprocessing.feature_selection'
fs.FeatureSelector = FeatureSelector
sys.modules['preprocessing'] = p
sys.modules['preprocessing.feature_selection'] = fs

def update_questionnaire_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q_model_path = os.path.join(base_dir, "exports", "questionnaire", "questionnaire_model.joblib")
    
    old_obj = joblib.load(q_model_path)
    transformer = old_obj['transformer']
    selector = old_obj['selector']
    
    np.random.seed(42)
    n_samples = 1500
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    data = []
    for label in y:
        if label == 1:
            age = np.random.normal(68, 8)
            gender = np.random.choice([0, 1])
            updrs = np.random.uniform(18, 55)
            tremor_sym = np.random.choice([0, 1], p=[0.25, 0.75])
            brady_sym = np.random.choice([0, 1], p=[0.15, 0.85])
            rigidity_sym = np.random.choice([0, 1], p=[0.20, 0.80])
            postural_sym = np.random.choice([0, 1], p=[0.30, 0.70])
            constipation = np.random.choice([0, 1], p=[0.40, 0.60])
            rbd = np.random.choice([0, 1], p=[0.45, 0.55])
            family_hist = np.random.choice([0, 1], p=[0.75, 0.25])
            anosmia = np.random.choice([0, 1], p=[0.35, 0.65])
        else:
            age = np.random.normal(62, 9)
            gender = np.random.choice([0, 1])
            updrs = np.random.uniform(0, 12)
            tremor_sym = np.random.choice([0, 1], p=[0.90, 0.10])
            brady_sym = np.random.choice([0, 1], p=[0.95, 0.05])
            rigidity_sym = np.random.choice([0, 1], p=[0.92, 0.08])
            postural_sym = np.random.choice([0, 1], p=[0.90, 0.10])
            constipation = np.random.choice([0, 1], p=[0.80, 0.20])
            rbd = np.random.choice([0, 1], p=[0.85, 0.15])
            family_hist = np.random.choice([0, 1], p=[0.92, 0.08])
            anosmia = np.random.choice([0, 1], p=[0.88, 0.12])
            
        motor_count = tremor_sym + brady_sym + rigidity_sym + postural_sym
        updrs_age_ratio = updrs / max(1.0, age)
        
        row = {
            'Age': age,
            'Gender': gender,
            'UPDRS_Score': updrs,
            'Tremor_Symptom': tremor_sym,
            'Bradykinesia_Symptom': brady_sym,
            'Rigidity_Symptom': rigidity_sym,
            'Postural_Instability': postural_sym,
            'Constipation_History': constipation,
            'RBD_Sleep_Disorder': rbd,
            'Family_History_PD': family_hist,
            'Smell_Loss_Anosmia': anosmia,
            'motor_symptom_count': motor_count,
            'updrs_age_ratio': updrs_age_ratio
        }
        data.append(row)
        
    df_train = pd.DataFrame(data)
    X_trans = transformer.transform(df_train)
    X_sel = selector.transform(X_trans)
    
    print(f"X_sel shape for training RF: {X_sel.shape}")
    
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_sel, y)
    
    new_obj = {
        'model': rf_model,
        'transformer': transformer,
        'selector': selector,
        'model_name': 'RandomForestClassifier',
        'modality': 'questionnaire'
    }
    
    joblib.dump(new_obj, q_model_path)
    print(f"[SUCCESS] Questionnaire model updated to RandomForestClassifier ({X_sel.shape[1]} inputs) successfully!")

if __name__ == "__main__":
    update_questionnaire_model()
