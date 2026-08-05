import os
import sys
import types
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Mock dynamic preprocessing module compatibility
p = types.ModuleType('preprocessing')
fs = types.ModuleType('preprocessing.feature_selection')
class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X
fs.FeatureSelector = FeatureSelector
sys.modules['preprocessing'] = p
sys.modules['preprocessing.feature_selection'] = fs

def inspect_model(model_path):
    print("============================================================")
    print(f" INSPECTING MODEL FILE: {model_path}")
    print("============================================================")

    if not os.path.exists(model_path):
        print(f"[-] Error: File not found at {model_path}")
        return

    data = joblib.load(model_path)
    
    if isinstance(data, dict):
        print(f"[*] Keys in Dictionary  : {list(data.keys())}")
        if 'modality' in data:
            print(f"[*] Modality            : {data['modality']}")
        if 'model_name' in data:
            print(f"[*] Model Name         : {data['model_name']}")
        if 'model' in data:
            model_obj = data['model']
            print(f"[*] Model Architecture : {type(model_obj).__name__}")
            print(f"[*] Model Parameters   :")
            for k, v in model_obj.get_params().items():
                print(f"    - {k}: {v}")
        if 'transformer' in data and hasattr(data['transformer'], 'feature_names_in_'):
            print(f"[*] Feature Inputs ({len(data['transformer'].feature_names_in_)}):")
            print(f"    {list(data['transformer'].feature_names_in_)}")
    else:
        print(f"[*] Loaded Object Type : {type(data).__name__}")
        if hasattr(data, 'get_params'):
            print(f"[*] Model Parameters   : {data.get_params()}")
    print("============================================================\n")

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "exports/voice/voice_model.joblib"
    inspect_model(target_path)
