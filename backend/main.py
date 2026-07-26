import os
import sys
import types
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

# ── Dynamic Module Setup for Model Compatibility ──────────────────────────────
p = types.ModuleType('preprocessing')
fs = types.ModuleType('preprocessing.feature_selection')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(self, 'selected_indices_') and self.selected_indices_ is not None:
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_indices_]
            elif hasattr(X, 'shape'):
                return X[:, self.selected_indices_]
        return X

fs.FeatureSelector = FeatureSelector
sys.modules['preprocessing'] = p
sys.modules['preprocessing.feature_selection'] = fs

# ── FastAPI App Initialization ────────────────────────────────────────────────
app = FastAPI(title="PMDS Parkinson Assessment AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{full_path:path}")
def options_handler(full_path: str):
    return {"status": "ok"}

MODELS: Dict[str, Dict[str, Any]] = {}
MODALITIES = ["finger", "gait", "questionnaire", "tremor", "voice"]

@app.on_event("startup")
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for modality in MODALITIES:
        # Check exports folder first, then assets/models
        model_path = os.path.join(base_dir, "exports", modality, f"{modality}_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "src", "assets", "models", modality, f"{modality}_model.joblib")
        
        if os.path.exists(model_path):
            try:
                loaded = joblib.load(model_path)
                MODELS[modality] = loaded
                print(f"[PMDS AI Backend] Loaded model for '{modality}' successfully from {model_path}")
            except Exception as e:
                print(f"[PMDS AI Backend] Error loading model for '{modality}': {e}")
        else:
            print(f"[PMDS AI Backend] Model file not found for '{modality}': {model_path}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictPayload(BaseModel):
    uid: str
    SpeechProblems: float = 0
    Tremor: float = 0
    PosturalInstability: float = 0
    Bradykinesia: float = 0
    UPDRS: float = 0
    Rigidity: float = 0
    MoCA: float = 26
    FunctionalAssessment: float = 100
    Age: float = 60
    speechScore: Optional[float] = None
    tremorScore: Optional[float] = None
    fingerScore: Optional[float] = None
    gaitScore: Optional[float] = None
    questionnaireScore: Optional[float] = None

class PredictResult(BaseModel):
    uid: str
    riskScore: float
    riskPercent: int
    diagnosis: str
    label: str
    color: str
    confidence: float
    details: Optional[Dict[str, float]] = None

def get_risk_level_info(val: float):
    norm = val / 100.0 if val > 1.0 else val
    if norm >= 0.80:
        return "เสี่ยงรุนแรง", "เสี่ยงรุนแรง", "#C10508"
    elif norm >= 0.60:
        return "ค่อนข้างเสี่ยง", "ค่อนข้างเสี่ยง", "#C10508"
    elif norm >= 0.40:
        return "เสี่ยงปานกลาง", "เสี่ยงปานกลาง", "#F59E0B"
    elif norm >= 0.20:
        return "เสี่ยงเล็กน้อย", "เสี่ยงเล็กน้อย", "#F59E0B"
    else:
        return "ปกติ (ไม่พบอาการ)", "ปกติ (ไม่พบอาการ)", "#05C134"

def predict_modality(modality: str, feature_dict: Dict[str, float]) -> float:
    if modality not in MODELS:
        return 0.0
    
    m_info = MODELS[modality]
    transformer = m_info.get("transformer")
    selector = m_info.get("selector")
    model = m_info["model"]

    cols = getattr(transformer, "feature_names_in_", None)
    if cols is None:
        return 0.0

    row_data = {}
    for col in cols:
        row_data[col] = feature_dict.get(col, 0.5)

    df = pd.DataFrame([row_data])
    transformed = transformer.transform(df)
    selected = selector.transform(transformed) if selector else transformed

    proba = model.predict_proba(selected)
    if proba.shape[1] > 1:
        if proba.shape[1] > 2:
            weights = np.linspace(0, 1, proba.shape[1])
            return float(np.sum(proba[0] * weights))
        return float(proba[0][1])
    return float(proba[0][0])

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "PMDS Parkinson AI Backend is running", "loaded_models": list(MODELS.keys())}

@app.get("/health")
def health_check():
    ready = len(MODELS) > 0
    return {"status": "ok" if ready else "warning", "model_ready": ready, "models_count": len(MODELS)}

@app.post("/predict", response_model=PredictResult)
def predict(payload: PredictPayload):
    modality_scores = {}

    # Extract continuous sensor scores from payload
    speech_risk = payload.speechScore if payload.speechScore is not None else float(payload.SpeechProblems)
    tremor_risk = payload.tremorScore if payload.tremorScore is not None else float(payload.Tremor)
    finger_risk = payload.fingerScore if payload.fingerScore is not None else float(payload.Bradykinesia)
    gait_risk = payload.gaitScore if payload.gaitScore is not None else float(payload.PosturalInstability)
    questionnaire_risk = payload.questionnaireScore if payload.questionnaireScore is not None else float(payload.UPDRS / 100.0)

    # 1. Questionnaire prediction
    q_dict = {
        'Age': payload.Age,
        'Gender': 1,
        'UPDRS_Score': questionnaire_risk * 50.0,
        'Tremor_Symptom': 1 if tremor_risk > 0.4 else 0,
        'Bradykinesia_Symptom': 1 if finger_risk > 0.4 else 0,
        'Rigidity_Symptom': 1 if tremor_risk > 0.4 else 0,
        'Postural_Instability': 1 if gait_risk > 0.4 else 0,
        'Constipation_History': 0,
        'RBD_Sleep_Disorder': 0,
        'Family_History_PD': 0,
        'Smell_Loss_Anosmia': 0,
        'motor_symptom_count': (tremor_risk + finger_risk + gait_risk + speech_risk) * 2.0,
        'updrs_age_ratio': (questionnaire_risk * 50.0) / max(payload.Age, 1)
    }
    modality_scores['questionnaire'] = predict_modality('questionnaire', q_dict)

    # 2. Tremor prediction
    t_dict = {
        'acc_rms': 0.05 + tremor_risk * 3.5,
        'gyro_x_rms': 0.02 + tremor_risk * 2.5,
        'is_pathological_frequency': 1 if tremor_risk > 0.35 else 0
    }
    modality_scores['tremor'] = predict_modality('tremor', t_dict)

    # 3. Finger tap prediction
    f_dict = {
        'tap_speed_hz': max(1.0, 5.5 - finger_risk * 3.5),
        'variability_index': 0.05 + finger_risk * 0.6,
        'miss_rate_percent': finger_risk * 25.0
    }
    modality_scores['finger'] = predict_modality('finger', f_dict)

    # 4. Gait prediction
    g_dict = {
        'walking_speed_m_per_s': max(0.4, 1.4 - gait_risk * 0.8),
        'gait_symmetry_index': max(0.4, 1.0 - gait_risk * 0.45),
        'step_variability': 0.05 + gait_risk * 0.55
    }
    modality_scores['gait'] = predict_modality('gait', g_dict)

    # 5. Voice prediction
    v_dict = {
        'MDVP:Jitter(%)': 0.003 + speech_risk * 0.025,
        'MDVP:Shimmer': 0.015 + speech_risk * 0.065,
        'NHR': 0.005 + speech_risk * 0.05,
        'HNR': max(8.0, 24.0 - speech_risk * 14.0)
    }
    modality_scores['voice'] = predict_modality('voice', v_dict)

    # Multi-Modal Max Severity & Weighted Ensemble
    max_modal_risk = max([speech_risk, tremor_risk, finger_risk, gait_risk, questionnaire_risk])
    weights = {'questionnaire': 0.25, 'tremor': 0.20, 'finger': 0.20, 'gait': 0.20, 'voice': 0.15}
    weighted_risk = sum(modality_scores[m] * weights[m] for m in MODALITIES if m in modality_scores)
    
    total_risk = max(max_modal_risk * 0.85, weighted_risk)
    risk_percent = int(round(total_risk * 100))
    diagnosis, label, color = get_risk_level_info(risk_percent)

    scores_array = np.array(list(modality_scores.values()))
    std_dev = np.std(scores_array) if len(scores_array) > 0 else 0
    confidence = float(np.clip(1.0 - (std_dev * 0.4), 0.78, 0.98))

    return PredictResult(
        uid=payload.uid,
        riskScore=round(total_risk, 4),
        riskPercent=risk_percent,
        diagnosis=diagnosis,
        label=label,
        color=color,
        confidence=round(confidence, 2),
        details=modality_scores
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
