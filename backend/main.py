import os
import sys
import types
from typing import Dict, Any, Optional

# Ensure backend directory is in sys.path for Render cloud deployment
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

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

from database import DatabaseService, SessionLocal
from security import SecurityService

# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterPayload(BaseModel):
    username: str
    password: str
    fullName: Optional[str] = ""
    email: Optional[str] = ""

class LoginPayload(BaseModel):
    username: str
    password: str

class AuthResult(BaseModel):
    status: str
    message: str
    token: Optional[str] = None
    userId: Optional[str] = None
    username: Optional[str] = None
    fullName: Optional[str] = None

class PredictPayload(BaseModel):
    uid: Optional[str] = "demo_user"
    SpeechProblems: Optional[float] = 0.0
    Tremor: Optional[float] = 0.0
    PosturalInstability: Optional[float] = 0.0
    Bradykinesia: Optional[float] = 0.0
    UPDRS: Optional[float] = 0.0
    Rigidity: Optional[float] = 0.0
    MoCA: Optional[float] = 26.0
    FunctionalAssessment: Optional[float] = 100.0
    Age: Optional[float] = 60.0
    speechScore: Optional[float] = 0.0
    tremorScore: Optional[float] = 0.0
    fingerScore: Optional[float] = 0.0
    gaitScore: Optional[float] = 0.0
    questionnaireScore: Optional[float] = 0.0

class PredictResult(BaseModel):
    uid: Optional[str] = "demo_user"
    level: int = 0  # 0, 1, 2, 3, 4 MDS-UPDRS Scale
    riskScore: Optional[float] = 0.0
    riskPercent: Optional[int] = 0
    diagnosis: str = "ไม่มีอาการ"
    label: str = "ไม่มีอาการ"
    color: str = "#05C134"
    description: Optional[str] = ""
    confidence: float = 0.95
    details: Optional[Dict[str, float]] = None

MDS_UPDRS_SCALE = {
    0: {"english": "Normal", "label": "ไม่มีอาการ", "color": "#05C134", "description": "ปกติ สมบูรณ์ ไม่มีอาการแสดงของโรคพาร์กินสัน"},
    1: {"english": "Slight", "label": "เล็กน้อย", "color": "#10B981", "description": "มีความผิดปกติเพียงเล็กน้อย ไม่กระทบต่อการดำเนินชีวิต"},
    2: {"english": "Mild", "label": "เสี่ยงปานกลาง", "color": "#F59E0B", "description": "มีอาการชัดเจนขึ้น เริ่มส่งผลกระทบต่อบางกิจกรรม"},
    3: {"english": "Moderate", "label": "เสี่ยงมาก", "color": "#F97316", "description": "มีอาการชัดเจน รบกวนการทำกิจกรรมประจำวันอย่างมาก"},
    4: {"english": "Severe", "label": "อาการรุนแรง", "color": "#C10508", "description": "มีอาการรุนแรงมาก ไม่สามารถพึ่งพาตนเองได้"}
}

def get_risk_level_info(level: int):
    level_int = max(0, min(4, int(round(level))))
    info = MDS_UPDRS_SCALE[level_int]
    return info["label"], info["label"], info["color"], info["description"], level_int

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
        row_data[col] = feature_dict.get(col, 0.0)

    try:
        df_in = pd.DataFrame([row_data])
        X_trans = transformer.transform(df_in)
        X_sel = selector.transform(X_trans)
        probs = model.predict_proba(X_sel)
        return float(probs[0][1])
    except Exception as e:
        print(f"[PMDS AI Backend] Error predicting '{modality}': {e}")
        return 0.0

# ── Endpoints ─────────────────────────────────────────────────────────────────
# ── Authentication Endpoints (Encrypted Data & Bcrypt Passwords) ─────────────
@app.post("/auth/register", response_model=AuthResult)
def register_user(payload: RegisterPayload):
    db = SessionLocal()
    try:
        res = DatabaseService.register_user(
            db=db,
            username=payload.username,
            password_raw=payload.password,
            full_name=payload.fullName or "",
            email=payload.email or ""
        )
        token = SecurityService.create_jwt_token({"sub": res["user_id"], "username": res["username"]})
        return AuthResult(
            status="ok",
            message="User registered successfully with AES-256 encryption",
            token=token,
            userId=res["user_id"],
            username=res["username"],
            fullName=res["full_name"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    finally:
        db.close()

@app.post("/auth/login", response_model=AuthResult)
def login_user(payload: LoginPayload):
    db = SessionLocal()
    try:
        user = DatabaseService.authenticate_user(db, payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        token = SecurityService.create_jwt_token({"sub": user["user_id"], "username": user["username"]})
        return AuthResult(
            status="ok",
            message="Authentication successful",
            token=token,
            userId=user["user_id"],
            username=user["username"],
            fullName=user["full_name"]
        )
    finally:
        db.close()

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

    # Extract safe continuous sensor scores and age from payload
    user_age = float(payload.Age) if payload.Age is not None else 60.0
    user_age = max(1.0, user_age)

    speech_risk = float(payload.speechScore) if payload.speechScore is not None else float(payload.SpeechProblems or 0.0)
    tremor_risk = float(payload.tremorScore) if payload.tremorScore is not None else float(payload.Tremor or 0.0)
    finger_risk = float(payload.fingerScore) if payload.fingerScore is not None else float(payload.Bradykinesia or 0.0)
    gait_risk = float(payload.gaitScore) if payload.gaitScore is not None else float(payload.PosturalInstability or 0.0)
    questionnaire_risk = float(payload.questionnaireScore) if payload.questionnaireScore is not None else float((payload.UPDRS or 0.0) / 100.0)

    # 1. Questionnaire prediction
    q_dict = {
        'Age': user_age,
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
        'updrs_age_ratio': (questionnaire_risk * 50.0) / user_age
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

    # ── Majority Decision Rule (3 out of 5 tests rule) ───────────────────
    modal_risks = [speech_risk, tremor_risk, finger_risk, gait_risk, questionnaire_risk]
    normal_count = sum(1 for r in modal_risks if r < 0.35)
    high_risk_count = sum(1 for r in modal_risks if r >= 0.50)

    if normal_count >= 3:
        updrs_level = 0
    elif high_risk_count >= 3:
        mean_risk = float(np.mean(modal_risks))
        updrs_level = 4 if mean_risk >= 0.80 else 3
    else:
        weighted_risk = sum(modality_scores[m] for m in MODALITIES if m in modality_scores) / len(MODALITIES)
        if weighted_risk >= 0.80:
            updrs_level = 4
        elif weighted_risk >= 0.60:
            updrs_level = 3
        elif weighted_risk >= 0.40:
            updrs_level = 2
        elif weighted_risk >= 0.20:
            updrs_level = 1
        else:
            updrs_level = 0

    diagnosis, label, color, description, level_int = get_risk_level_info(updrs_level)

    scores_array = np.array(list(modality_scores.values()))
    std_dev = np.std(scores_array) if len(scores_array) > 0 else 0
    confidence = float(np.clip(1.0 - (std_dev * 0.4), 0.78, 0.98))

    # Save encrypted assessment results to secure database
    try:
        db = SessionLocal()
        sensor_json_raw = str(payload.model_dump())
        DatabaseService.save_test_result(
            db=db,
            user_id=payload.uid or "demo_user",
            sensor_data_raw=sensor_json_raw,
            risk_score=float(level_int),
            risk_percent=level_int * 25.0,
            diagnosis=diagnosis
        )
        db.close()
    except Exception as db_err:
        print(f"[PMDS DB Warning] Could not store encrypted test result: {db_err}")

    return PredictResult(
        uid=payload.uid or "demo_user",
        level=level_int,
        riskScore=float(level_int),
        riskPercent=int(level_int * 25),
        diagnosis=diagnosis,
        label=label,
        color=color,
        description=description,
        confidence=round(confidence, 2),
        details=modality_scores
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app" if os.path.exists("backend") else "main:app", host="0.0.0.0", port=port)
