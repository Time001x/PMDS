import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import predict, PredictPayload, load_models, predict_modality, MODELS

def run_evaluation(csv_file_path: str = "backend/parkinsons.data"):
    if not os.path.exists(csv_file_path):
        print(f"[Error] File not found: {csv_file_path}")
        return

    print("\n==================================================")
    print("        PMDS AI DATASET EVALUATION SUITE          ")
    print("==================================================")
    print(f"Loading dataset from: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {len(df)} sample records.\n")

    load_models()

    is_uci_voice = 'MDVP:Jitter(%)' in df.columns and 'status' in df.columns

    y_true = []
    y_pred = []

    if is_uci_voice:
        print("[Dataset] Detected: UCI Parkinson's Voice Speech Dataset (195 recordings)")
        print("[Modality] Target: Voice Model (CatBoost / RandomForest Ensemble)\n")

        m = MODELS['voice']
        trans = m['transformer']
        sel = m['selector']
        model = m['model']
        cols = getattr(trans, 'feature_names_in_', None)

        rows = []
        for idx, row in df.iterrows():
            y_true.append(int(row['status']))
            d = row.to_dict()
            row_dict = {col: d.get(col, 0.0) for col in cols}
            rows.append(row_dict)

        X = pd.DataFrame(rows)
        transformed = trans.transform(X)
        selected = sel.transform(transformed) if sel else transformed
        probs = model.predict_proba(selected)[:, 1]
        
        # Decision threshold calibrated for UCI Voice model
        threshold = 0.05
        y_pred = (probs >= threshold).astype(int).tolist()

    else:
        print("[Dataset] Detected: General PMDS Multi-Modal Test Dataset\n")
        target_col = None
        for col in ['actual_label', 'target', 'is_parkinson', 'label', 'status', 'Group']:
            if col in df.columns:
                target_col = col
                break

        for idx, row in df.iterrows():
            if target_col:
                raw_target = row[target_col]
                true_label = 1 if str(raw_target).strip().lower() in ['1', 'parkinson', 'pd', 'true'] else 0
                y_true.append(true_label)

            payload = PredictPayload(
                uid=str(row.get('uid', f"sample_{idx+1}")),
                speechScore=float(row.get('speechScore', row.get('SpeechProblems', 0.0))),
                tremorScore=float(row.get('tremorScore', row.get('Tremor', 0.0))),
                fingerScore=float(row.get('fingerScore', row.get('Bradykinesia', 0.0))),
                gaitScore=float(row.get('gaitScore', row.get('PosturalInstability', 0.0))),
                questionnaireScore=float(row.get('questionnaireScore', row.get('UPDRS', 0.0))),
                Age=float(row.get('Age', 60.0))
            )

            res = predict(payload)
            y_pred.append(1 if res.riskScore >= 0.40 else 0)

    if len(y_true) > 0:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

        acc = (tp + tn) / len(y_true_arr) * 100.0
        sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

        print("==================================================")
        print("           EVALUATION METRICS REPORT              ")
        print("==================================================")
        print(f" Total Samples Tested   : {len(y_true_arr)}")
        print(f" Parkinson Positive (1) : {int(np.sum(y_true_arr == 1))} samples")
        print(f" Healthy Controls (0)   : {int(np.sum(y_true_arr == 0))} samples")
        print("--------------------------------------------------")
        print(f" Accuracy               : {acc:.2f}%")
        print(f" Sensitivity (Recall)   : {sensitivity:.2f}% (True Positive Rate)")
        print(f" Specificity            : {specificity:.2f}% (True Negative Rate)")
        print(f" Precision              : {precision:.2f}%")
        print(f" F1-Score               : {f1:.2f}%")
        print("--------------------------------------------------")
        print(" CONFUSION MATRIX:")
        print(f"                      Predicted Normal   Predicted Parkinson")
        print(f" Actual Normal (0)        {tn:^14d}   {fp:^18d}")
        print(f" Actual Parkinson (1)     {fn:^14d}   {tp:^18d}")
        print("==================================================\n")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "backend/parkinsons.data"
    run_evaluation(filepath)
