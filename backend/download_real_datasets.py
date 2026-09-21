"""
PMDS Real Clinical Dataset Downloader & Loader
==============================================
Downloads public real-world clinical datasets for Parkinson's Disease:
1. UCI Oxford Parkinson's Voice Dataset (parkinsons.data)
2. UCI Parkinson's Telemonitoring Dataset (parkinsons_updrs.data) - 5,875 real clinical recordings from 42 patients
3. PhysioNet Gait in Parkinson's Disease Dataset (demographics + gait recordings)
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np

# Force UTF-8 encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

def download_file(url, target_path):
    print(f"Downloading: {url} -> {target_path}")
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[SUCCESS] Downloaded {target_path} ({os.path.getsize(target_path)} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

def setup_real_datasets():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(backend_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    print("==========================================================")
    print(" PMDS REAL CLINICAL DATASET DOWNLOADER")
    print("==========================================================")

    # 1. Download UCI Oxford Voice Dataset (parkinsons.data)
    url_oxford = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    path_oxford = os.path.join(datasets_dir, "parkinsons_oxford_voice.csv")
    if not os.path.exists(path_oxford):
        download_file(url_oxford, path_oxford)
    else:
        print(f"[INFO] {path_oxford} already exists.")

    # 2. Download UCI Telemonitoring Dataset (parkinsons_updrs.data)
    url_tele = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
    path_tele = os.path.join(datasets_dir, "parkinsons_telemonitoring_updrs.csv")
    if not os.path.exists(path_tele):
        download_file(url_tele, path_tele)
    else:
        print(f"[INFO] {path_tele} already exists.")

    # 3. Download PhysioNet Gait Demographics
    url_gait_demo = "https://physionet.org/files/gait-parkinsons/1.0.0/demographics.txt"
    path_gait_demo = os.path.join(datasets_dir, "gait_demographics_physionet.txt")
    if not os.path.exists(path_gait_demo):
        download_file(url_gait_demo, path_gait_demo)
    else:
        print(f"[INFO] {path_gait_demo} already exists.")

    print("\n----------------------------------------------------------")
    print(" REAL DATASET AUDIT & PREPARATION")
    print("----------------------------------------------------------")

    # Load and process Telemonitoring dataset
    if os.path.exists(path_tele):
        df_tele = pd.read_csv(path_tele)
        print(f"UCI Telemonitoring Dataset : {len(df_tele)} recordings across {df_tele['subject#'].nunique()} patients")
        print(f"  Features                 : {df_tele.shape[1]} columns (Jitter, Shimmer, NHR, HNR, motor_UPDRS, total_UPDRS, age, sex)")

        # Create binary label based on clinical UPDRS threshold (total_UPDRS >= 25 is moderate PD)
        df_tele['is_parkinson'] = (df_tele['total_UPDRS'] >= 25).astype(int)
        df_tele['subject_id'] = df_tele['subject#'].apply(lambda x: f"patient_{x}")
        
        processed_tele_path = os.path.join(datasets_dir, "real_clinical_telemonitoring_processed.csv")
        df_tele.to_csv(processed_tele_path, index=False)
        print(f"[SUCCESS] Created processed dataset: {processed_tele_path}")

    return datasets_dir

if __name__ == "__main__":
    setup_real_datasets()
