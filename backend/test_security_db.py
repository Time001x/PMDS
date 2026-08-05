import urllib.request
import json
import sqlite3
import os

BASE_URL = "http://localhost:8000"

def test_api():
    print("==================================================")
    print(" TESTING PMDS ENCRYPTED DATABASE & AUTH APIS")
    print("==================================================")

    # 1. Health check
    req = urllib.request.urlopen(f"{BASE_URL}/health")
    health = json.loads(req.read().decode())
    print(f"[+] Health Check Response: {health}")

    # 2. Register New User
    reg_payload = {
        "username": "test_patient_001",
        "password": "SuperSecretPass123!",
        "fullName": "Somchai Jaidee",
        "email": "somchai.j@hospital.or.th"
    }
    data_bytes = json.dumps(reg_payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/auth/register", data=data_bytes, headers={'Content-Type': 'application/json'})
    
    try:
        res = urllib.request.urlopen(req)
        reg_res = json.loads(res.read().decode())
        print(f"[+] User Register Response: {reg_res}")
    except Exception as e:
        print(f"[-] Register error (User may already exist): {e}")

    # 3. User Login
    login_payload = {
        "username": "test_patient_001",
        "password": "SuperSecretPass123!"
    }
    data_bytes = json.dumps(login_payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/auth/login", data=data_bytes, headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    login_res = json.loads(res.read().decode())
    print(f"[+] User Login Response: {login_res}")
    token = login_res.get("token")
    user_id = login_res.get("userId")

    # 4. Predict & Save Encrypted Assessment
    pred_payload = {
        "uid": user_id,
        "speechScore": 0.1,
        "tremorScore": 0.1,
        "fingerScore": 0.1,
        "gaitScore": 0.1,
        "questionnaireScore": 0.1,
        "Age": 55
    }
    data_bytes = json.dumps(pred_payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/predict", data=data_bytes, headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    })
    res = urllib.request.urlopen(req)
    pred_res = json.loads(res.read().decode())
    print(f"[+] Predict Assessment Response: riskPercent={pred_res.get('riskPercent')}%, riskScore={pred_res.get('riskScore')}")

    # 5. Inspect Encrypted SQLite Database
    db_path = "backend/pmds_secure.db"
    print("\n==================================================")
    print(f" INSPECTING RAW ENCRYPTED DATABASE ({db_path})")
    print("==================================================")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, username, password_hash, encrypted_full_name, encrypted_email FROM users WHERE username='test_patient_001'")
    row = cursor.fetchone()
    if row:
        print(f"User ID               : {row[0]}")
        print(f"Username              : {row[1]}")
        print(f"Bcrypt Password Hash  : {row[2][:35]}... (One-Way Salted Hash)")
        print(f"AES-256 Full Name     : {row[3][:35]}... (ENCRYPTED)")
        print(f"AES-256 Email         : {row[4][:35]}... (ENCRYPTED)")

    cursor.execute("SELECT test_id, user_id, risk_percent, diagnosis, encrypted_sensor_data FROM test_results ORDER BY created_at DESC LIMIT 1")
    t_row = cursor.fetchone()
    if t_row:
        print(f"\nLatest Test ID       : {t_row[0]}")
        print(f"User ID               : {t_row[1]}")
        print(f"Diagnosis Risk        : {t_row[2]}%")
        print(f"AES-256 Sensor Data   : {t_row[4][:40]}... (ENCRYPTED)")

    conn.close()
    print("==================================================\n")

if __name__ == "__main__":
    test_api()
