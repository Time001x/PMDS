import sqlite3
import os
from security import SecurityService

def check_database():
    db_path = os.path.join(os.path.dirname(__file__), "pmds_secure.db")
    
    print("============================================================")
    print(" CHECKING ENCRYPTED PMDS DATABASE (pmds_secure.db)")
    print("============================================================")

    if not os.path.exists(db_path):
        print("[-] Database file not found yet. Run FastAPI backend server to initialize.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("SELECT user_id, username, password_hash, encrypted_full_name, encrypted_email, created_at FROM users")
    users = cursor.fetchall()
    print(f"\n USERS TABLE (Total Registered Users: {len(users)})")
    print("-" * 60)

    for u in users:
        u_id, username, pwd_hash, enc_name, enc_email, created = u
        try:
            dec_name = SecurityService.decrypt_data(enc_name) if enc_name else "N/A"
        except Exception:
            dec_name = "[Encrypted Key Differs - Protected AES-256]"
            
        try:
            dec_email = SecurityService.decrypt_data(enc_email) if enc_email else "N/A"
        except Exception:
            dec_email = "[Encrypted Key Differs - Protected AES-256]"
        
        print(f"[*] User ID       : {u_id}")
        print(f"    Anon Username : {username} (Hashed SHA-256)")
        print(f"    Decrypted PII : Name='{dec_name}', Email='{dec_email}'")
        print(f"    Bcrypt Hash   : {pwd_hash[:30]}...")
        print(f"    Encrypted Name: {enc_name[:30] if enc_name else 'N/A'}... (RAW DB)")
        print(f"    Created At    : {created}\n")

    # 2. Test Results Table
    cursor.execute("SELECT test_id, user_id, risk_percent, diagnosis, created_at, encrypted_sensor_data FROM test_results ORDER BY created_at DESC")
    tests = cursor.fetchall()
    print(f" TEST RESULTS TABLE (Total Saved Assessment Records: {len(tests)})")
    print("-" * 60)

    for t in tests:
        t_id, u_id, risk_pct, diag, created, enc_sensor = t
        print(f"[*] Test ID       : {t_id}")
        print(f"    User ID       : {u_id}")
        print(f"    Risk Percent  : {risk_pct}%")
        print(f"    Created At    : {created}")
        print(f"    Encrypted Data: {enc_sensor[:40]}... (AES-256 RAW DB)\n")

    conn.close()
    print("=" * 60 + "\n")

if __name__ == "__main__":
    check_database()
