import os
import base64
from typing import Optional
import jwt
import datetime
import bcrypt
from cryptography.fernet import Fernet

# Secret keys (Loaded from environment variables, with fixed secure key for dev environment)
SECRET_KEY = os.getenv("PMDS_SECRET_KEY", "pmds-secret-key-medical-ai-2026-super-secure-jwt-key")
ENCRYPTION_KEY = os.getenv("PMDS_ENCRYPTION_KEY", "uO-qX7M0J5f1zQ9xW8vK3mR2tY4pL6sN8bV0c9dE1fA=")

# Ensure Fernet Key is valid 32-byte base64
try:
    cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
except Exception:
    # Deterministic fallback key for development consistency
    fixed_key = b'uO-qX7M0J5f1zQ9xW8vK3mR2tY4pL6sN8bV0c9dE1fA='
    cipher_suite = Fernet(fixed_key)

class SecurityService:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hashes password using bcrypt with automatic salt."""
        pwd_bytes = password.encode('utf-8')[:72] # Bcrypt max 72 bytes limit
        salt = bcrypt.gensalt(12)
        hashed_bytes = bcrypt.hashpw(pwd_bytes, salt)
        return hashed_bytes.decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifies plain password against bcrypt hash."""
        try:
            pwd_bytes = plain_password.encode('utf-8')[:72]
            hash_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(pwd_bytes, hash_bytes)
        except Exception:
            return False

    @staticmethod
    def encrypt_data(data: str) -> str:
        """Encrypts sensitive medical PII data using AES-256 Fernet encryption."""
        if not data:
            return ""
        encrypted_bytes = cipher_suite.encrypt(data.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')

    @staticmethod
    def decrypt_data(encrypted_str: str) -> str:
        """Decrypts AES-256 encrypted data."""
        if not encrypted_str:
            return ""
        try:
            decrypted_bytes = cipher_suite.decrypt(encrypted_str.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception:
            return "[Encrypted Data / Decryption Error]"

    @staticmethod
    def create_jwt_token(data: dict, expires_delta_hours: int = 24) -> str:
        """Generates a secure JWT token for user authentication."""
        to_encode = data.copy()
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=expires_delta_hours)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

    @staticmethod
    def decode_jwt_token(token: str) -> Optional[dict]:
        """Decodes and validates a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None
