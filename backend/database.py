import os
import uuid
import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from security import SecurityService

# Database URI (Supports SQLite for local dev, PostgreSQL for production)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "pmds_secure.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserTable(Base):
    __tablename__ = "users"

    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Encrypted PII Fields (AES-256)
    encrypted_full_name = Column(Text, nullable=True)
    encrypted_email = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    test_results = relationship("TestResultTable", back_populates="user", cascade="all, delete-orphan")

class TestResultTable(Base):
    __tablename__ = "test_results"

    test_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    
    # Encrypted Clinical Sensor & Biomarker Data (AES-256)
    encrypted_sensor_data = Column(Text, nullable=True)
    
    risk_score = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)
    diagnosis = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("UserTable", back_populates="test_results")

# Initialize database tables
def init_db():
    Base.metadata.create_all(bind=engine)

class DatabaseService:
    @staticmethod
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def register_user(db, username: str, password_raw: str, full_name: str = "", email: str = "") -> Dict[str, Any]:
        """Registers user with bcrypt password hash and AES-256 encrypted PII."""
        import re
        email_clean = email.strip()
        if email_clean:
            email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            if not re.match(email_regex, email_clean):
                raise ValueError("รูปแบบอีเมลไม่ถูกต้อง (ต้องมีเครื่องหมาย @ และชื่อโดเมน เช่น user@example.com)")

        import hashlib
        # Hash username deterministically so plain text username/email is not stored
        anon_username = hashlib.sha256(username.lower().strip().encode('utf-8')).hexdigest()[:16]

        existing = db.query(UserTable).filter(UserTable.username == anon_username).first()
        if existing:
            raise ValueError("ผู้ใช้งานนี้ถูกลงทะเบียนแล้ว")

        hashed_pwd = SecurityService.hash_password(password_raw)
        enc_name = SecurityService.encrypt_data(full_name) if full_name else ""
        enc_email = SecurityService.encrypt_data(email) if email else ""

        user = UserTable(
            username=anon_username,
            password_hash=hashed_pwd,
            encrypted_full_name=enc_name,
            encrypted_email=enc_email
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "user_id": user.user_id,
            "username": user.username,
            "full_name": full_name,
            "email": email
        }

    @staticmethod
    def authenticate_user(db, username: str, password_raw: str) -> Optional[Dict[str, Any]]:
        """Authenticates user and decrypts PII upon valid login."""
        import hashlib
        anon_username = hashlib.sha256(username.lower().strip().encode('utf-8')).hexdigest()[:16]
        user = db.query(UserTable).filter(UserTable.username == anon_username).first()
        if not user or not SecurityService.verify_password(password_raw, user.password_hash):
            return None

        dec_name = SecurityService.decrypt_data(user.encrypted_full_name)
        dec_email = SecurityService.decrypt_data(user.encrypted_email)

        return {
            "user_id": user.user_id,
            "username": user.username,
            "full_name": dec_name,
            "email": dec_email
        }

    @staticmethod
    def save_test_result(db, user_id: str, sensor_data_raw: str, risk_score: float, risk_percent: float, diagnosis: str) -> Dict[str, Any]:
        """Saves clinical test result with AES-256 encrypted raw sensor biomarkers."""
        enc_sensor = SecurityService.encrypt_data(sensor_data_raw)
        
        record = TestResultTable(
            user_id=user_id,
            encrypted_sensor_data=enc_sensor,
            risk_score=risk_score,
            risk_percent=risk_percent,
            diagnosis=diagnosis
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "test_id": record.test_id,
            "user_id": record.user_id,
            "risk_score": record.risk_score,
            "risk_percent": record.risk_percent,
            "diagnosis": record.diagnosis,
            "created_at": record.created_at.isoformat()
        }

# Create tables upon import
init_db()
