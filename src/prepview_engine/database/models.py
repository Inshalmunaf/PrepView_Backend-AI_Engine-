from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

# Base class for all models
Base = declarative_base()

# ==========================================
# 1. USER & PROFILE (Authentication & Resume)
# ==========================================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    interviews = relationship("InterviewSession", back_populates="user")
    reports = relationship("FinalReport", back_populates="user")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    full_name = Column(String(100))
    education = Column(JSON, nullable=True)  # e.g., [{"degree": "BS", "year": "2024"}]
    experience = Column(JSON, nullable=True) # e.g., [{"role": "Dev", "company": "X"}]
    skills = Column(JSON, nullable=True)     # e.g., ["Python", "FastAPI"]
    projects = Column(JSON, nullable=True)   # e.g., [{"title": "PrepView"}]
    
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user = relationship("User", back_populates="profile")


# ==========================================
# 2. SESSION MANAGEMENT (The Unique Link)
# ==========================================
class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    # UUID based Session ID (Unique per interview attempt)
    session_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    target_role = Column(String(50), nullable=False) # e.g., "AI Engineer"
    status = Column(String(20), default="in_progress") # "in_progress", "completed"
    started_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    chunks = relationship("InterviewChunk", back_populates="session")
    final_report = relationship("FinalReport", back_populates="session", uselist=False)
    user = relationship("User", back_populates="interviews")


# ==========================================
# 3. INTERVIEW CHUNKS (Per Question Analysis)
# ==========================================
class InterviewChunk(Base):
    __tablename__ = "interview_chunks"

    # --- Core Identifiers ---
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("interview_sessions.session_id"), nullable=False, index=True)
    question_id = Column(String(10), nullable=False) # e.g., "Q1"
    # ======================================
    # 🧠 NLP DATA STORAGE
    # ======================================
    # 1. Full Raw Data (Backup)
    nlp_full_json = Column(JSON, nullable=True) 

    # 2. Separated Components (Easy Access)
    transcript = Column(Text, nullable=True)
    speech_metrics = Column(JSON, nullable=True)      # Stores entire speech_metrics dict
    linguistic_metrics = Column(JSON, nullable=True)  # Stores entire linguistic_metrics dict
    
    # 3. Direct Scores (Fast Querying)
    phase1_score = Column(Float, nullable=True)
    prosodic_confidence = Column(Float, nullable=True)

    # ======================================
    # 👁️ CV DATA STORAGE
    # ======================================
    # 1. Full Raw Data (Backup)
    cv_full_json = Column(JSON, nullable=True)

    # 2. Separated Components (Easy Access)
    head_movement = Column(JSON, nullable=True)       # Stores head_movement dict
    eye_gaze = Column(JSON, nullable=True)            # Stores eye_gaze dict
    facial_expression = Column(JSON, nullable=True)   # Stores facial_expression dict

    # --- Timestamps ---
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("InterviewSession", back_populates="chunks")


# ==========================================
# 4. FINAL REPORT (Aggregated Summary)
# ==========================================
class FinalReport(Base):
    __tablename__ = "final_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("interview_sessions.session_id"), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # --- Aggregated Key Stats ---
    avg_score = Column(Float, nullable=True)
    avg_wpm = Column(Float, nullable=True)
    avg_eye_contact = Column(Float, nullable=True)
    avg_prosodic_confidence = Column(Float, nullable=True) # <--- NEW: Average Audio Score
    
    # --- Full Summary & Feedback ---
    summary_metrics = Column(JSON, nullable=False) # Master JSON of averages
    ai_feedback = Column(Text, nullable=False)     # Text generated by LLM
    
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("InterviewSession", back_populates="final_report")
    user = relationship("User", back_populates="reports")