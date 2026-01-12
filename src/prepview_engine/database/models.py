from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import datetime
import uuid

Base = declarative_base()

# 1. USER TABLE (Login Info)
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


# 2. USER PROFILE (Resume Data)
class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    full_name = Column(String(100))
    # Hum JSON use kar rahay hain taakay flexible data store ho sakay
    education = Column(JSON, nullable=True)  # e.g., [{"degree": "BS CS", "year": "2024"}]
    experience = Column(JSON, nullable=True) # e.g., [{"role": "Intern", "company": "XYZ"}]
    skills = Column(JSON, nullable=True)     # e.g., ["Python", "React", "SQL"]
    projects = Column(JSON, nullable=True)   # e.g., [{"title": "PrepView", "desc": "..."}]
    
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="profile")


# 3. INTERVIEW SESSION (The "Unique ID" Holder)
class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    # Yeh hai wo UNIQUE ID (Session ID) jo har cheez ko link karegi
    session_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    target_role = Column(String(50), nullable=False) # e.g., "AI Engineer", "Frontend Dev"
    status = Column(String(20), default="in_progress") # "in_progress", "completed"
    started_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="interviews")
    chunks = relationship("InterviewChunk", back_populates="session")
    final_report = relationship("FinalReport", back_populates="session", uselist=False)


# 4. INTERVIEW CHUNKS (Per Question Analysis)
class InterviewChunk(Base):
    __tablename__ = "interview_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("interview_sessions.session_id"), nullable=False, index=True)
    
    question_id = Column(String(10), nullable=False) # e.g., "Q1", "Q2"
    video_path = Column(String(255), nullable=True)
    
    # Store Full Analysis JSONs here
    cv_analysis = Column(JSON, nullable=True)   # Head, Eye, Expression stats
    nlp_analysis = Column(JSON, nullable=True)  # WPM, Score, Transcript
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("InterviewSession", back_populates="chunks")


# 5. FINAL REPORT (Aggregated Result)
class FinalReport(Base):
    __tablename__ = "final_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("interview_sessions.session_id"), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Aggregated Stats
    summary_metrics = Column(JSON, nullable=False) # Averages
    ai_feedback = Column(Text, nullable=False)     # LLM Response
    
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("InterviewSession", back_populates="final_report")
    user = relationship("User", back_populates="reports")