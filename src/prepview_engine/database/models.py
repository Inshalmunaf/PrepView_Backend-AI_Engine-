from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Base class for our models
Base = declarative_base()

class AnalysisReport(Base):
    """
    SQLAlchemy model for storing analysis reports.
    """
    __tablename__ = "analysis_reports"

    # Core columns
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    # question_id = Column(String, index=True) # Baad mai add kar saktay hain
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # NLP Analysis Results
    transcript = Column(String, nullable=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    filler_word_count = Column(Integer, nullable=True)
    total_words = Column(Integer, nullable=True)

    # CV Analysis Results
    total_frames = Column(Integer, nullable=True)
    gaze_analysis_percent = Column(JSON, nullable=True) # e.g., {'center': 90.0, ...}
    posture_analysis_percent = Column(JSON, nullable=True) # e.g., {'upright': 80.0, ...}

    # Scoring Engine Results
    gaze_score = Column(Float, nullable=True)
    posture_score = Column(Float, nullable=True)
    filler_word_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Feedback text
    gaze_feedback = Column(String, nullable=True)
    posture_feedback = Column(String, nullable=True)
    filler_word_feedback = Column(String, nullable=True)

    def __repr__(self):
        return f"<AnalysisReport(session_id='{self.session_id}', score='{self.overall_score}')>"