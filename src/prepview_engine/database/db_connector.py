from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import DatabaseConfig
from .models import Base, InterviewChunk, FinalReport, InterviewSession
import numpy as np 

class DatabaseConnector:
    def __init__(self, config: DatabaseConfig):
        """
        Handles PostgreSQL connection via SQLAlchemy.
        """
        # Connection String format: postgresql://user:password@host:port/dbname
        self.db_uri = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        
        try:
            self.engine = create_engine(self.db_uri)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("✅ Database connection established.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to DB: {e}")
            raise e

    def init_db(self):
        """Creates tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise e

    def get_session(self):
        """Context manager for DB session."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
  # To convert the float32 into float 
    def _sanitize(self, obj):
        """
        Recursively converts Numpy types to standard Python types.
        (float32 -> float, int64 -> int, etc.)
        """
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return round(float(obj), 4) # 4 decimal places tak round kar diya
        elif isinstance(obj, (np.ndarray, list)):
            return [self._sanitize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        return obj
  # TO save the interview chunks in database 
    def save_chunk_result(self, session_id: str, question_id: str, cv_result: dict, nlp_result: dict):
        """
        Saves analysis data safely (Handles NoneTypes automatically).
        """
        session = self.SessionLocal()
        try:

            # --- 🛡️ SAFETY CHECK (The Fix) ---
            # Agar galti se None pass ho jaye, to usay Empty Dict bana do taakay .get() fail na ho
            if nlp_result is None: nlp_result = {}
            if cv_result is None: cv_result = {}
            nlp_result = self._sanitize(nlp_result)
            cv_result = self._sanitize(cv_result)
            # --- 1. Safe Extraction Helpers ---
            # Nested dictionaries ko safely nikalna (agar key ho magar value None ho)
            speech_metrics = nlp_result.get("speech_metrics") or {}
            linguistic_metrics = nlp_result.get("linguistic_metrics") or {}
            
            head_movement = cv_result.get("head_movement") or {}
            eye_gaze = cv_result.get("eye_gaze") or {}
            facial_expression = cv_result.get("facial_expression") or {}

            # --- 2. Create Object ---
            new_chunk = InterviewChunk(
                session_id=session_id,
                question_id=question_id,
                
                # NLP Mappings
                nlp_full_json=nlp_result,
                transcript=nlp_result.get("transcript", ""),
                
                # Humne upar 'or {}' lagaya hai, isliye ye ab safe hain
                speech_metrics=speech_metrics, 
                linguistic_metrics=linguistic_metrics,
                
                phase1_score=nlp_result.get("phase1_quality_score", 0.0),
                prosodic_confidence=nlp_result.get("prosodic_confidence", 0.0),

                # CV Mappings
                cv_full_json=cv_result,
                head_movement=head_movement,
                eye_gaze=eye_gaze,
                facial_expression=facial_expression,
                cv_score=cv_result.get("cv_score", 0.0)
            )

            session.add(new_chunk)
            session.commit()
            logger.info(f" Data Stored Safely: {question_id}")

        except Exception as e:
            session.rollback()
            logger.error(f" DB Save Error: {e}")
            raise e
        finally:
            session.close()

# Access All the chunks Analyses by Session id 
    def get_all_chunks(self, session_id: str):
        """
        Fetches all chunks matching the Session ID.
        Retrieves exact attributes shown in the database schema image.
        """
        session = self.SessionLocal()
        try:
            # 1. Query Database for specific session
            chunks = session.query(InterviewChunk).filter_by(session_id=session_id).all()
            
            logger.info(f"📂 Fetched {len(chunks)} chunks for Session: {session_id}")

            results_list = []
            for chunk in chunks:
                # 2. Map Database Columns to Dictionary
                # Hum wahi fields utha rahe hain jo Image mein hain
                chunk_data = {
                    "session_id": chunk.session_id,
                    "question_id": chunk.question_id,
                    
                    # --- Text Data ---
                    "transcript": chunk.transcript,

                    # --- Scores (Floats) ---
                    "phase1_score": chunk.phase1_score,
                    "prosodic_confidence": chunk.prosodic_confidence,

                    # --- NLP JSON Data (Detailed Metrics) ---
                    "speech_metrics": chunk.speech_metrics,          # Contains wpm, filler_rate
                    "linguistic_metrics": chunk.linguistic_metrics,  # Contains lexical_richness
                    "nlp_full_json": chunk.nlp_full_json,            # Backup Full Data

                    # --- CV JSON Data (Detailed Metrics) ---
                    "head_movement": chunk.head_movement,
                    "eye_gaze": chunk.eye_gaze,                      # Contains eye_contact_pct
                    "facial_expression": chunk.facial_expression,    # Contains mood, nervousness
                    "cv_full_json": chunk.cv_full_json,              # Backup Full Data
                    "cv_score": chunk.cv_score
                }
                
                results_list.append(chunk_data)

            return results_list

        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            return []
        finally:
            session.close()

# TO save final reports
    def save_final_report(self, session_id: str, nlp_data: dict, cv_data: dict, feedback_text: str):
        """
        Saves the Generated AI Report + Aggregated Metrics into the DB.
        Automatically finds user_id from the session.
        """
        session = self.SessionLocal()
        try:
            # 1. Find User ID linked to this Session
            # Hum session table check karte hain ke ye interview kis user ka tha
            interview_session = session.query(InterviewSession).filter_by(session_id=session_id).first()
            
            if not interview_session:
                logger.error(f"❌ Session {session_id} not found! Cannot save report.")
                return False
            
            user_id = interview_session.user_id

            # 2. Check if report already exists (Update logic)
            existing_report = session.query(FinalReport).filter_by(session_id=session_id).first()
            
            if existing_report:
                logger.info(f"🔄 Updating existing report for Session: {session_id}")
                existing_report.nlp_aggregate = nlp_data
                existing_report.cv_aggregate = cv_data
                existing_report.ai_feedback = feedback_text
            else:
                # 3. Create New Report
                logger.info(f"📝 Creating new report for Session: {session_id}")
                new_report = FinalReport(
                    session_id=session_id,
                    user_id=user_id,
                    nlp_aggregate=nlp_data,
                    cv_aggregate=cv_data,
                    ai_feedback=feedback_text
                )
                session.add(new_report)

            session.commit()
            logger.info("✅ Final Report Saved Successfully!")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to save final report: {e}")
            return False
        finally:
            session.close()