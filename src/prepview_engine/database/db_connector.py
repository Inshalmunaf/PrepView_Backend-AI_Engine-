from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import DatabaseConfig
from .models import Base, InterviewChunk, FinalReport, InterviewSession

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

    # --- SPECIFIC QUERIES FOR YOUR FLOW ---

    def save_chunk_result(self, session_id: str, question_id: str, cv_data: dict, nlp_data: dict, video_path: str):
        """Saves a single question's analysis."""
        session = self.SessionLocal()
        try:
            chunk = InterviewChunk(
                session_id=session_id,
                question_id=question_id,
                cv_analysis=cv_data,
                nlp_analysis=nlp_data,
                video_path=video_path
            )
            session.add(chunk)
            session.commit()
            logger.info(f"💾 Chunk Saved: {question_id} for Session {session_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving chunk: {e}")
            raise e
        finally:
            session.close()

    def get_all_chunks(self, session_id: str):
        """Fetches all chunks for aggregation."""
        session = self.SessionLocal()
        try:
            chunks = session.query(InterviewChunk).filter_by(session_id=session_id).all()
            # Convert SQLAlchemy objects to list of dicts for Aggregator
            return [
                {
                    "question_id": c.question_id, 
                    "cv_analysis": c.cv_analysis, 
                    "nlp_analysis": c.nlp_analysis
                } 
                for c in chunks
            ]
        finally:
            session.close()

    def save_final_report(self, session_id: str, user_id: int, summary: dict, feedback: str):
        """Saves the final generated report."""
        session = self.SessionLocal()
        try:
            report = FinalReport(
                session_id=session_id,
                user_id=user_id,
                summary_metrics=summary,
                ai_feedback=feedback
            )
            session.add(report)
            
            # Update Session Status to completed
            interview = session.query(InterviewSession).filter_by(session_id=session_id).first()
            if interview:
                interview.status = "completed"
            
            session.commit()
            logger.info(f"✅ Final Report Saved for Session {session_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving report: {e}")
            raise e
        finally:
            session.close()