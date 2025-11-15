from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prepview_engine.config.configuration import DatabaseConfig
from prepview_engine.utils.common import logger
from .models import Base # Apni models.py say Base import karain

class DatabaseConnector:
    """
    Handles database connection and session management.
    """
    def __init__(self, config: DatabaseConfig):
        self.db_uri = config.get_sqlalchemy_uri()
        try:
            self.engine = create_engine(self.db_uri)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Database connection pool established.")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

    def create_tables(self):
        """Creates all the tables defined in models.py"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables checked/created successfully.")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def get_session(self):
        """Provides a new database session"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()