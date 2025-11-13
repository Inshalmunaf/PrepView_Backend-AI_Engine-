from prepview_engine.utils.common import read_yaml, logger
from prepview_engine.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pydantic import BaseModel, DirectoryPath, FilePath
from pathlib import Path
import os
from dotenv import load_dotenv
# --- Step 1: Define data structures using Pydantic ---
# Ye models hamain config.yaml ki structure ko validate karnay mai madad dengay
# Aur IDE par auto-completion bhi dengey.
load_dotenv()
class DatabaseConfig(BaseModel):
    # UPDATED FOR POSTGRESQL
    db_type: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str

    def get_sqlalchemy_uri(self) -> str:
        return f"{self.db_type}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

class PreprocessingConfig(BaseModel):
    # DirectoryPath check karay ga kay ye folder exist karta hai
    temp_video_path: Path 

class NLPConfig(BaseModel):
    stt_model_name: str
    sentiment_model_name: str

class ScoringConfig(BaseModel):
    gaze_good_threshold: int
    gaze_avg_threshold: int
    filler_good_threshold: int
    filler_avg_threshold: int


# --- Step 2: The Main Configuration Manager Class ---

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):
        
        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            
            # config.yaml say artifacts_root ko create karna
            # (ye .gitignore mai hona chahaiye)
            os.makedirs(self.config.artifacts_root, exist_ok=True)
            
            logger.info("Configuration files loaded and artifacts root directory ensured.")
            
        except Exception as e:
            logger.error(f"Error loading configuration files: {e}")
            raise

    # --- Step 3: Getter methods ---
    # Har component/pipeline apni config in methods say mangay ga.
    
    def get_database_config(self) -> DatabaseConfig:
        """
        Returns database configuration.
        Non-sensitive info (db_type) comes from config.yaml.
        Sensitive info (user, pass, host, etc.) comes from .env file.
        """
        try:
            config = self.config.database # config.yaml say
            
            # Ab credentials environment say uthayen
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = int(os.getenv("DB_PORT")) # Port ko integer mai convert kara hai
            db_name = os.getenv("DB_NAME")

    
            if not all([db_user, db_password, db_host, db_port, db_name]):
                raise ValueError("Database credentials (DB_USER, DB_PASSWORD, etc.) not found in .env file.")

            return DatabaseConfig(
                db_type=config.db_type, # Ye config.yaml say aaya
                db_user=db_user,         # Ye .env say aaya
                db_password=db_password, # Ye .env say aaya
                db_host=db_host,         # Ye .env say aaya
                db_port=db_port,         # Ye .env say aaya
                db_name=db_name          # Ye .env say aaya
            )
        except Exception as e:
            logger.error(f"Error parsing database config: {e}")
            raise

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Returns preprocessing configuration (e.g., file paths)"""
        try:
            # config.yaml say artifacts_root aur temp_video_path ko join karna
            temp_path = Path(os.path.join(
                self.config.artifacts_root, 
                self.config.temp_video_path
            ))
            # Ensure directory exists
            os.makedirs(temp_path, exist_ok=True) 

            return PreprocessingConfig(temp_video_path=temp_path)
            
        except Exception as e:
            logger.error(f"Error parsing preprocessing config: {e}")
            raise

    def get_nlp_config(self) -> NLPConfig:
        """Returns NLP model parameters from params.yaml"""
        try:
            params = self.params.models
            return NLPConfig(
                stt_model_name=params.stt_model_name,
                sentiment_model_name=params.sentiment_model_name
            )
        except Exception as e:
            logger.error(f"Error parsing NLP params: {e}")
            raise

    def get_scoring_config(self) -> ScoringConfig:
        """Returns scoring thresholds from params.yaml"""
        try:
            params = self.params.scoring
            return ScoringConfig(
                gaze_good_threshold=params.gaze.good_threshold,
                gaze_avg_threshold=params.gaze.avg_threshold,
                filler_good_threshold=params.filler_words.good_threshold,
                filler_avg_threshold=params.filler_words.avg_threshold
            )
        except Exception as e:
            logger.error(f"Error parsing Scoring params: {e}")
            raise