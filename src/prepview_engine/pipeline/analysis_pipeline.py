from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.components.preprocessing import PreprocessingComponent
from prepview_engine.components.cv_analyzer import CVAnalyzerComponent
from prepview_engine.components.nlp_analyzer import NLPAnalyzerComponent
from prepview_engine.components.report_generator import ReportGeneratorComponent
from prepview_engine.database.db_connector import DatabaseConnector
from pathlib import Path
import os

class AnalysisPipeline:
    def __init__(self, video_path: str, session_id: str):
        """
        Initializes the main analysis pipeline.
        
        Args:
            video_path (str): The path to the uploaded video file.
            session_id (str): The unique session ID for this analysis.
        """
        self.video_path = Path(video_path)
        self.session_id = session_id
        
        # Load all configurations at once
        logger.info("Initializing AnalysisPipeline: Loading configurations...")
        config_manager = ConfigurationManager()
        self.pre_config = config_manager.get_preprocessing_config()
        self.nlp_config = config_manager.get_nlp_config()
        self.scoring_config = config_manager.get_scoring_config()
        db_config = config_manager.get_database_config()
        
        # Initialize database connector
        self.db_connector = DatabaseConnector(config=db_config)
        
        # Ensure database tables are created (idempotent check)
        self.db_connector.create_tables()
        logger.info("AnalysisPipeline initialized successfully.")

    def run(self):
        """
        Runs the full analysis pipeline from start to finish.
        """
        logger.info(f"--- Starting Full Analysis Pipeline for Session: {self.session_id} ---")
        try:
            # --- Step 1: Preprocessing ---
            preprocessor = PreprocessingComponent(
                original_video_path=self.video_path,
                config=self.pre_config
            )
            video_path, audio_path = preprocessor.run()
            
            # --- Step 2: CV Analysis ---
            cv_analyzer = CVAnalyzerComponent(video_path=video_path)
            cv_results = cv_analyzer.run()
            
            # --- Step 3: NLP Analysis ---
            nlp_analyzer = NLPAnalyzerComponent(
                audio_path=audio_path,
                config=self.nlp_config
            )
            nlp_results = nlp_analyzer.run()
            
            # --- Step 4: Report Generation & Saving ---
            report_gen = ReportGeneratorComponent(
                cv_results=cv_results,
                nlp_results=nlp_results,
                session_id=self.session_id,
                scoring_config=self.scoring_config,
                db_connector=self.db_connector
            )
            report_gen.run()
            
            logger.info(f"--- Full Analysis Pipeline COMPLETED for Session: {self.session_id} ---")

        except Exception as e:
            logger.error(f"Full analysis pipeline FAILED for Session {self.session_id}: {e}")
            logger.exception(e)
            # Yahan aap error ko DB mai bhi log kar saktay hain
            
        finally:
            # --- Step 5: Cleanup (Delete temp files) ---
            # Analysis kay baad temp files delete kar dain
            try:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Cleaned up temp audio file: {audio_path}")
                
                # Note: Original video file ko abhi delete nahi karain gay
                # kyunkay woh `app.py` say aa rahi hai. Ham `app.py`
                # mai isay baad mai handle kar saktay hain.
                # Agar video file bhi isi pipeline nay generate ki hoti,
                # toh ham usay yahan delete kartay.
                
            except Exception as e:
                logger.warning(f"Error during file cleanup: {e}")