import os
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.database.db_connector import DatabaseConnector

# ✅ Import Your Actual Components
from prepview_engine.components.preprocessing import PreprocessingComponent
from prepview_engine.components.cv_analyzer import CVAnalyzerComponent
from prepview_engine.components.nlp_analyzer import NLPAnalyzerComponent

class AnalysisPipeline:
    def __init__(self):
        """
        Initializes all workers (DB, Preprocessor, NLP, CV).
        """
        logger.info("⚙️ Initializing Engine Components...")
        
        self.config_manager = ConfigurationManager()
        
        # 1. Connect to Database
        self.db = DatabaseConnector(self.config_manager.get_database_config())
        
        # 2. Initialize Preprocessor (Video -> Audio converter)
        # (Assuming data ingestion config holds raw data paths)
        self.preprocessor = PreprocessingComponent(self.config_manager.get_data_ingestion_config())
        
        # 3. Initialize AI Models (FIXED HERE 👇)
        # Ab hum alag alag configs pass kar rahay hain jaisa aapne bataya
        logger.info("loading CV Config...")
        self.cv_analyzer = CVAnalyzerComponent(self.config_manager.get_cv_config())
        
        logger.info("loading NLP Config...")
        self.nlp_analyzer = NLPAnalyzerComponent(self.config_manager.get_nlp_config())
        
        logger.info("✅ Pipeline Components Ready!")

    def run_pipeline(self, session_id: str, question_id: str, video_path: str):
        """
        Main execution flow: Preprocess -> CV -> NLP -> DB Store
        """
        logger.info(f"🚀 [START] Pipeline triggered for Session: {session_id} | Question: {question_id}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}") # Emoji removed to prevent crash
            return False

        try:
            # --- STEP 1: PREPROCESSING (Run Audio Extraction) ---
            logger.info("Step 1: Running Preprocessing (Extracting Audio)...")
            
            # Ye function video lega aur audio file ka path return karega
            audio_path = self.preprocessor.run(video_path) 
            
            if not audio_path or not os.path.exists(audio_path):
                raise Exception("Audio extraction failed or file not created.")

            # --- STEP 2: CV ANALYSIS (Run Visuals) ---
            logger.info("Step 2: Running CV Analyzer...")
            # CV Analyzer video path lega
            cv_results = self.cv_analyzer.run(video_path)
            
            # --- STEP 3: NLP ANALYSIS (Run Audio/Text) ---
            logger.info("Step 3: Running NLP Analyzer...")
            # NLP Analyzer audio path lega
            nlp_results = self.nlp_analyzer.run(audio_path)

            # --- STEP 4: DATABASE STORAGE ---
            logger.info("Step 4: Storing Results in Database...")
            
            self.db.save_chunk_result(
                session_id=session_id,
                question_id=question_id,
                cv_result=cv_results,
                nlp_result=nlp_results
            )

            logger.info(f"✅ [SUCCESS] Analysis Completed & Saved for {question_id}")
            return True

        except Exception as e:
            logger.error(f"Pipeline Error on {question_id}: {e}") # Emoji removed
            import traceback
            traceback.print_exc()
            return False