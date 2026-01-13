import os
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.database.db_connector import DatabaseConnector

# ✅ Import Actual Components
from prepview_engine.components.preprocessing import PreprocessingComponent
from prepview_engine.components.cv_analyzer import CVAnalyzerComponent
from prepview_engine.components.nlp_analyzer import NLPAnalyzerComponent

class AnalysisPipeline:
    def __init__(self):
        """
        Initializes the entire engine: DB, Preprocessor, NLP, and CV models.
        """
        logger.info("Initializing Analysis Pipeline Components...")
        
        self.config_manager = ConfigurationManager()
        
        # 1. Database Connection
        self.db = DatabaseConnector(self.config_manager.get_database_config())
        
        # 2. Initialize Preprocessor
        # Hum sirf Config pass kar rahay hain (Video path run time pe ayega)
        self.preprocessor = PreprocessingComponent(config=self.config_manager.get_preprocessing_config())
        
        # 3. Initialize AI Models
        logger.info("Loading CV Model Configuration...")
        self.cv_analyzer = CVAnalyzerComponent(self.config_manager.get_cv_config())
        
        logger.info("Loading NLP Model Configuration...")
        self.nlp_analyzer = NLPAnalyzerComponent(self.config_manager.get_nlp_config())
        
        logger.info("Pipeline Components Ready.")

    def run_pipeline(self, session_id: str, question_id: str, video_path: str):
        """
        Orchestrates the analysis flow for a single video chunk.
        Flow: Video -> Audio Extraction -> CV Analysis -> NLP Analysis -> Database
        """
        logger.info(f"[START] Pipeline triggered for Session: {session_id} | Question: {question_id}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found at: {video_path}")
            return False

        try:
            # --- STEP 1: PREPROCESSING (Audio Extraction) ---
            logger.info("Step 1: Running Preprocessing (Extracting Audio)...")
            
            # New Style: Calling run with video path
            audio_path = self.preprocessor.run(video_path) 
            
            if not audio_path or not os.path.exists(audio_path):
                raise Exception("Audio extraction failed or file not created.")

            # --- STEP 2: CV ANALYSIS (Video) ---
            logger.info("Step 2: Running CV Analysis...")
            cv_results = self.cv_analyzer.run(video_path)
            
            # --- STEP 3: NLP ANALYSIS (Audio) ---
            logger.info("Step 3: Running NLP Analysis...")
            # Ensure path is converted to string for safety
            nlp_results = self.nlp_analyzer.run(str(audio_path))

            # --- STEP 4: STORAGE (Database) ---
            logger.info("Step 4: Saving Results to Database...")
            
            self.db.save_chunk_result(
                session_id=session_id,
                question_id=question_id,
                cv_result=cv_results,
                nlp_result=nlp_results
            )

            logger.info(f"[SUCCESS] Chunk {question_id} Processed & Saved Successfully!")
            return True

        except Exception as e:
            logger.error(f"Pipeline Failed for {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return False