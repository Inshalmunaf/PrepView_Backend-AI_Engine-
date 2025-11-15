from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.utils.common import logger
from prepview_engine.components.preprocessing import PreprocessingComponent
import os 
from prepview_engine.components.cv_analyzer import CVAnalyzerComponent
from prepview_engine.components.nlp_analyzer import NLPAnalyzerComponent

from pathlib import Path

def test_configuration():
    """
    Tests the ConfigurationManager to ensure all configs are loaded correctly.
    """
    logger.info("Starting configuration test...")
    
    try:
        # 1. Initialize the Configuration Manager
        config_manager = ConfigurationManager()
        
        # 2. Get Database Config (UPDATED)
        db_config = config_manager.get_database_config()
        logger.info("--- Database Config ---")
        logger.info(f"DB Type: {db_config.db_type}")
        logger.info(f"DB Host: {db_config.db_host}")
        logger.info(f"DB Name: {db_config.db_name}")
        logger.info(f"DB User: {db_config.db_user}")
        # Hamain password log nahi karna chahiyay (security risk hai)
        # logger.info(f"DB Password: {db_config.db_password}") 
        
        # Hamari nayi function ko test kartay hain:
        logger.info(f"SQLAlchemy URI: {db_config.get_sqlalchemy_uri()}")
        
        # 3. Get Preprocessing Config
        pre_config = config_manager.get_preprocessing_config()
        logger.info("--- Preprocessing Config ---")
        logger.info(f"Temp Video Path: {pre_config.temp_video_path}")
        
        # 4. Get NLP Config
        nlp_config = config_manager.get_nlp_config()
        logger.info("--- NLP Config ---")
        logger.info(f"STT Model: {nlp_config.stt_model_name}")
        
        # 5. Get Scoring Config
        score_config = config_manager.get_scoring_config() # Typo fixed: config_manager
        logger.info("--- Scoring Config ---")
        logger.info(f"Gaze Good Threshold: {score_config.gaze_good_threshold}")
        
        logger.info("Configuration test passed successfully!")
        
    except Exception as e:
        logger.error(f"Configuration test FAILED: {e}")
        logger.exception(e) # Ye poora error traceback print karay ga

def test_preprocessing():
    """
    Tests the PreprocessingComponent.
    """
    logger.info("--- Starting Preprocessing Component Test ---")
    
    try:
        # 1. Config load karain
        config_manager = ConfigurationManager()
        pre_config = config_manager.get_preprocessing_config()
        
        # 2. Aik dummy video file ka path banayen
        # (ASSUMPTION: Aapnay 'test_video.mp4' naam ki file
        # 'artifacts/temp_uploads/' folder mai rakh di hai)
        dummy_video_path = os.path.join(pre_config.temp_video_path, "test_video.mp4")
        
        # Check karain kay dummy file mojood hai
        if not os.path.exists(dummy_video_path):
            logger.warning(f"Test file not found at: {dummy_video_path}")
            logger.warning("Please place a 'test_video.mp4' file in 'artifacts/temp_uploads/' to run this test.")
            return

        # 3. Component ko initialize karain
        preprocessor = PreprocessingComponent(
            original_video_path=dummy_video_path,
            config=pre_config
        )
        
        # 4. Component ko run karain
        video_path, audio_path = preprocessor.run()
        
        # 5. Results check karain
        logger.info(f"Original video path returned: {video_path}")
        logger.info(f"Extracted audio path returned: {audio_path}")
        
        if os.path.exists(audio_path):
            logger.info("SUCCESS: Audio file was created successfully.")
        else:
            logger.error("FAILURE: Audio file was NOT created.")
            
        logger.info("--- Finished Preprocessing Component Test ---")

    except Exception as e:
        logger.error(f"Preprocessing test FAILED: {e}")
        logger.exception(e)

def test_cv_analyzer():
    """
    Tests the CVAnalyzerComponent.
    """
    logger.info("--- Starting CV Analyzer Component Test ---")
    
    try:
        # 1. Config load karain (sirf path k liye)
        config_manager = ConfigurationManager()
        pre_config = config_manager.get_preprocessing_config()
        
        # 2. Wahi dummy video file ka path
        dummy_video_path_str = os.path.join(pre_config.temp_video_path, "test_video.mp4")
        dummy_video_path = Path(dummy_video_path_str)
        
        if not dummy_video_path.exists():
            logger.warning(f"Test file not found at: {dummy_video_path}")
            logger.warning("Please place a 'test_video.mp4' file in 'artifacts/temp_uploads/' to run this test.")
            return

        # 3. Component ko initialize karain
        cv_analyzer = CVAnalyzerComponent(video_path=dummy_video_path)
        
        # 4. Component ko run karain
        results = cv_analyzer.run()
        
        # 5. Results check karain
        logger.info("CV Analyzer Test Results:")
        logger.info(results)
        
        if results and results.get("cv_total_frames", 0) > 0:
            logger.info("SUCCESS: CV Analyzer processed the video.")
        else:
            logger.error("FAILURE: CV Analyzer returned no results or processed 0 frames.")
            
        logger.info("--- Finished CV Analyzer Component Test ---")

    except Exception as e:
        logger.error(f"CV Analyzer test FAILED: {e}")
        logger.exception(e)

def test_nlp_analyzer():
    """
    Tests the NLPAnalyzerComponent.
    """
    logger.info("--- Starting NLP Analyzer Component Test ---")
    
    try:
        # ... (Config loading code waisay hi rahay ga) ...
        config_manager = ConfigurationManager()
        nlp_config = config_manager.get_nlp_config()
        pre_config = config_manager.get_preprocessing_config()
        
        dummy_audio_path_str = os.path.join(pre_config.temp_video_path, "test_video_audio.wav")
        dummy_audio_path = Path(dummy_audio_path_str)
        
        if not dummy_audio_path.exists():
            logger.warning(f"Test audio file not found at: {dummy_audio_path}")
            logger.warning("Please run 'test_preprocessing()' first (in main.py) to create this file.")
            return

        logger.info("Initializing NLPAnalyzerComponent... (This may take a moment to download models)")
        nlp_analyzer = NLPAnalyzerComponent(
            audio_path=dummy_audio_path,
            config=nlp_config
        )
        
        results = nlp_analyzer.run()
        
        logger.info("NLP Analyzer Test Results:")
        logger.info(results)
        
        # --- UPDATED TEST LOGIC ---
        if results and results.get("transcript"): # Check if transcript is not empty
            logger.info("SUCCESS: NLP Analyzer processed the audio and produced a transcript.")
        else:
            logger.error("FAILURE: NLP Analyzer did not produce a transcript.")
            
        logger.info("--- Finished NLP Analyzer Component Test ---")

    except Exception as e:
        logger.error(f"NLP Analyzer test FAILED: {e}")
        logger.exception(e)

if __name__ == "__main__":
    #test_configuration()
    #test_preprocessing()
    #test_cv_analyzer()
    test_nlp_analyzer()
    
