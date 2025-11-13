from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.utils.common import logger

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

if __name__ == "__main__":
    test_configuration()