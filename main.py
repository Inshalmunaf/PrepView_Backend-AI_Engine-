from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.utils.common import logger
from prepview_engine.components.preprocessing import PreprocessingComponent
import os 
from prepview_engine.components.cv_analyzer import CVAnalyzerComponent
from prepview_engine.components.nlp_analyzer import NLPAnalyzerComponent
from prepview_engine.database.db_connector import DatabaseConnector
from prepview_engine.components.report_generator import ReportGenerator
from prepview_engine.pipeline.analysis_pipeline import AnalysisPipeline
from prepview_engine.components.result_aggregator import ResultAggregator
import uuid
from pathlib import Path

from prepview_engine.database.models import User, InterviewSession


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
        dummy_video_path = os.path.join(pre_config.temp_video_path, "samplev14.mp4")
        
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
        cv_config = config_manager.get_cv_config()
        pre_config = config_manager.get_preprocessing_config()
        
        # 2. Wahi dummy video file ka path
        dummy_video_path_str = os.path.join(pre_config.temp_video_path, "samplev14.mp4")
        dummy_video_path = Path(dummy_video_path_str)
        
        if not dummy_video_path.exists():
            logger.warning(f"Test file not found at: {dummy_video_path}")
            logger.warning("Please place a '.mp4' file in 'artifacts/temp_uploads/' to run this test.")
            return

        # 3. Component ko initialize karain
        cv_analyzer = CVAnalyzerComponent(
            video_path=dummy_video_path,
            config = cv_config
        )
        
        # 4. Component ko run karain
        results = cv_analyzer.run()
        print(results)
        
        # 5. Results check karain
        logger.info("CV Analyzer Test Results:")
        logger.info(results)
        
        if results:
            logger.info("SUCCESS: CV Analyzer processed the video.")
        else:
            logger.error("FAILURE: CV Analyzer returned no results or processed 0 frames.")
            
        logger.info("--- Finished CV Analyzer Component Test ---")
        return results
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
        
        dummy_audio_path_str = os.path.join(pre_config.temp_video_path, "samplec.wav")
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
        
        results = nlp_analyzer.run(session_id=223,question_id=1)
        
        logger.info("NLP Analyzer Test Results:")
        logger.info(results)
        
        # --- UPDATED TEST LOGIC ---
        if results and results.get("transcript"): # Check if transcript is not empty
            logger.info("SUCCESS: NLP Analyzer processed the audio and produced a transcript.")
        else:
            logger.error("FAILURE: NLP Analyzer did not produce a transcript.")
            
        logger.info("--- Finished NLP Analyzer Component Test ---")
        return results
    except Exception as e:
        logger.error(f"NLP Analyzer test FAILED: {e}")
        logger.exception(e)

def test_report_generator():
    """
    Tests the ReportGeneratorComponent (Scoring + DB Write).
    """
    logger.info("--- Starting Report Generator Component Test ---")
    
    try:
        # 1. Config load karain
        config_manager = ConfigurationManager()
        db_config = config_manager.get_database_config()
        scoring_config = config_manager.get_scoring_config()
        
        # 2. Database Connector aur Tables banayen
        logger.info("Initializing Database Connector...")
        db_connector = DatabaseConnector(config=db_config)
        
        # Ye line database mai 'analysis_reports' table banaye gi
        db_connector.create_tables() 

        # 3. Dummy data (jo hamaray pichlay components say milta)
        # Ham yahan sample data hardcode kar rahay hain
        dummy_cv_results = {
            'cv_total_frames': 1500,
            'gaze_analysis': {'center': 85.0, 'no_face_detected': 15.0},
            'posture_analysis': {'upright': 90.0, 'slouched': 10.0}
        }
        dummy_nlp_results = {
            'transcript': 'This is a test transcript. Um, I think it is good.',
            'sentiment': {'label': 'POSITIVE', 'score': 0.95},
            'communication': {'filler_word_count': 1, 'total_words': 10}
        }
        dummy_session_id = "test_session_52345"

        # 4. Component ko initialize karain
        report_gen = ReportGeneratorComponent(
            cv_results=dummy_cv_results,
            nlp_results=dummy_nlp_results,
            session_id=dummy_session_id,
            scoring_config=scoring_config,
            db_connector=db_connector
        )
        
        # 5. Component ko run karain
        report_gen.run()
        
        logger.info("SUCCESS: Report Generator ran and saved data to database.")
        logger.info("Please check your 'prepview_db' database in PostgreSQL to confirm.")
        logger.info("--- Finished Report Generator Component Test ---")

    except Exception as e:
        logger.error(f"Report Generator test FAILED: {e}")
        logger.exception(e)


def test_full_analysis_pipeline():
    """
    Tests the full AnalysisPipeline from video-in to database-out.
    """
    logger.info("--- Starting FULL End-to-End Pipeline Test ---")
    
    try:
        # 1. Config load karain (sirf path k liye)
        config_manager = ConfigurationManager()
        pre_config = config_manager.get_preprocessing_config()
        
        # 2. Wahi dummy video file ka path
        dummy_video_path_str = os.path.join(pre_config.temp_video_path, "test_video.mp4")
        
        if not os.path.exists(dummy_video_path_str):
            logger.warning(f"Test file not found at: {dummy_video_path_str}")
            logger.warning("Please place a 'test_video.mp4' file in 'artifacts/temp_uploads/' to run this test.")
            return

        # 3. Aik unique session ID banayen
        test_session_id = f"e2e_test_{uuid.uuid4()}"
        logger.info(f"Using test session ID: {test_session_id}")

        # 4. Pipeline ko initialize karain
        pipeline = AnalysisPipeline(
            video_path=dummy_video_path_str,
            session_id=test_session_id
        )
        
        # 5. Pipeline ko run karain
        # Ye poora process (Pre, CV, NLP, Report) chalaye ga
        pipeline.run()
        
        logger.info("--- Finished FULL End-to-End Pipeline Test ---")
        logger.info(f"SUCCESS: Pipeline ran. Please check your database for a report with session_id: {test_session_id}")

    except Exception as e:
        logger.error(f"Full Pipeline test FAILED: {e}")
        logger.exception(e)

def init_database():
    print("🚀 Initializing Database...")
    
    # 1. Config Load karein
    try:
        config_manager = ConfigurationManager()
        db_config = config_manager.get_database_config()
        print(f"✅ Configuration Loaded for DB: {db_config.database}")
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    # 2. Connection banayen
    try:
        connector = DatabaseConnector(db_config)
        print("✅ Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"❌ Connection Failed. Check .env credentials! Error: {e}")
        return

    # 3. Tables Create karein
    try:
        connector.init_db()  # Yeh models.py se tables bana dega
        print("🎉 SUCCESS! All tables created in 'prepview_db'.")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")


# To Check interview chunks are saved to database or not 
def ensure_setup(db, sess_id):
    """User aur Session create karta hai taakay Data save hosakay"""
    s = db.SessionLocal()
    try:
        # Dummy User
        if not s.query(User).filter_by(username="Tester").first():
            s.add(User(username="Tester", email="test@test.com", password_hash="123"))
            s.commit()
            print("👤 Dummy User Created")
        
        # Dummy Session
        user = s.query(User).filter_by(username="Tester").first()
        if not s.query(InterviewSession).filter_by(session_id=sess_id).first():
            s.add(InterviewSession(session_id=sess_id, user_id=user.id, target_role="Dev"))
            s.commit()
            print("📅 Dummy Session Created")
    except Exception as e:
        print(e)
    finally:
        s.close()

def saving_test():
    print("🚀 Starting Test...")
    
    # A. Setup DB
    config = ConfigurationManager()
    db = DatabaseConnector(config.get_database_config())
    
    
    # B. Define IDs
    SESSION_ID = "test_session_no_video_01"
    QUESTION_ID = "Q1"
    
    # C. Prepare Foreign Keys
    ensure_setup(db, SESSION_ID)

    # D. Get Mock Data
    nlp_data = test_nlp_analyzer()
    cv_data = test_cv_analyzer()
    print(f"DEBUG MAIN: NLP Data Type: {type(nlp_data)}") # Should be <class 'dict'>
    print(f"DEBUG MAIN: CV Data Type: {type(cv_data)}")
    # E. SAVE (Calling the function)
    try:
        db.save_chunk_result(
            session_id=SESSION_ID,
            question_id=QUESTION_ID,
            cv_result=cv_data,
            nlp_result=nlp_data
        )
        print("\n✅ TEST PASSED: Data saved successfully without video_path!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")

# Testing if all chuncks are accessed from the database based on session id
def test_retrieval():
    print("🚀 TESTING DATA RETRIEVAL...")

    # 1. Initialize DB
    config = ConfigurationManager()
    db = DatabaseConnector(config.get_database_config())

    # 2. Session ID (Jo aapne Save Test mein use kiya tha)
    TARGET_SESSION = "test_session_no_video_01" 

    # 3. Fetch Data
    chunks = db.get_all_chunks(TARGET_SESSION)

    # 4. Validate Results
    if chunks:
        print(f"✅ Success! Found {len(chunks)} chunks.")
        
        # Pehle chunk ko inspect karte hain
        for i in range(len(chunks)):
            first_chunk = chunks[i]
            
            print("\n--- 🔍 Inspecting Attributes (Based on Image) ---")
            print(f"1. Question ID: {first_chunk.get('question_id')}")
            print(f"2. Transcript:  {first_chunk.get('transcript')}")
            print(f"3. Prosodic Conf: {first_chunk.get('prosodic_confidence')}")
            
            # Checking JSONs
            print(f"4. Speech Metrics: {first_chunk.get('speech_metrics')}")
            print(f"5. Facial Expression: {first_chunk.get('facial_expression')}")
            
            print("\n✅ All attributes accessed successfully!")
    else:
        print("❌ No data found. Did you run the 'Saving Test' first?")

#Check the Aggregation is done properly or not 
def run_db_aggregator_test():
        print("🚀 TESTING AGGREGATOR (With Internal DB Call)...")

        # 1. Setup DB Connection
        config_manager = ConfigurationManager()
        db = DatabaseConnector(config_manager.get_database_config())

        # 2. Setup Aggregator (Pass DB instance here)
        aggregator = ResultAggregator(db_connector=db)

        # 3. Use the Session ID from your previous Database Test
        TARGET_SESSION = "test_session_no_video_01" 

        # 4. Run Aggregation (Sirf ID deni hai!)
        print(f"🔄 Calling aggregator for session: {TARGET_SESSION}")
        result = aggregator.aggregate_session(TARGET_SESSION)

        # 5. Validate
        if result:
            print("\n✅ AGGREGATION SUCCESSFUL!")
            print("="*40)
            
            nlp_data = result.get("nlp_aggregate", {})
            cv_data = result.get("cv_aggregate", {})

            print(f"🔹 Avg WPM: {nlp_data.get('avg_wpm')}")
            print(f"🔹 Avg Eye Contact: {cv_data.get('avg_eye_contact')}%")
            print(f"🔹 Weakest Question: {nlp_data.get('weakest_answer_id')}")
            print(f"🔹 Transcript Context: \"{nlp_data.get('transcript_sample')[:50]}...\"")
            print(f"🔹 Avg WPM: {cv_data.get('avg_cv_score')}")


            print("="*40)
        else:
            print("❌ Failed: Result empty. Check Session ID or DB.")


# To Check that feedback generation is done properly ornot 
def run_generation():
    print("\n" + "="*50)
    print("🚀 STARTING FULL PIPELINE TEST")
    print("="*50)

    try:
        # --- STEP 1: INITIALIZE COMPONENTS ---
        logger.info("1️⃣ Loading Configuration...")
        config_manager = ConfigurationManager()
        
        logger.info("2️⃣ Connecting to Database...")
        db = DatabaseConnector(config_manager.get_database_config())

        logger.info("3️⃣ Initializing Aggregator...")
        aggregator = ResultAggregator(db_connector=db)

        logger.info("4️⃣ Initializing Report Generator (LLM)...")
        report_gen = ReportGenerator(config_manager.get_report_generation_config())

        # --- STEP 2: DEFINE TARGET ---
        # Wahi Session ID use karein jo DB mai save kiya tha
        TARGET_SESSION_ID = "test_session_no_video_01"

        # --- STEP 3: AGGREGATE DATA ---
        print(f"\n🔄 Fetching & Aggregating Data for Session: {TARGET_SESSION_ID}...")
        aggregated_data = aggregator.aggregate_session(TARGET_SESSION_ID)

        if not aggregated_data:
            logger.error(" Aggregation returned empty data! Check Session ID.")
            return

        # Quick Check of what we found
        nlp = aggregated_data.get("nlp_aggregate", {})
        print(f"   > Weakest Question Found: {nlp.get('weakest_answer_id')}")
        print(f"   > Transcript Snippet: \"{nlp.get('transcript_sample')[:50]}...\"")

        # --- STEP 4: GENERATE AI REPORT ---
        print("\n🤖 Sending Data to AI Coach (Ollama)... This may take a few seconds.")
        final_report = report_gen.generate_feedback(aggregated_data)

        # --- STEP 5: PRINT RESULT ---
        print("\n" + "="*50)
        print("📄 FINAL PROFESSIONAL REPORT")
        print("="*50)
        print(final_report)
        print("="*50)
        print("\n TEST COMPLETED SUCCESSFULLY!")

    except Exception as e:
        logger.error(f" Pipeline Crashed: {e}")
        import traceback
        traceback.print_exc()


# To test the saving of finalreport in database 
def run_saving_final_report():
    print("\n" + "="*50)
    print("STARTING FULL PIPELINE (WITH SAVE)")
    print("="*50)

    try:
        # 1. Init
        logger.info("[STEP 1] Loading Config & DB...")
        config_manager = ConfigurationManager()
        db = DatabaseConnector(config_manager.get_database_config())
        
        # ⚠️ IMPORTANT: Re-init DB tables because we changed models.py
        # (Only do this in Dev environment, NOT production)
        db.init_db() 

        aggregator = ResultAggregator(db_connector=db)
        report_gen = ReportGenerator(config_manager.get_report_generation_config())

        # 2. Define Target
        TARGET_SESSION_ID = "test_session_no_video_01"

        # 3. Aggregate
        print(f"\n[PROCESSING] Aggregating Data for: {TARGET_SESSION_ID}...")
        aggregated_data = aggregator.aggregate_session(TARGET_SESSION_ID)
        
        if not aggregated_data:
            logger.error("No data found to aggregate.")
            return

        # 4. Generate AI Report
        print("[AI] Generating Feedback...")
        feedback_text = report_gen.generate_feedback(aggregated_data)

        # 5. SAVE TO DATABASE (The New Step)
        print("\n[DB] Saving Report to Database...")
        
        # Data ko split karke bhejna hai
        nlp_data = aggregated_data.get("nlp_aggregate", {})
        cv_data = aggregated_data.get("cv_aggregate", {})

        success = db.save_final_report(
            session_id=TARGET_SESSION_ID,
            nlp_data=nlp_data,
            cv_data=cv_data,
            feedback_text=feedback_text
        )

        if success:
            print("\n" + "="*50)
            print("PIPELINE COMPLETED & SAVED SUCCESSFULLY")
            print("="*50)
        else:
            print("\n[ERROR] Report Generation worked, but Saving Failed.")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

#testing full analysis pipeline 
def run_pipeline_integration_test():
    print("\n" + "="*50)
    print("STARTING ANALYSIS PIPELINE INTEGRATION TEST")
    print("="*50)

    # --- CONFIGURATION ---
    # ⚠️ Yahan apni kisi bhi real MP4 video ka path dein
    # Agar file nahi mili to test ruk jayega.
    VIDEO_PATH = "artifacts/data_ingestion/test_video.mp4" 
    
    TEST_SESSION_ID = "pipeline_final_test_session_01"
    TEST_QUESTION_ID = "Q1_Test"

    # --- STEP 1: CHECK FILE EXISTENCE ---
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"[ERROR] Test Video not found at: {VIDEO_PATH}")
        logger.info("Please place a video file there or update VIDEO_PATH in the script.")
        return

    try:
        # --- STEP 2: INITIALIZE PIPELINE ---
        logger.info("[INIT] Initializing Analysis Pipeline...")
        pipeline = AnalysisPipeline()

        # --- STEP 3: RUN PIPELINE ---
        print(f"\n[PROCESSING] Processing Video: {VIDEO_PATH}...")
        print(f"Session ID: {TEST_SESSION_ID} | Question ID: {TEST_QUESTION_ID}")
        
        success = pipeline.run_pipeline(
            session_id=TEST_SESSION_ID,
            question_id=TEST_QUESTION_ID,
            video_path=VIDEO_PATH
        )

        # --- STEP 4: VERIFY IN DATABASE ---
        if success:
            print("\n" + "-"*30)
            print("[VERIFICATION] Pipeline returned Success. Checking Database...")
            
            # DB Connector alag se banayen taakay check kar sakein
            config = ConfigurationManager()
            db = DatabaseConnector(config.get_database_config())
            
            # Fetch specifically this chunk
            # (Assuming get_all_chunks returns a list, we filter or just get all)
            chunks = db.get_all_chunks(TEST_SESSION_ID)
            
            target_chunk = None
            for chunk in chunks:
                if chunk['question_id'] == TEST_QUESTION_ID:
                    target_chunk = chunk
                    break
            
            if target_chunk:
                print("\n✅ TEST PASSED: Data found in Database!")
                print(f"   > NLP Score: {target_chunk.get('phase1_score')}")
                print(f"   > CV Score: {target_chunk.get('cv_score')}")
                print(f"   > Transcript: {str(target_chunk.get('transcript'))[:50]}...")
            else:
                print("\n❌ TEST FAILED: Pipeline said success, but Data NOT found in DB.")
        else:
            print("\n❌ TEST FAILED: Pipeline execution returned False.")

    except Exception as e:
        logger.error(f"[CRASH] Test script failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    #test_configuration()
    #test_preprocessing()
    #test_cv_analyzer()
    #test_nlp_analyzer()
    #test_report_generator()
    #test_full_analysis_pipeline()
    #init_database()
    #saving_test()
    #test_retrieval()
    #run_db_aggregator_test()
    #run_generation()
    #run_saving_final_report()
    
