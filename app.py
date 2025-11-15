from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
import uvicorn
import os
import uuid
from pathlib import Path
from prepview_engine.utils.common import logger
from prepview_engine.pipeline.analysis_pipeline import AnalysisPipeline
from prepview_engine.config.configuration import ConfigurationManager

app = FastAPI(
    title="PrepView AI Analysis Engine",
    description="API for analyzing interview video responses."
)

# --- Configuration Loading ---
try:
    config_manager = ConfigurationManager()
    preprocessing_config = config_manager.get_preprocessing_config()
    # Ensure temp directory exists
    os.makedirs(preprocessing_config.temp_video_path, exist_ok=True)
    logger.info(f"Temporary storage directory ensured at: {preprocessing_config.temp_video_path}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    preprocessing_config = None


# --- Background Task Helper ---
def run_analysis_in_background(video_path: str, session_id: str):
    """
    This is the function that BackgroundTasks will run.
    It initializes and triggers the main analysis pipeline
    and cleans up the temp video file afterwards.
    """
    logger.info(f"Background task started for session: {session_id}, file: {video_path}")
    try:
        # 1. Initialize aur run
        pipeline = AnalysisPipeline(video_path=video_path, session_id=session_id)
        pipeline.run()
        
        logger.info(f"Background task (pipeline) finished for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Error in background pipeline for {session_id}: {e}")
        # Yahan aap error ko database mai log kar saktay hain
    
    finally:
        # 2. Cleanup
        # Pipeline audio file ko khud delete kar daiti hai.
        # Ab ham yahan original video file ko delete karain gay.
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up temp video file: {video_path}")
        except Exception as e:
            logger.warning(f"Warning: Could not delete temp video file {video_path}: {e}")


# --- API Endpoint ---
@app.post("/analyze-response/")
async def analyze_response(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...), # Form(...) use karain taakay video k sath aa sakay
    video_file: UploadFile = File(...)
):
    """
    Analyzes a single video response for a given session.
    
    - **session_id**: A unique ID for the entire interview session.
    - **video_file**: The uploaded .webm or .mp4 video file.
    """
    if not preprocessing_config:
        logger.error("Server configuration error. Preprocessing config not loaded.")
        return {"status": "error", "message": "Server configuration error."}, 500

    try:
        # 1. Create a unique filename
        file_extension = Path(video_file.filename).suffix
        unique_filename = f"{session_id}_{uuid.uuid4()}{file_extension}"
        temp_video_path = os.path.join(preprocessing_config.temp_video_path, unique_filename)
        
        # 2. Save the uploaded video to the temp storage
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video_file.read())
        
        logger.info(f"File saved temporarily at: {temp_video_path}")

        # 3. Add the heavy analysis task to the background
        background_tasks.add_task(
            run_analysis_in_background, 
            video_path=temp_video_path, 
            session_id=session_id
        )

        # 4. Return an immediate response to the frontend
        return {
            "status": "processing_started",
            "message": "Your response has been received and analysis has started.",
            "session_id": session_id,
            "filename": unique_filename
        }
    
    except Exception as e:
        logger.error(f"Error in /analyze-response/ endpoint: {e}")
        return {"status": "error", "message": f"An error occurred: {e}"}, 500

# --- Report Retrieval Endpoint ---
# Ye endpoint frontend ko report fetch karnay k liye chahiyay hoga
# (Isko ham baad mai implement kar saktay hain)
@app.get("/get-report/{session_id}")
async def get_report(session_id: str):
    """
    Retrieves all analysis reports for a given session_id.
    (NOTE: This is a placeholder. Needs implementation.)
    """
    logger.info(f"Received request for report: {session_id}")
    # TODO:
    # 1. DatabaseConnector ko initialize karain
    # 2. Database say session_id ki bunyad par reports fetch karain
    # 3. JSON response return karain
    return {"message": "Not implemented yet. Fetching data for session:", "session_id": session_id}


# --- Server Runner ---
if __name__ == "__main__":
    logger.info("Starting PrepView AI Engine API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)