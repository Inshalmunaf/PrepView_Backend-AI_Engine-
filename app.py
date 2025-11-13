from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import uvicorn
import os
import uuid
from pathlib import Path
from prepview_engine.pipeline.analysis_pipeline import AnalysisPipeline
from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.utils.common import logger



app = FastAPI(
    title="PrepView AI Analysis Engine",
    description="API for analyzing interview video responses."
)

# --- Configuration Loading ---
# Ham yahan configuration ko pehlay hi load kar rahay hain
try:
    config_manager = ConfigurationManager()
    preprocessing_config = config_manager.get_preprocessing_config()
    os.makedirs(preprocessing_config.temp_video_path, exist_ok=True)
    logger.info(f"Temporary storage directory ensured at: {preprocessing_config.temp_video_path}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    preprocessing_config = None 


# --- Helper Function for Pipeline ---
def run_analysis_in_background(video_path: str, session_id: str):
    """
    This is the function that BackgroundTasks will run.
    It initializes and triggers the main analysis pipeline.
    """
    logger.info(f"Background task started for session: {session_id}, file: {video_path}")
    try:
        pipeline = AnalysisPipeline(video_path=video_path, session_id=session_id)
        pipeline.run()
        logger.info(f"Background task finished for session: {session_id}")
    except Exception as e:
        logger.error(f"Error in background pipeline for {session_id}: {e}")
        pass

# --- API Endpoint ---
@app.post("/analyze-response/")
async def analyze_response(
    background_tasks: BackgroundTasks,
    session_id: str, 
    video_file: UploadFile = File(...)
):
    """
    Analyzes a single video response for a given session.
    
    - **session_id**: A unique ID for the entire interview session.
    - **video_file**: The uploaded .webm or .mp4 video file.
    """

    if not preprocessing_config:
        return {"status": "error", "message": "Server configuration error."}, 500

    try:
        # 1. Create a unique filename to avoid conflicts
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
        return {"status": "error", "message": f"An error occurred: {e}"}, 500

# --- Server Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)