import os
from pathlib import Path
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import PreprocessingConfig # Ensure this import exists
from moviepy import VideoFileClip

class PreprocessingComponent:
    def __init__(self, config: PreprocessingConfig):
        """
        Initializes the component with configuration ONLY.
        Video path will be passed later during execution.
        """
        self.config = config
        logger.info("✅ PreprocessingComponent Initialized.")

    def extract_audio(self, video_path: Path) -> Path:
        """
        Extracts audio from the video file and saves it as a .wav file.
        """
        try:
            video_name = video_path.stem  # File name without extension
            
            # Ensure temp directory exists
            os.makedirs(self.config.temp_video_path, exist_ok=True)
            
            # Define output audio path
            audio_file_name = f"{video_name}_audio.wav"
            audio_file_path = Path(self.config.temp_video_path) / audio_file_name
            
            logger.info(f"🔉 Starting audio extraction for: {video_path.name}")
            
            # Load Video & Write Audio
            # Note: Putting inside 'with' block ensures it closes automatically
            with VideoFileClip(str(video_path)) as video_clip:
                video_clip.audio.write_audiofile(str(audio_file_path), codec='pcm_s16le', logger=None)
            
            logger.info(f"✅ Audio saved to: {audio_file_path}")
            return audio_file_path
            
        except Exception as e:
            logger.error(f"❌ Error during audio extraction: {e}")
            raise e

    def run(self, video_path_str: str) -> Path:
        """
        Runs the full preprocessing step on a specific video.
        
        Args:
            video_path_str (str): Path to the input video.
            
        Returns:
            Path: The path to the extracted audio file.
        """
        video_path = Path(video_path_str)
        
        # Validation
        if not video_path.exists():
            raise FileNotFoundError(f"Original video file not found at: {video_path}")

        logger.info(f"--- Processing Video: {video_path.name} ---")
        
        # 1. Extract Audio
        audio_path = self.extract_audio(video_path)
        
        # Return only audio path (Video path caller ke paas already hai)
        return audio_path