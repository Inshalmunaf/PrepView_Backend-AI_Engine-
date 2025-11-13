import os
from pathlib import Path
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import PreprocessingConfig
from moviepy import VideoFileClip

class PreprocessingComponent:
    def __init__(self, original_video_path: str, config: PreprocessingConfig):
        """
        Initializes the preprocessing component.
        
        Args:
            original_video_path (str): The path to the uploaded video file.
            config (PreprocessingConfig): The configuration object for preprocessing.
        """
        self.original_video_path = Path(original_video_path)
        self.config = config
        
        # Ensure the paths are valid
        if not self.original_video_path.exists():
            logger.error(f"Original video file not found at: {original_video_path}")
            raise FileNotFoundError(f"Original video file not found at: {original_video_path}")
            
        logger.info(f"PreprocessingComponent initialized for video: {self.original_video_path.name}")

    def extract_audio(self) -> Path:
        """
        Extracts audio from the video file and saves it as a .wav file.
        
        Returns:
            Path: The path to the saved audio file.
        """
        try:
            video_name = self.original_video_path.stem # File ka naam bina extension
            # Audio file ko usi temp folder mai save karain gay
            audio_file_name = f"{video_name}_audio.wav"
            audio_file_path = self.config.temp_video_path / audio_file_name
            
            logger.info(f"Starting audio extraction for: {self.original_video_path.name}")
            
            # Load the video file
            video_clip = VideoFileClip(str(self.original_video_path))
            
            # Write the audio file
            video_clip.audio.write_audiofile(str(audio_file_path), codec='pcm_s16le', logger=None)
            
            video_clip.close()
            
            logger.info(f"Audio extracted successfully and saved to: {audio_file_path}")
            return audio_file_path
            
        except Exception as e:
            logger.error(f"Error during audio extraction: {e}")
            raise

    def run(self) -> (Path, Path):
        """
        Runs the full preprocessing step.
        
        Returns:
            tuple (Path, Path): (path_to_video, path_to_extracted_audio)
        """
        logger.info("--- Starting Preprocessing Component ---")
        
        # 1. Extract Audio
        audio_path = self.extract_audio()
        
        # 2. Return paths for next components
        video_path = self.original_video_path
        
        logger.info("--- Finished Preprocessing Component ---")
        return video_path, audio_path