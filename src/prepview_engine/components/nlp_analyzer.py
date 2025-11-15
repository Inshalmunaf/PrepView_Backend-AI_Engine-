from pathlib import Path
from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import NLPConfig
from transformers import pipeline
import spacy
import torch

class NLPAnalyzerComponent:
    def __init__(self, audio_path: Path, config: NLPConfig):
        """
        Initializes the NLP Analyzer component.
        
        Args:
            audio_path (Path): The path to the extracted audio file (.wav).
            config (NLPConfig): The configuration object for NLP models.
        """
        self.audio_path = str(audio_path)
        self.config = config
        
        # Check if GPU is available
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"NLP models will run on device: {self.device}")

        try:
            # 1. Speech-to-Text (Whisper)
            logger.info(f"Loading STT model: {self.config.stt_model_name}")
            self.stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config.stt_model_name,
                device=self.device
            )
            
            # 2. Sentiment Analysis
            logger.info(f"Loading Sentiment model: {self.config.sentiment_model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model_name,
                device=self.device
            )
            
            # 3. SpaCy for filler words
            logger.info("Loading spaCy model: en_core_web_sm")
            self.nlp_spacy = spacy.load("en_core_web_sm")
            
            # Common filler words (aap isay params.yaml mai bhi daal saktay hain)
            self.FILLER_WORDS = set([
                'um', 'umm', 'uh', 'uhh', 'ah', 'ahh', 'er', 'err', 
                'like', 'so', 'you know', 'basically', 'actually'
            ])

            logger.info("NLPAnalyzerComponent initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            raise

    def transcribe_audio(self) -> str:
        """Transcribes the audio file to text using Whisper."""
        try:
            logger.info(f"Starting transcription for: {self.audio_path}")
            # chunk_length_s=30 STT ko long audio par behtar chalata hai
            result = self.stt_pipeline(self.audio_path, chunk_length_s=30, return_timestamps=False)
            transcript = result["text"].strip()
            logger.info(f"Transcription successful. Transcript: {transcript[:50]}...")
            return transcript
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return "" # Return empty string on failure

    def analyze_sentiment(self, text: str) -> dict:
        """Analyzes the sentiment of the given text."""
        if not text:
            return {"label": "NEUTRAL", "score": 0.0}
        try:
            result = self.sentiment_pipeline(text)
            logger.info(f"Sentiment analysis result: {result[0]}")
            return result[0] # Returns {'label': 'POSITIVE', 'score': 0.99}
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            return {"label": "ERROR", "score": 0.0}

    def analyze_communication(self, text: str) -> dict:
        """Analyzes communication clarity (filler words, WPM)."""
        if not text:
            return {"filler_word_count": 0, "total_words": 0}
            
        doc = self.nlp_spacy(text.lower())
        filler_count = 0
        total_words = len(doc)
        
        # Filler words count
        for token in doc:
            if token.text in self.FILLER_WORDS:
                filler_count += 1
        
        # Aap yahan Words Per Minute (WPM) bhi calculate kar saktay hain
        # agar aap audio ki duration (preprocessing say) yahan pass karain
        
        result = {
            "filler_word_count": filler_count,
            "total_words": total_words
        }
        logger.info(f"Communication analysis result: {result}")
        return result

    def run(self) -> dict:
        """
        Runs the full NLP analysis on the audio file.
        
        Returns:
            dict: A dictionary containing aggregated NLP analysis results.
        """
        logger.info("--- Starting NLP Analysis Component ---")
        
        # 1. Transcribe
        transcript = self.transcribe_audio()
        
        if not transcript:
            logger.warning("Transcription failed or audio was silent. Skipping further NLP analysis.")
            return {"transcript": "", "sentiment": {}, "communication": {}}
            
        # 2. Analyze Sentiment
        sentiment_result = self.analyze_sentiment(transcript)
        
        # 3. Analyze Communication
        communication_result = self.analyze_communication(transcript)
        
        final_results = {
            "transcript": transcript,
            "sentiment": sentiment_result,
            "communication": communication_result
        }
        
        logger.info("--- Finished NLP Analysis Component ---")
        return final_results