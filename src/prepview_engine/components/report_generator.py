from prepview_engine.utils.common import logger
from prepview_engine.config.configuration import ScoringConfig, DatabaseConfig
from prepview_engine.database.db_connector import DatabaseConnector
from prepview_engine.database.models import AnalysisReport
from typing import Dict, Any

class ReportGeneratorComponent:
    def __init__(self,
                 cv_results: Dict[str, Any],
                 nlp_results: Dict[str, Any],
                 session_id: str,
                 scoring_config: ScoringConfig,
                 db_connector: DatabaseConnector):
        """
        Initializes the Report Generator component.
        """
        self.cv_results = cv_results
        self.nlp_results = nlp_results
        self.session_id = session_id
        self.config = scoring_config
        self.db_connector = db_connector
        
        self.final_report = AnalysisReport(session_id=self.session_id)
        logger.info("ReportGeneratorComponent initialized.")

    def _run_scoring_engine(self):
        """
        Applies scoring rules based on thresholds.
        """
        logger.info("Running scoring engine...")
        
        # 1. Gaze Scoring (out of 10)
        gaze_center_pct = self.cv_results.get("gaze_analysis", {}).get("center", 0.0)
        if gaze_center_pct >= self.config.gaze_good_threshold:
            self.final_report.gaze_score = 10.0
            self.final_report.gaze_feedback = "Excellent eye contact! You stayed focused."
        elif gaze_center_pct >= self.config.gaze_avg_threshold:
            self.final_report.gaze_score = 7.0
            self.final_report.gaze_feedback = "Good eye contact, but try to look at the camera more."
        else:
            self.final_report.gaze_score = 3.0
            self.final_report.gaze_feedback = "You seemed distracted. Try to maintain focus on the camera."

        # 2. Filler Words Scoring (out of 10)
        filler_count = self.nlp_results.get("communication", {}).get("filler_word_count", 99)
        if filler_count <= self.config.filler_good_threshold:
            self.final_report.filler_word_score = 10.0
            self.final_report.filler_word_feedback = "Very clear and concise speaking."
        elif filler_count <= self.config.filler_avg_threshold:
            self.final_report.filler_word_score = 6.0
            self.final_report.filler_word_feedback = "Your answer was clear, but you used a few filler words."
        else:
            self.final_report.filler_word_score = 2.0
            self.final_report.filler_word_feedback = "Try to reduce filler words to sound more confident."
            
        # 3. Posture Scoring (Placeholder - aap params.yaml mai add kar saktay hain)
        posture_upright_pct = self.cv_results.get("posture_analysis", {}).get("upright", 0.0)
        if posture_upright_pct >= 70: # Example threshold
             self.final_report.posture_score = 10.0
             self.final_report.posture_feedback = "Great posture, you appeared confident."
        else:
             self.final_report.posture_score = 5.0
             self.final_report.posture_feedback = "Try to sit up straight to appear more engaged."

        # 4. Overall Score (Simple Average)
        scores = [s for s in [self.final_report.gaze_score, self.final_report.filler_word_score, self.final_report.posture_score] if s is not None]
        self.final_report.overall_score = sum(scores) / len(scores) if scores else 0.0
        
        logger.info(f"Scoring complete. Overall score: {self.final_report.overall_score}")

    def _populate_report_data(self):
        """Populates the AnalysisReport object with all raw data."""
        logger.info("Populating raw data into report object.")
        
        # NLP Data
        self.final_report.transcript = self.nlp_results.get("transcript")
        self.final_report.sentiment_label = self.nlp_results.get("sentiment", {}).get("label")
        self.final_report.sentiment_score = self.nlp_results.get("sentiment", {}).get("score")
        self.final_report.filler_word_count = self.nlp_results.get("communication", {}).get("filler_word_count")
        self.final_report.total_words = self.nlp_results.get("communication", {}).get("total_words")
        
        # CV Data
        self.final_report.total_frames = self.cv_results.get("cv_total_frames")
        self.final_report.gaze_analysis_percent = self.cv_results.get("gaze_analysis")
        self.final_report.posture_analysis_percent = self.cv_results.get("posture_analysis")

    def run(self):
        """
        Runs scoring and saves the final report to the database.
        """
        logger.info("--- Starting Report Generator Component ---")
        try:
            # 1. Populate raw data
            self._populate_report_data()
            
            # 2. Run scoring engine
            self._run_scoring_engine()
            
            # 3. Save to database
            logger.info(f"Saving report for session_id: {self.session_id} to database...")
            
            # Database session ko acces karnay kay liya hai
            session_gen = self.db_connector.get_session()
            session = next(session_gen)
            
            session.add(self.final_report) # Report object ko session mai add karain
            session.commit() # Database mai save karain
            session.refresh(self.final_report) # Saved object ka ID hasil karain
            
            logger.info(f"Report saved successfully with ID: {self.final_report.id}")
            logger.info("--- Finished Report Generator Component ---")
            
        except Exception as e:
            logger.error(f"Error during report generation or saving: {e}")
            raise
        finally:
            if 'session' in locals():
                session.close() # Session ko hamesha close karain