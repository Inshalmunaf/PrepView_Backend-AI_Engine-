import time
import requests  # <--- Make sure ye imported ho (pip install requests)
import json
from typing import Dict, Any
from prepview_engine.utils.common import logger
from prepview_engine.entity import ReportConfig
from prepview_engine.components.result_aggregator import ResultAggregator
from prepview_engine.components.database_connector import DatabaseConnector

class ReportGeneratorComponent:
    def __init__(self, config: ReportConfig, db_connector: DatabaseConnector):
        self.config = config
        self.db = db_connector
        self.aggregator = ResultAggregator()
        logger.info(f"✅ Report Generator Initialized (Provider: {self.config.llm_provider})")

    # ==========================================================
    # 🦙 OLLAMA INTEGRATION LOGIC
    # ==========================================================
    def _generate_llm_feedback(self, stats: Dict) -> str:
        """
        Sends aggregated metrics to Local Ollama instance and gets feedback.
        """
        try:
            logger.info(f"🤖 Calling Ollama ({self.config.model_name}) for Feedback...")
            
            # 1. Prepare Data
            nlp = stats.get("nlp_aggregate", {})
            cv = stats.get("cv_aggregate", {})
            
            # 2. Format Prompt
            user_msg = self.config.user_prompt_template.format(
                avg_wpm=nlp.get("avg_wpm", 0),
                avg_pause_ratio=round(nlp.get("avg_pause_ratio", 0) * 100, 1),
                avg_filler_rate=nlp.get("avg_filler_rate", 0),
                avg_prosodic_confidence=nlp.get("avg_prosodic_confidence", 0),
                avg_eye_contact=cv.get("avg_eye_contact", 0),
                dominant_mood=cv.get("dominant_mood", "neutral"),
                avg_nervousness=cv.get("avg_nervousness", 0),
                transcript_snippet=nlp.get("transcript_sample", "")[:800]
            )

            # Combine System + User Prompt for Ollama
            full_prompt = f"System: {self.config.system_prompt}\n\nUser: {user_msg}"

            # 3. Call Ollama API
            payload = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,  # Stream False rakhein taakay pura text aik sath mile
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }

            response = requests.post(self.config.base_url, json=payload)
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                logger.info("✅ AI Feedback Generated Successfully.")
                return result_text
            else:
                logger.error(f"Ollama Error {response.status_code}: {response.text}")
                return "AI Feedback unavailable (Model Error)."

        except requests.exceptions.ConnectionError:
            logger.error("❌ Could not connect to Ollama! Is 'ollama serve' running?")
            return "Error: Local AI is offline."
            
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return "AI Feedback generation failed due to technical error."

    # ==========================================================
    # 🚀 MAIN RUNNER (Logic Same Rahegi)
    # ==========================================================
    def run(self, session_id: str, user_id: int, question_id: str, 
            cv_result: Dict, nlp_result: Dict, video_path: str, is_last_chunk: bool):
        
        # ... (Ye code bilkul pichlay reply wala hi rahega) ...
        # ... (Sanitization, DB Save, Aggregation Logic) ...
        
        # Sirf DB Save aur Conditional logic same rahegi
        
        # 1. Sanitize & Extract Data
        clean_cv = self.aggregator._sanitize(cv_result)
        clean_nlp = self.aggregator._sanitize(nlp_result)
        
        transcript = clean_nlp.get("transcript", "")
        speech = clean_nlp.get("speech_metrics", {})

        # 2. Save Chunk
        try:
            self.db.save_chunk_result(
                session_id=session_id,
                question_id=question_id,
                video_path=video_path,
                cv_json=clean_cv,
                nlp_json=clean_nlp,
                transcript=transcript,
                wpm=speech.get("speech_rate_wpm", 0),
                phase1_score=clean_nlp.get("phase1_quality_score", 0),
                filler_rate=speech.get("filler_rate", 0),
                prosodic_confidence=clean_nlp.get("prosodic_confidence", 0),
                eye_contact_pct=clean_cv.get("eye_gaze", {}).get("eye_contact_percentage", 0),
                nervousness_pct=clean_cv.get("facial_expression", {}).get("nervousness_analysis", {}).get("total_concerned_percentage", 0),
                dominant_mood=clean_cv.get("facial_expression", {}).get("dominant_mood", "neutral")
            )
        except Exception as e:
            logger.error(f"Failed to save chunk: {e}")

        # 3. Check Flow
        if not is_last_chunk:
            return {"status": "partial_saved", "is_final_report": False}
        
        else:
            logger.info("🛑 Last chunk detected. Starting Report Generation...")
            
            # A. Fetch History
            all_chunks = self.db.get_all_chunks(session_id)
            
            # B. Aggregate
            summary = self.aggregator.aggregate_session(all_chunks)
            
            # C. AI Feedback (Now using OLLAMA)
            ai_feedback = self._generate_llm_feedback(summary)
            
            # D. Save Final Report
            self.db.save_final_report(
                session_id=session_id, 
                user_id=user_id, 
                avg_score=summary.get("final_score", 0),
                avg_wpm=summary["nlp_aggregate"].get("avg_wpm", 0),
                avg_eye=summary["cv_aggregate"].get("avg_eye_contact", 0),
                avg_prosodic_confidence=summary["nlp_aggregate"].get("avg_prosodic_confidence", 0),
                full_summary=summary, 
                feedback=ai_feedback
            )
            
            return {
                "status": "completed",
                "is_final_report": True,
                "summary": summary,
                "feedback": ai_feedback
            }