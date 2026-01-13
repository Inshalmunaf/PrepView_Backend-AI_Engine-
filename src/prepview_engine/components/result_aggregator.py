import numpy as np
from typing import List, Dict, Any
from collections import Counter
from prepview_engine.utils.common import logger

class ResultAggregator:
    def __init__(self):
        logger.info("✅ Result Aggregator Initialized.")

    def _sanitize(self, value):
        """
        Converts Numpy types to standard Python types.
        (Database or JSON serialization will crash without this).
        """
        if isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, (np.floating, float)):
            return round(float(value), 2)
        elif isinstance(value, (np.ndarray, list)):
            return [self._sanitize(x) for x in value]
        elif isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        return value

    def _safe_mean(self, values: List[float]) -> float:
        """Calculates average safely (handles empty lists to avoid ZeroDivisionError)."""
        if not values:
            return 0.0
        return round(float(np.mean(values)), 2)

    def aggregate_session(self, chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines multiple InterviewChunk data into a single Master Summary.
        
        Args:
            chunks_data: List of dictionaries representing InterviewChunk rows.
                         Expected keys: 'wpm', 'eye_contact_pct', 'transcript', etc.
        
        Returns:
            Dict: Aggregated Summary ready for FinalReport.summary_metrics
        """
        if not chunks_data:
            logger.warning("⚠️ No chunks provided for aggregation.")
            return {}

        logger.info(f"📊 Aggregating data from {len(chunks_data)} chunks...")

        # --- 1. Storage Lists ---
        transcripts = []
        
        # NLP Lists
        scores = []
        wpms = []
        filler_rates = []
        prosodic_scores = []
        lexical_richness_list = [] # Ye JSON ke andar se nikalna padega
        
        # CV Lists
        eye_contacts = []
        nervousness_scores = []
        dominant_moods = []

        # --- 2. Iterate & Extract ---
        for chunk in chunks_data:
            # A. Direct Columns (Fast Access)
            if chunk.get("transcript"):
                transcripts.append(chunk["transcript"])
            
            if chunk.get("phase1_score") is not None:
                scores.append(chunk["phase1_score"])
            
            if chunk.get("wpm") is not None:
                wpms.append(chunk["wpm"])
                
            if chunk.get("filler_rate") is not None:
                filler_rates.append(chunk["filler_rate"])
                
            if chunk.get("prosodic_confidence") is not None:
                prosodic_scores.append(chunk["prosodic_confidence"])

            if chunk.get("eye_contact_pct") is not None:
                eye_contacts.append(chunk["eye_contact_pct"])

            if chunk.get("nervousness_pct") is not None:
                nervousness_scores.append(chunk["nervousness_pct"])
                
            if chunk.get("dominant_mood"):
                dominant_moods.append(chunk["dominant_mood"])

            # B. Deep Dive into JSON (agar koi extra info chahiye jo column mai nahi hai)
            # Example: Lexical Richness column mai nahi tha, toh JSON se nikal rahe hain
            nlp_json = chunk.get("nlp_data_json", {})
            if nlp_json and "linguistic_metrics" in nlp_json:
                richness = nlp_json["linguistic_metrics"].get("lexical_richness", 0)
                lexical_richness_list.append(richness)

        # --- 3. Calculation & Logic ---

        # Text Merging
        full_transcript = " ".join(transcripts)

        # Dominant Mood (Most Frequent)
        final_mood = "neutral"
        if dominant_moods:
            final_mood = Counter(dominant_moods).most_common(1)[0][0]

        # Construct Summary Dictionary
        summary = {
            "session_meta": {
                "total_questions": len(chunks_data),
                "aggregated_at": str(np.datetime64('now'))
            },
            
            # Key Averages (For Columns)
            "final_score": self._safe_mean(scores),
            
            "nlp_aggregate": {
                "avg_wpm": self._safe_mean(wpms),
                "avg_filler_rate": self._safe_mean(filler_rates),
                "avg_prosodic_confidence": self._safe_mean(prosodic_scores),
                "avg_lexical_richness": self._safe_mean(lexical_richness_list),
                "transcript_full_length": len(full_transcript),
                # LLM Context (First 1000 chars to save tokens)
                "transcript_sample": full_transcript[:1200] 
            },
            
            "cv_aggregate": {
                "avg_eye_contact": self._safe_mean(eye_contacts),
                "avg_nervousness": self._safe_mean(nervousness_scores),
                "dominant_mood": final_mood,
                "mood_consistency": self._calculate_consistency(dominant_moods)
            }
        }

        # --- 4. Final Sanitization ---
        return self._sanitize(summary)

    def _calculate_consistency(self, moods: List[str]) -> str:
        """Helper to check if mood was stable."""
        if not moods: return "Unknown"
        most_common_count = Counter(moods).most_common(1)[0][1]
        consistency_ratio = most_common_count / len(moods)
        
        if consistency_ratio > 0.8: return "Highly Consistent"
        elif consistency_ratio > 0.5: return "Variable"
        else: return "Fluctuating"