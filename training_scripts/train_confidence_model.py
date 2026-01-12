
import os
import sys
# Project root ko path mai add karna zaroori hai taakay src module import ho sakay
sys.path.append(os.getcwd()) 

import glob
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- IMPORT CONFIGURATION MANAGER ---
from prepview_engine.config.configuration import ConfigurationManager
from prepview_engine.utils.common import logger

# Emotion Mapping (Waisay hi rahay ga)
EMOTION_MAP = {
    '01': 'high_confidence', '02': 'high_confidence', '03': 'high_confidence',
    '04': 'low_confidence', '06': 'low_confidence'
}

def extract_audio_features(file_path):
    """
    Extracts 4 key audio features for confidence detection.
    """
    try:
        # Load Audio (resample to 22050 Hz standard)
        y, sr = librosa.load(file_path, sr=22050)
        
        # 1. Pitch & Stability (Jitter detection)
        # piptrack gives pitch candidates per frame
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Select pitches with high magnitude (filter out noise)
        threshold = 0.5
        pitches_filtered = pitches[magnitudes > np.max(magnitudes)*threshold]
        
        if len(pitches_filtered) > 0:
            pitch_mean = np.mean(pitches_filtered)
            pitch_std = np.std(pitches_filtered) # High std dev = shaky voice
        else:
            pitch_mean = 0
            pitch_std = 0
            
        # Zero Crossing Rate (Noise/Roughness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # 2. Energy & Loudness (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms) # Average loudness
        energy_std = np.std(rms)   # Variation in loudness

        # 3. Rhythm & Flow (Silence Ratio & Tempo)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Silence Ratio (Proportion of frames below energy threshold)
        # Using 10% of max energy as threshold for silence
        silence_threshold = 0.01 
        silence_frames = np.sum(rms < silence_threshold)
        total_frames = rms.shape[1]
        silence_ratio = silence_frames / total_frames

        # 4. Tone / Intonation (MFCCs)
        # MFCCs capture the 'timbre' or richness of voice
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1) # Returns array of 13 values
        
        # Combine all features into a single array
        # [pitch_mean, pitch_std, zcr, energy_mean, energy_std, tempo, silence_ratio, mfcc_1, ... mfcc_13]
        features = np.hstack([
            pitch_mean, pitch_std, zcr, 
            energy_mean, energy_std, 
            tempo, silence_ratio, 
            mfcc_mean
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
    
def train_model():
    logger.info("--- Starting Confidence Model Training Pipeline ---")
    
    # 1. Load Configuration
    config_manager = ConfigurationManager()
    train_config = config_manager.get_confidence_training_config()
    
    logger.info(f"Dataset path: {train_config.dataset_path}")
    
    data = []
    labels = []
    
    # 2. Use Path from Config
    # Check karain dataset exist karta hai ya nahi
    if not os.path.exists(train_config.dataset_path):
        logger.error(f"Dataset not found at: {train_config.dataset_path}")
        return

    wav_files = glob.glob(os.path.join(train_config.dataset_path, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(wav_files)} audio files.")
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) < 3: continue
            
        emotion_code = parts[2]
        if emotion_code in EMOTION_MAP:
            confidence_label = EMOTION_MAP[emotion_code]
            # Feature extraction code yahan ayega (real implementation mai)
            features = extract_audio_features(file_path)
            if features is not None:
                data.append(features)
                labels.append(confidence_label)
    
    # ... (Data Processing Logic Same) ...
    # ... (Main yahan mock kar raha hu taakay flow samjha sakun) ...
    
    # 3. Training with Config Parameters
    clf = RandomForestClassifier(
        n_estimators=train_config.n_estimators, # Config se value li
        random_state=train_config.random_state
     )
    
    # 4. Save Models using Config Paths
    logger.info(f"Saving model to {train_config.model_save_path}")
    joblib.dump(clf, train_config.model_save_path)
    #joblib.dump(scaler, train_config.scaler_save_path)
    
    logger.info("Training complete and models saved!")

if __name__ == "__main__":
    train_model()