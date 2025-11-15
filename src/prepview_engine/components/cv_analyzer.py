import cv2
import mediapipe as mp
from pathlib import Path
from prepview_engine.utils.common import logger
from collections import Counter

class CVAnalyzerComponent:
    def __init__(self, video_path: Path):
        """
        Initializes the CV Analyzer component.
        
        Args:
            video_path (Path): The path to the video file to be analyzed.
        """
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {self.video_path}")
            raise IOError(f"Failed to open video file: {self.video_path}")

        # MediaPipe models ko initialize karna
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Aankhon k liye zaroori
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info(f"CVAnalyzerComponent initialized for video: {video_path.name}")

    def _analyze_gaze(self, frame, face_landmarks) -> str:
        """Helper function to analyze gaze direction."""
        # Ye aik basic implementation hai.
        # Ham left aur right eye landmarks ko dekhtay hain.
        # 
        
        # Landmark indices for eyes
        LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        h, w, _ = frame.shape
        
        try:
            # Aankh k landmarks ka center nikalna
            left_eye_center = (
                sum([face_landmarks.landmark[i].x for i in LEFT_EYE_INDICES]) / len(LEFT_EYE_INDICES),
                sum([face_landmarks.landmark[i].y for i in LEFT_EYE_INDICES]) / len(LEFT_EYE_INDICES)
            )
            right_eye_center = (
                sum([face_landmarks.landmark[i].x for i in RIGHT_EYE_INDICES]) / len(RIGHT_EYE_INDICES),
                sum([face_landmarks.landmark[i].y for i in RIGHT_EYE_INDICES]) / len(RIGHT_EYE_INDICES)
            )

            # Iris landmarks (refined landmarks zaroori hain)
            left_iris = face_landmarks.landmark[473]
            right_iris = face_landmarks.landmark[468]

            # Gaze ratio nikalna (horizontal)
            # Ye check karta hai k iris aankh k center say kitna left/right hai
            # Note: Ye bohot basic hai aur lighting/head angle say affect ho sakta hai
            
            # Simple assumption: Agar user ka chehra bilkul seedha hai,
            # toh dono aankhon ka center x-coordinate iris k x-coordinate k qareeb hona chahiyay.
            
            # Aik behtar tareeqa 'eye aspect ratio' ki tarah hai, 
            # lekin gaze k liye implementation complex hai.
            # Ham abhi k liye simplified version use kartay hain.
            
            # Simplified: Check if user is looking away (extreme left/right)
            # Hum iris aur eye corners (e.g., 362, 263) ka distance check kar saktay hain
            
            # For simplicity, we'll just check if face is detected.
            # A full gaze implementation is very complex.
            # We'll return 'center' as a placeholder for now.
            # TODO: Implement a more robust gaze ratio
            
            return "center" # Placeholder
        
        except Exception:
            return "unknown" # Agar landmarks detect na hon

    def _analyze_posture(self, pose_landmarks) -> str:
        """Helper function to analyze posture (slouching)."""
        try:
            landmarks = pose_landmarks.landmark
            
            # Shoulder landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Ear landmarks
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            
            # Check visibility
            if not (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and 
                    left_ear.visibility > 0.5 and right_ear.visibility > 0.5):
                return "unknown" # Agar landmarks saaf nahi hain

            # Slouching detection logic:
            # Agar kaan (ears) shoulders say bohot aagay hon (y-axis par)
            # ya shoulders ka mid-point bohot neechay ho.
            
            # Simple Logic: Check if ear is 'in front of' the shoulder mid-point
            # (y-coordinate of ear is significantly lower than y-coordinate of shoulder)
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            ear_mid_y = (left_ear.y + right_ear.y) / 2
            
            # Threshold (ye adjust karna paray ga)
            slouch_threshold = 0.1 # Example value
            
            if (shoulder_mid_y - ear_mid_y) < slouch_threshold:
                return "slouched"
            else:
                return "upright"
                
        except Exception:
            return "unknown" # Agar pose detect na ho

    def run(self) -> dict:
        """
        Runs the full CV analysis on the video file.
        
        Returns:
            dict: A dictionary containing aggregated CV analysis results.
        """
        logger.info("--- Starting CV Analysis Component ---")
        
        frame_count = 0
        gaze_results = []
        posture_results = []
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break # Video khatam ho gai
                
            frame_count += 1
            
            # Performance k liye frame ko chota kar saktay hain
            # frame = cv2.resize(frame, (640, 480)) 
            
            # Frame ko RGB mai convert karna (MediaPipe k liye zaroori)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Face Mesh Analysis (Gaze)
            face_results = self.face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    gaze = self._analyze_gaze(frame, face_landmarks)
                    gaze_results.append(gaze)
            else:
                gaze_results.append("no_face_detected")

            # 2. Pose Analysis (Posture)
            pose_results = self.pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                posture = self._analyze_posture(pose_results.pose_landmarks)
                posture_results.append(posture)
            else:
                posture_results.append("no_pose_detected")
        
        # Release resources
        self.cap.release()
        self.face_mesh.close()
        self.pose.close()
        
        logger.info(f"CV Analysis finished. Processed {frame_count} frames.")
        
        # 3. Aggregate results
        if frame_count == 0:
            logger.warning("Video file was empty or unreadable.")
            return {}

        gaze_counts = Counter(gaze_results)
        posture_counts = Counter(posture_results)
        
        # Percentages mai convert karna
        final_results = {
            "cv_total_frames": frame_count,
            "gaze_analysis": {k: (v / frame_count) * 100 for k, v in gaze_counts.items()},
            "posture_analysis": {k: (v / frame_count) * 100 for k, v in posture_counts.items()}
        }
        
        logger.info(f"CV Analysis results: {final_results}")
        logger.info("--- Finished CV Analysis Component ---")
        
        return final_results