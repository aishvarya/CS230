import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from collections import deque
import logging
import os
from gtts import gTTS
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File and directory configurations
BASE_DIR = '/home/ubuntu/v2/yoga'
MODEL_PREFIX = 'yoga_model_top10_20241130_095719'
MODEL_FILE = os.path.join(BASE_DIR, f'models/{MODEL_PREFIX}.keras')
SCALER_FILE = os.path.join(BASE_DIR, f'models/{MODEL_PREFIX}_scaler.pkl')
ENCODER_FILE = os.path.join(BASE_DIR, f'models/{MODEL_PREFIX}_encoder.pkl')

# Video configurations
VIDEO_FILE = os.path.join(BASE_DIR, 'videos/sy-6.mp4')  
TEMP_VIDEO = os.path.join(BASE_DIR, 'videos/temp_output.mp4')  
FINAL_OUTPUT = os.path.join(BASE_DIR, 'videos/final_output.mp4')  

# Audio configuration
AUDIO_DIR = os.path.join(BASE_DIR, 'audio_feedback')
os.makedirs(AUDIO_DIR, exist_ok=True)

# Create videos directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'videos'), exist_ok=True)


# Constants
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7
POSE_STABILITY_THRESHOLD = 5
FEEDBACK_COOLDOWN = 5  # seconds

class LivePoseDetector:
    def __init__(self, model_path, scaler_path, encoder_path):
        # Load ML components
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize state tracking
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.pose_history = deque(maxlen=POSE_STABILITY_THRESHOLD)
        self.feedback_history = deque(maxlen=30)
        self.last_audio_time = 0
        self.current_pose = None
        self.current_feedback = None
        self.debug = False
        
        # Supported poses
        self.supported_poses = [
            "balancing-table", "tree", "downward-dog", "staff",
            "shoulder-pressing", "reclining-hero", "cockerel",  
            "frog", "reclining-cobbler", "wind-relieving"
        ]

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360 - angle

    def get_body_orientation(self, landmarks):
        """Determine if body is standing, inverted, or horizontal"""
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        head_y = landmarks[0].y
        feet_y = (landmarks[27].y + landmarks[28].y) / 2
        
        is_inverted = hip_y > head_y and hip_y > feet_y
        is_standing = abs(head_y - feet_y) > 0.5 and abs(landmarks[11].y - landmarks[12].y) < 0.1
        is_horizontal = abs(hip_y - feet_y) < 0.2
        
        return "inverted" if is_inverted else "standing" if is_standing else "horizontal"

    def calculate_pose_features(self, landmarks):
        """Extract geometric features from pose landmarks"""
        hip_center_y = (landmarks[23].y + landmarks[24].y) / 2
        hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
        shoulder_center_y = (landmarks[11].y + landmarks[12].y) / 2
        
        features = []
        for idx, landmark in enumerate(landmarks):
            features.append({
                'x': landmark.x - hip_center_x,
                'y': landmark.y - hip_center_y,
                'z': landmark.z,
                'landmark_index': idx,
                'sequence_id': 0,
                'height_ratio': landmark.y / hip_center_y
            })
        
        # Calculate additional pose features
        spine_angle = np.degrees(np.arctan2(shoulder_center_y - hip_center_y, 
                                          landmarks[12].x - landmarks[11].x))
        head_hip_ratio = landmarks[0].y / hip_center_y
        
        features = pd.DataFrame(features)
        features['spine_angle'] = spine_angle
        features['head_hip_ratio'] = head_hip_ratio
        
        return features

    def is_balancing_table(self, landmarks):
        """Detect balancing table pose"""
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        
        trunk_level = abs(hip_y - shoulder_y) < 0.15
        supporting_leg_grounded = max(landmarks[27].y, landmarks[28].y) > 0.7
        leg_lifted = abs(landmarks[27].y - landmarks[28].y) > 0.1
        hands_grounded = max(landmarks[15].y, landmarks[16].y) > 0.7
        
        return trunk_level and supporting_leg_grounded and leg_lifted and hands_grounded

    def is_tree(self, landmarks):
        """Detect tree pose"""
        if self.get_body_orientation(landmarks) != "standing":
            return False
        
        ankle_diff = abs(landmarks[27].y - landmarks[28].y)
        spine_vertical = abs(landmarks[0].x - landmarks[24].x) < 0.2
        hip_level = abs(landmarks[23].y - landmarks[24].y) < 0.1
        
        if self.debug:
            print(f"Tree Pose Check:")
            print(f"Ankle difference: {ankle_diff:.2f}")
            print(f"Spine vertical: {spine_vertical}")
            print(f"Hip level: {hip_level}")
        
        return ankle_diff > 0.15 and spine_vertical and hip_level

    def is_downward_dog(self, landmarks):
        """Detect downward dog pose"""
        if self.get_body_orientation(landmarks) != "inverted":
            return False
        
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        head_y = landmarks[0].y
        arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        
        if self.debug:
            print(f"Downward Dog Check:")
            print(f"Hip height: {abs(hip_y - head_y):.2f}")
            print(f"Arm angle: {arm_angle:.2f}")
            print(f"Leg angle: {leg_angle:.2f}")
        
        return abs(hip_y - head_y) > 0.2 and arm_angle > 160 and leg_angle > 160

    def is_staff(self, landmarks):
        """Detect staff pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        spine_angle = abs(90 - self.calculate_angle(landmarks[0], landmarks[23], landmarks[24]))
        leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        legs_straight = leg_angle > 160
        back_straight = spine_angle < 20
        sitting = landmarks[23].y > 0.6
        
        return back_straight and legs_straight and sitting

    def is_shoulder_pressing(self, landmarks):
        """Detect shoulder pressing pose"""
        if self.get_body_orientation(landmarks) != "standing":
            return False
        
        shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        elbow_bend = shoulder_angle < 90
        arms_raised = landmarks[13].y < landmarks[11].y
        forearms_vertical = abs(landmarks[15].x - landmarks[13].x) < 0.1
        
        return elbow_bend and arms_raised and forearms_vertical

    def is_reclining_hero(self, landmarks):
        """Detect reclining hero pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        back_flat = abs(landmarks[11].y - landmarks[23].y) < 0.1
        knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        knees_bent = knee_angle < 90
        lying_back = landmarks[0].y < landmarks[23].y
        feet_position = abs(landmarks[27].y - landmarks[28].y) < 0.1
        
        if self.debug:
            print(f"Reclining Hero Check:")
            print(f"Back flat: {back_flat}")
            print(f"Knee angle: {knee_angle:.2f}")
            print(f"Lying back: {lying_back}")
            print(f"Feet aligned: {feet_position}")
        
        return back_flat and knees_bent and lying_back and feet_position

    def is_cockerel(self, landmarks):
        """Detect cockerel pose"""
        if self.get_body_orientation(landmarks) != "standing":
            return False
        
        # Calculate leg angles
        left_leg = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        right_leg = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        leg_bend = min(left_leg, right_leg)
        
        # Calculate spine and balance points
        spine_vertical = abs(landmarks[0].x - landmarks[24].x) < 0.25  # Relaxed from 0.2
        hip_level = abs(landmarks[23].y - landmarks[24].y) < 0.15
        one_foot_raised = abs(landmarks[27].y - landmarks[28].y) > 0.2
        
        if self.debug:
            print(f"Cockerel Check:")
            print(f"Leg bend: {leg_bend:.2f}")
            print(f"Spine vertical: {spine_vertical}")
            print(f"Hip level: {hip_level}")
            print(f"One foot raised: {one_foot_raised}")
        
        return (leg_bend < 70 and  
                spine_vertical and
                hip_level and
                one_foot_raised)

    def is_frog(self, landmarks):
        """Detect frog pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        hip_width = abs(landmarks[23].x - landmarks[24].x)
        feet_width = abs(landmarks[27].x - landmarks[28].x)
        knees_wide = knee_angle < 60
        hips_open = hip_width > 0.3
        feet_aligned = abs(feet_width - hip_width) < 0.1
        
        if self.debug:
            print(f"Frog Check:")
            print(f"Knee angle: {knee_angle:.2f}")
            print(f"Hip width: {hip_width:.2f}")
            print(f"Feet aligned: {feet_aligned}")
        
        return knees_wide and hips_open and feet_aligned

    def is_reclining_cobbler(self, landmarks):
        """Detect reclining cobbler pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        back_flat = abs(landmarks[11].y - landmarks[23].y) < 0.1
        feet_together = abs(landmarks[27].x - landmarks[28].x) < 0.2
        knees_apart = abs(landmarks[25].x - landmarks[26].x) > 0.3
        lying_back = landmarks[0].y < landmarks[23].y
        
        if self.debug:
            print(f"Reclining Cobbler Check:")
            print(f"Back flat: {back_flat}")
            print(f"Feet together: {feet_together}")
            print(f"Knees apart: {knees_apart}")
            print(f"Lying back: {lying_back}")
        
        return back_flat and feet_together and knees_apart and lying_back

    def is_wind_relieving(self, landmarks):
        """Detect wind relieving pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        knee_chest = abs(landmarks[25].y - landmarks[11].y) < 0.2
        other_leg = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        other_leg_straight = other_leg > 150
        back_flat = abs(landmarks[11].y - landmarks[23].y) < 0.1
        
        if self.debug:
            print(f"Wind Relieving Check:")
            print(f"Knee to chest: {knee_chest}")
            print(f"Other leg angle: {other_leg:.2f}")
            print(f"Back flat: {back_flat}")
        
        return knee_chest and other_leg_straight and back_flat

    def generate_pose_feedback(self, landmarks, pose_name):
        """Generate pose-specific feedback for all supported poses"""
        if not pose_name:
            return None

        feedback = None
        orientation = self.get_body_orientation(landmarks)

        if pose_name == "balancing-table":
            trunk_angle = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            leg_lift = abs(landmarks[27].y - landmarks[28].y)
            hip_level = abs(landmarks[23].y - landmarks[24].y)
            hands_position = abs(landmarks[15].x - landmarks[16].x)

            if hands_position < 0.3:
                feedback = "Widen your hand position for better stability"
            elif abs(trunk_angle - 90) > 20:
                feedback = "Keep your spine parallel to the floor"
            elif leg_lift < 0.2:
                feedback = "Lift your extended leg higher"
            elif hip_level > 0.1:
                feedback = "Keep your hips level"

        elif pose_name == "tree":
            head_x = landmarks[0].x
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            standing_leg = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            hip_level = abs(landmarks[23].y - landmarks[24].y)

            if orientation != "standing":
                feedback = "Come to a standing position"
            elif abs(head_x - hip_x) > 0.2:
                feedback = "Stack your head over your hips"
            elif standing_leg < 160:
                feedback = "Ground firmly through your standing foot"
            elif hip_level > 0.1:
                feedback = "Level your hips"

        elif pose_name == "downward-dog":
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            head_y = landmarks[0].y
            arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])

            if orientation != "inverted":
                feedback = "Lift your hips higher to form an inverted V"
            elif hip_y >= head_y:
                feedback = "Lift your hips higher towards the ceiling"
            elif arm_angle < 160:
                feedback = "Press the floor away, straighten your arms"
            elif leg_angle < 160:
                feedback = "Work towards straightening your legs"

        elif pose_name == "staff":
            spine_angle = self.calculate_angle(landmarks[0], landmarks[23], landmarks[24])
            leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            shoulder_level = abs(landmarks[11].y - landmarks[12].y)

            if abs(spine_angle - 90) > 20:
                feedback = "Sit up taller, lengthen your spine"
            elif leg_angle < 160:
                feedback = "Engage your thighs and straighten your legs"
            elif shoulder_level > 0.1:
                feedback = "Level your shoulders"

        elif pose_name == "shoulder-pressing":
            shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            wrist_position = landmarks[15].y - landmarks[13].y
            elbow_width = abs(landmarks[13].x - landmarks[14].x)

            if orientation != "standing":
                feedback = "Come to a standing position"
            elif shoulder_angle >= 90:
                feedback = "Bend your elbows more deeply"
            elif wrist_position > 0:
                feedback = "Lift your forearms higher"
            elif elbow_width > 0.3:
                feedback = "Keep your elbows closer to your body"

        elif pose_name == "reclining-hero":
            back_flat = abs(landmarks[11].y - landmarks[23].y)
            knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            hip_alignment = abs(landmarks[23].x - landmarks[24].x)

            if orientation != "horizontal":
                feedback = "Lie back with your spine on the floor"
            elif back_flat > 0.1:
                feedback = "Keep your back flat on the ground"
            elif knee_angle > 90:
                feedback = "Bend your knees more deeply"
            elif hip_alignment > 0.2:
                feedback = "Keep your hips square to the ground"

        return feedback

    def run_live_detection(self):
        """Run pose detection on pre-recorded video"""
        cap = cv2.VideoCapture(VIDEO_FILE)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {VIDEO_FILE}")

        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video writer
        out = cv2.VideoWriter(TEMP_VIDEO, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, 
                             (frame_width, frame_height))

        audio_files = []
        logging.info(f"Processing video: {VIDEO_FILE}")
        logging.info(f"Total frames: {total_frames}")

        try:
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame
                    processed_frame, pose_name, confidence, audio_file = self.process_frame(frame)

                    # Handle audio feedback
                    if audio_file and os.path.exists(audio_file):
                        audio_files.append(audio_file)
                        logging.info(f"Generated feedback: {audio_file}")

                    # Write frame to output
                    out.write(processed_frame)
                    pbar.update(1)

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # Handle audio merging if any audio files were generated
            if audio_files:
                logging.info("Processing audio files...")
                concat_file = os.path.join(AUDIO_DIR, "concat.txt")
                with open(concat_file, "w") as f:
                    for audio in audio_files:
                        f.write(f"file '{os.path.basename(audio)}'\n")

                # Combine audio files
                combined_audio = os.path.join(AUDIO_DIR, "combined.wav")
                os.system(f"cd {AUDIO_DIR} && ffmpeg -y -f concat -safe 0 -i concat.txt -c copy {os.path.basename(combined_audio)}")

                # Merge video with audio
                logging.info("Merging video and audio...")
                os.system(f"ffmpeg -y -i {TEMP_VIDEO} -i {combined_audio} -c:v copy -c:a aac {FINAL_OUTPUT}")

                # Cleanup
                os.remove(TEMP_VIDEO)
                os.remove(combined_audio)
                os.remove(concat_file)
                for audio_file in audio_files:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)

                logging.info(f"Final output saved to: {FINAL_OUTPUT}")
            else:
                os.rename(TEMP_VIDEO, FINAL_OUTPUT)
                logging.info(f"Output saved to: {FINAL_OUTPUT}")

if __name__ == "__main__":
    try:
        detector = LivePoseDetector(MODEL_FILE, SCALER_FILE, ENCODER_FILE)
        logging.info("Starting pose detection...")
        logging.info(f"Supported poses: {detector.supported_poses}")
        detector.run_live_detection()
    except Exception as e:
        logging.error(f"Error in pose detection: {e}")
        raise
