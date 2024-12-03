import cv2
import mediapipe as mp
import numpy as np
import logging
import os
from collections import deque
from gtts import gTTS
import time

logging.basicConfig(level=logging.INFO)

# File paths
BASE_DIR = '/home/ubuntu/v2/yoga'
VIDEO_FILE = os.path.join(BASE_DIR, 'videos/sy-6.mp4')
TEMP_VIDEO = os.path.join(BASE_DIR, 'videos/temp_output.mp4')
FINAL_OUTPUT = os.path.join(BASE_DIR, 'videos/final_output.mp4')
AUDIO_DIR = os.path.join(BASE_DIR, 'audio_feedback')
os.makedirs(AUDIO_DIR, exist_ok=True)

def draw_feedback(frame, pose_name, confidence, feedback_text):
    """Draw pose name and feedback on frame with background for better visibility"""
    # Background for pose name
    text_size = cv2.getTextSize(f'{pose_name}: {confidence:.2f}', 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, 
                 (5, 25), 
                 (15 + text_size[0], 65),
                 (0, 0, 0),
                 -1)
    
    # Pose name
    color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
    cv2.putText(
        frame,
        f'{pose_name}: {confidence:.2f}',
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )
    
    # Feedback text with background
    if feedback_text:
        # Split feedback text if too long
        words = feedback_text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_text = ' '.join(current_line)
            text_size = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            
            if text_size[0] > frame.shape[1] - 30:  # 30 pixels margin
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw each line
        for i, line in enumerate(lines):
            y_pos = 90 + i * 40
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            
            cv2.rectangle(frame, 
                         (5, y_pos - 25), 
                         (15 + text_size[0], y_pos + 15),
                         (0, 0, 0),
                         -1)
            
            cv2.putText(
                frame,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

class PoseClassifier:
    def __init__(self):
        self.pose_history = deque(maxlen=5)
        self.last_audio_time = 0
        self.last_feedback = None
        self.last_pose = None
        self.feedback_history = deque(maxlen=30)
        self.debug = True
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360-angle

    def get_body_orientation(self, landmarks):
        """Determine if body is standing, inverted, or horizontal"""
        spine_angle = self.calculate_angle(landmarks[0], landmarks[23], landmarks[24])
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        feet_y = (landmarks[27].y + landmarks[28].y) / 2
        is_inverted = hip_y < landmarks[0].y
        
        orientation = ("inverted" if is_inverted and feet_y > hip_y 
                      else "horizontal" if abs(hip_y - feet_y) < 0.2 and landmarks[0].y < hip_y 
                      else "standing")
        
        if self.debug:
            print(f"Spine angle: {spine_angle:.2f}")
            print(f"Hip Y: {hip_y:.2f}, Feet Y: {feet_y:.2f}")
            print(f"Is inverted: {is_inverted}")
            print(f"Detected orientation: {orientation}")
            
        return orientation

    def is_stable_pose(self, pose_name):
        """Check if pose has been consistently detected"""
        if len(self.pose_history) < self.pose_history.maxlen:
            return False
        return all(p == pose_name for p in self.pose_history)

    def generate_audio_feedback(self, pose_name, feedback_text=None):
        """Generate audio feedback with rate limiting"""
        current_time = time.time()
        if feedback_text and feedback_text not in self.feedback_history and current_time - self.last_audio_time > 5:
            try:
                self.feedback_history.append(feedback_text)
                tts = gTTS(text=feedback_text, lang='en')
                audio_file = os.path.join(AUDIO_DIR, f'feedback_{int(current_time)}.wav')
                mp3_file = os.path.join(AUDIO_DIR, f'feedback_{int(current_time)}.mp3')
                
                tts.save(mp3_file)
                os.system(f'ffmpeg -i {mp3_file} -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}')
                
                if os.path.exists(mp3_file):
                    os.remove(mp3_file)
                    
                self.last_audio_time = current_time
                return audio_file
            except Exception as e:
                print(f"Audio generation error: {e}")
                return None
        return None

    def is_balancing_table(self, landmarks):
        """Detect balancing table pose"""
        if self.get_body_orientation(landmarks) != "horizontal":
            return False
        
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        
        trunk_level = abs(hip_y - shoulder_y) < 0.15
        supporting_leg_grounded = max(landmarks[27].y, landmarks[28].y) > 0.7
        leg_lifted = abs(landmarks[27].y - landmarks[28].y) > 0.1
        hands_grounded = max(landmarks[15].y, landmarks[16].y) > 0.7
        
        if self.debug:
            print(f"Balancing Table Check:")
            print(f"Trunk level: {trunk_level}")
            print(f"Supporting leg grounded: {supporting_leg_grounded}")
            print(f"Leg lifted: {leg_lifted}")
            print(f"Hands grounded: {hands_grounded}")
        
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

    def get_pose_feedback(self, landmarks, pose_name):
        """Get feedback for detected pose"""
        if self.last_pose != pose_name:
            self.last_pose = pose_name
            self.feedback_history.clear()

        feedback = None
        
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
            
            if abs(head_x - hip_x) > 0.2:
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
            
            if hip_y >= head_y:
                feedback = "Lift your hips higher towards the ceiling"
            elif arm_angle < 160:
                feedback = "Press the floor away, straighten your arms"
            elif leg_angle < 160:
                feedback = "Work towards straightening your legs"

        elif pose_name == "staff":
            spine_angle = self.calculate_angle(landmarks[0], landmarks[23], landmarks[24])
            leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            
            if abs(spine_angle - 90) > 20:
                feedback = "Sit up taller, lengthen your spine"
            elif leg_angle < 160:
                feedback = "Engage your thighs and straighten your legs"
            elif abs(landmarks[11].y - landmarks[12].y) > 0.1:
                feedback = "Level your shoulders"

        elif pose_name == "shoulder-pressing":
            shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            wrist_position = landmarks[15].y - landmarks[13].y
            elbow_width = abs(landmarks[13].x - landmarks[14].x)
            
            if shoulder_angle >= 90:
                feedback = "Bend your elbows more deeply"
            elif wrist_position > 0:
                feedback = "Lift your forearms higher"
            elif elbow_width > 0.3:
                feedback = "Keep your elbows closer to your body"

        elif pose_name == "reclining-hero":
            back_flat = abs(landmarks[11].y - landmarks[23].y)
            knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            hip_alignment = abs(landmarks[23].x - landmarks[24].x)
            
            if back_flat > 0.1:
                feedback = "Keep your back flat on the ground"
            elif knee_angle > 90:
                feedback = "Bend your knees more deeply"
            elif hip_alignment > 0.2:
                feedback = "Keep your hips square to the ground"

        elif pose_name == "cockerel":
            leg_bend = min(
                self.calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
                self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            )
            spine_vertical = abs(landmarks[0].x - landmarks[24].x)
            balance = all(landmark.y > 0.3 for landmark in [landmarks[27], landmarks[28]])
            
            if leg_bend > 60:
                feedback = "Bend your leg more deeply"
            elif spine_vertical > 0.2:
                feedback = "Keep your spine vertical"
            elif not balance:
                feedback = "Find your balance point"

        elif pose_name == "frog":
            knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            hip_width = abs(landmarks[23].x - landmarks[24].x)
            back_alignment = abs(landmarks[11].y - landmarks[23].y)
            
            if knee_angle > 60:
                feedback = "Widen your knees more"
            elif hip_width < 0.3:
                feedback = "Open your hips wider"
            elif back_alignment > 0.1:
                feedback = "Keep your torso parallel to the ground"

        elif pose_name == "reclining-cobbler":
            back_flat = abs(landmarks[11].y - landmarks[23].y)
            knee_height = landmarks[25].y - landmarks[23].y
            feet_distance = abs(landmarks[27].x - landmarks[28].x)
            
            if back_flat > 0.1:
                feedback = "Press your lower back into the ground"
            elif knee_height > 0:
                feedback = "Let your knees relax towards the ground"
            elif feet_distance > 0.2:
                feedback = "Bring your feet closer together"

        elif pose_name == "wind-relieving":
            knee_chest = abs(landmarks[25].y - landmarks[11].y)
            other_leg = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            back_flat = abs(landmarks[11].y - landmarks[23].y)
            
            if knee_chest > 0.2:
                feedback = "Draw your knee closer to your chest"
            elif other_leg < 150:
                feedback = "Straighten your extended leg"
            elif back_flat > 0.1:
                feedback = "Keep your back pressed into the ground"

        return feedback

    def classify_pose(self, landmarks):
        """Classify the pose based on landmarks"""
        orientation = self.get_body_orientation(landmarks)
        
        checks = {
            "balancing-table": self.is_balancing_table,
            "tree": self.is_tree,
            "downward-dog": self.is_downward_dog,
            "staff": self.is_staff,
            "shoulder-pressing": self.is_shoulder_pressing,
            "reclining-hero": self.is_reclining_hero,
            "cockerel": self.is_cockerel,
            "frog": self.is_frog,
            "reclining-cobbler": self.is_reclining_cobbler,
            "wind-relieving": self.is_wind_relieving
        }
        
        for pose_name, check_func in checks.items():
            if check_func(landmarks):
                confidence = 0.9
                feedback = self.get_pose_feedback(landmarks, pose_name)
                audio_file = None
                
                # Only generate audio feedback if pose is stable
                if self.is_stable_pose(pose_name):
                    audio_file = self.generate_audio_feedback(pose_name, feedback)
                
                self.pose_history.append(pose_name)
                return pose_name, confidence, audio_file, feedback
        
        return "unknown", 0.0, None, None

def main():
    classifier = PoseClassifier()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_FILE}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(TEMP_VIDEO, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (frame_width, frame_height))

    audio_files = []
    frame_count = 0
    current_feedback = None

    print(f"Processing {total_frames} frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pose_name, confidence, audio_file, feedback = classifier.classify_pose(
                results.pose_landmarks.landmark
            )
            
            if pose_name != "unknown":
                # Update feedback if new feedback exists
                if feedback:
                    current_feedback = feedback
                    if audio_file and os.path.exists(audio_file):
                        audio_files.append(audio_file)
                        print(f"Generated feedback at frame {frame_count}: {feedback}")

                # Draw pose information and feedback
                draw_feedback(frame, pose_name, confidence, current_feedback)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

        out.write(frame)
        frame_count += 1

        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")

    # Release video resources
    cap.release()
    out.release()

    print("Video processing complete. Merging with audio...")
    if audio_files:
        # Create audio list file
        concat_file = os.path.join(AUDIO_DIR, "concat.txt")
        with open(concat_file, "w") as f:
            for audio in audio_files:
                f.write(f"file '{os.path.basename(audio)}'\n")

        # Combine audio files
        combined_audio = os.path.join(AUDIO_DIR, "combined.wav")
        os.system(f"cd {AUDIO_DIR} && ffmpeg -y -f concat -safe 0 -i concat.txt -c copy {os.path.basename(combined_audio)}")

        # Merge video with audio
        print("Merging video and audio...")
        os.system(f"ffmpeg -y -i {TEMP_VIDEO} -i {combined_audio} -c:v copy -c:a aac {FINAL_OUTPUT}")

        # Cleanup
        os.remove(TEMP_VIDEO)
        os.remove(combined_audio)
        os.remove(concat_file)
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)

        print(f"Final output saved to: {FINAL_OUTPUT}")
    else:
        os.rename(TEMP_VIDEO, FINAL_OUTPUT)
        print(f"Output saved to: {FINAL_OUTPUT}")

    pose.close()

if __name__ == "__main__":
    main()