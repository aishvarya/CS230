# import joblib
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)

# def analyze_scaler(scaler_path):
#     """Analyze the current scaler file"""
#     try:
#         scaler = joblib.load(scaler_path)
#         logging.info("Scaler Analysis:")
#         logging.info(f"Type: {type(scaler)}")
#         logging.info(f"Mean shape: {scaler.mean_.shape}")
#         logging.info(f"Scale shape: {scaler.scale_.shape}")
#         logging.info(f"Number of features: {len(scaler.mean_)}")
#         logging.info("\nFirst few means:")
#         logging.info(scaler.mean_[:5])
#         logging.info("\nFirst few scales:")
#         logging.info(scaler.scale_[:5])
#         return scaler
#     except Exception as e:
#         logging.error(f"Error loading scaler: {e}")
#         return None

# if __name__ == "__main__":
#     SCALER_FILE = '/home/ubuntu/v2/yoga/models/scaler.pkl'
#     analyze_scaler(SCALER_FILE)
#code for checking old scaler

#code for generating new scaler 
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import joblib
# import logging
# import mediapipe as mp
# import cv2
# import os
# from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)

# def process_landmarks(landmarks):
#     """Convert landmarks to feature vector"""
#     features = []
#     for landmark in landmarks:
#         features.extend([landmark.x, landmark.y, landmark.z])  # 3 coordinates per landmark
#     return features

# def extract_features_from_video(video_path):
#     """Extract pose features from a video file"""
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=2,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
    
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     features_list = []
    
#     logging.info(f"Processing video: {video_path}")
#     with tqdm(total=total_frames, desc="Extracting Features") as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             # Convert to RGB and process
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
            
#             if results.pose_landmarks:
#                 features = process_landmarks(results.pose_landmarks.landmark)
#                 features_list.append(features)
            
#             pbar.update(1)
    
#     cap.release()
#     pose.close()
    
#     return np.array(features_list)

# def create_and_analyze_scaler():
#     """Create new scaler and analyze it"""
#     # Video paths
#     video_dir = '/home/ubuntu/v2/yoga/videos'
#     sample_videos = ['sy-1.mp4', 'sy-2.mp4', 'sy-3.mp4']
#     video_paths = [os.path.join(video_dir, video) for video in sample_videos]
    
#     # Collect features from all videos
#     all_features = []
#     for video_path in video_paths:
#         if os.path.exists(video_path):
#             features = extract_features_from_video(video_path)
#             all_features.extend(features)
#             logging.info(f"Extracted {len(features)} samples from {os.path.basename(video_path)}")
#         else:
#             logging.warning(f"Video not found: {video_path}")
    
#     all_features = np.array(all_features)
#     logging.info(f"\nTotal samples collected: {len(all_features)}")
#     logging.info(f"Feature shape: {all_features.shape}")
    
#     # Create and fit scaler
#     scaler = StandardScaler()
#     scaler.fit(all_features)
    
#     # Analyze new scaler
#     logging.info("\nNew Scaler Analysis:")
#     logging.info(f"Type: {type(scaler)}")
#     logging.info(f"Mean shape: {scaler.mean_.shape}")
#     logging.info(f"Scale shape: {scaler.scale_.shape}")
#     logging.info(f"Number of features: {len(scaler.mean_)}")
#     logging.info("\nFirst few means:")
#     logging.info(scaler.mean_[:6])  # Show first two landmarks (x,y,z coordinates)
#     logging.info("\nFirst few scales:")
#     logging.info(scaler.scale_[:6])
    
#     # Save new scaler
#     output_path = '/home/ubuntu/v2/yoga/models/new_scaler.pkl'
#     joblib.dump(scaler, output_path)
#     logging.info(f"\nNew scaler saved to: {output_path}")
    
#     # Verify saved scaler
#     loaded_scaler = joblib.load(output_path)
#     logging.info("\nVerifying saved scaler:")
#     logging.info(f"Mean shape matches: {np.array_equal(scaler.mean_, loaded_scaler.mean_)}")
#     logging.info(f"Scale shape matches: {np.array_equal(scaler.scale_, loaded_scaler.scale_)}")
    
#     return scaler

# if __name__ == "__main__":
#     create_and_analyze_scaler()

#--------------------------------------

#script for checking model encoder
# import joblib
# import tensorflow as tf
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # File paths
# BASE_DIR = '/home/ubuntu/v2/yoga'
# MODEL_FILE = f'{BASE_DIR}/models/3dyoga90_pose_model.keras'
# ENCODER_FILE = f'{BASE_DIR}/models/label_encoder.pkl'

# # Original training order
# TRAINING_ORDER = [
#     'tree', 'downward-dog', 'staff', 'shoulder-pressing', 'cockerel',
#     'balancing-table', 'wind-relieving', 'reclining-cobbler', 'reclining-hero', 'frog'
# ]

# def check_encoder_order():
#     # Load label encoder
#     logging.info("Loading label encoder...")
#     encoder = joblib.load(ENCODER_FILE)
    
#     # Get encoder classes
#     encoder_classes = encoder.classes_
#     logging.info("\nLabel Encoder Order:")
#     for i, pose in enumerate(encoder_classes):
#         logging.info(f"{i}: {pose}")
        
#     # Compare with training order
#     logging.info("\nTraining Order:")
#     for i, pose in enumerate(TRAINING_ORDER):
#         logging.info(f"{i}: {pose}")
        
#     # Check if orders match
#     logging.info("\nOrder Comparison:")
#     logging.info(f"Orders match: {np.array_equal(encoder_classes, TRAINING_ORDER)}")
    
#     if not np.array_equal(encoder_classes, TRAINING_ORDER):
#         logging.info("\nMismatched positions:")
#         for pose in TRAINING_ORDER:
#             old_pos = TRAINING_ORDER.index(pose)
#             new_pos = np.where(encoder_classes == pose)[0][0]
#             if old_pos != new_pos:
#                 logging.info(f"Pose '{pose}' moved from position {old_pos} to {new_pos}")

# if __name__ == "__main__":
#     check_encoder_order()

# #---------------- checking training order
# import pandas as pd
# import joblib
# import logging
# import tensorflow as tf
# import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # File paths
# BASE_DIR = '/home/ubuntu/v2/yoga'
# DATA_FILE = f'{BASE_DIR}/csv_files/top_20_poses_combined_data.csv'
# MODEL_FILE = f'{BASE_DIR}/models/3dyoga90_pose_model.keras'
# ENCODER_FILE = f'{BASE_DIR}/models/label_encoder.pkl'

# def analyze_training_data():
#     # Load the original training data
#     logging.info("Loading training data...")
#     df = pd.read_csv(DATA_FILE)
#     logging.info(f"Total samples in dataset: {len(df)}")

#     # Get the original order from value counts
#     top_10_poses = df['l3_pose'].value_counts()
    
#     logging.info("\nActual training order (by frequency):")
#     for i, (pose, count) in enumerate(top_10_poses[:10].items()):
#         logging.info(f"{i}: {pose} - {count} samples")

#     # Load current label encoder
#     logging.info("\nLoading current label encoder...")
#     encoder = joblib.load(ENCODER_FILE)
#     logging.info("\nCurrent label encoder order:")
#     for i, pose in enumerate(encoder.classes_):
#         logging.info(f"{i}: {pose}")

#     # Load model and check output layer
#     logging.info("\nLoading model to check output layer...")
#     model = tf.keras.models.load_model(MODEL_FILE)
#     output_layer = model.layers[-1]
#     logging.info(f"Model output shape: {output_layer.output_shape}")
#     logging.info(f"Number of output classes: {output_layer.output_shape[-1]}")

#     # Verify all poses are present
#     encoder_poses = set(encoder.classes_)
#     training_poses = set(top_10_poses[:10].index)
    
#     logging.info("\nVerifying pose sets:")
#     logging.info(f"Poses in encoder: {len(encoder_poses)}")
#     logging.info(f"Top 10 poses in training data: {len(training_poses)}")
    
#     if encoder_poses != training_poses:
#         logging.info("\nMismatches found:")
#         logging.info(f"In encoder but not in top 10: {encoder_poses - training_poses}")
#         logging.info(f"In top 10 but not in encoder: {training_poses - encoder_poses}")

#     # Sample distribution analysis
#     logging.info("\nPose distribution in training data:")
#     distribution = df['l3_pose'].value_counts(normalize=True) * 100
#     for pose, percentage in distribution[:10].items():
#         logging.info(f"{pose}: {percentage:.2f}%")

# if __name__ == "__main__":
#     analyze_training_data()

#---------- checking encoder order


# import joblib
# import logging
# import numpy as np

# logging.basicConfig(level=logging.INFO)

# # Expected order from training data
# EXPECTED_ORDER = [
#     'staff',
#     'reclining-hero',
#     'cockerel',
#     'downward-dog',
#     'frog',
#     'reclining-cobbler',
#     'wind-relieving',
#     'shoulder-pressing',
#     'balancing-table',
#     'tree'
# ]

# def check_encoder():
#     # Load current encoder
#     encoder = joblib.load('/home/ubuntu/v2/yoga/models/label_encoder.pkl')
    
#     logging.info("Current Encoder Classes:")
#     for i, cls in enumerate(encoder.classes_):
#         logging.info(f"{i}: {cls}")
        
#     logging.info("\nExpected Order:")
#     for i, cls in enumerate(EXPECTED_ORDER):
#         logging.info(f"{i}: {cls}")
        
#     # Check transformations
#     logging.info("\nTesting Transformations:")
#     test_indices = np.arange(len(EXPECTED_ORDER))
#     transformed = encoder.inverse_transform(test_indices)
#     logging.info("Index -> Pose Mapping:")
#     for idx, pose in zip(test_indices, transformed):
#         logging.info(f"{idx} -> {pose}")

# if __name__ == "__main__":
#     check_encoder()

#-------- checking trained models order

# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import joblib
# import logging

# logging.basicConfig(level=logging.INFO)

# # File paths
# BASE_DIR = '/home/ubuntu/v2/yoga'
# MODEL_FILE = f'{BASE_DIR}/models/3dyoga90_pose_model.keras'
# ENCODER_FILE = f'{BASE_DIR}/models/label_encoder.pkl'
# TRAINING_DATA = f'{BASE_DIR}/csv_files/top_20_poses_combined_data.csv'

# def analyze_model_and_training():
#     # Load model and encoder
#     logging.info("Loading model and encoder...")
#     model = tf.keras.models.load_model(MODEL_FILE)
#     encoder = joblib.load(ENCODER_FILE)
    
#     # Check model architecture
#     logging.info("\nModel Architecture:")
#     model.summary(print_fn=logging.info)
    
#     # Check output layer size
#     output_layer = model.layers[-1]
#     logging.info(f"\nOutput layer size: {output_layer.units}")
#     logging.info(f"Number of poses in encoder: {len(encoder.classes_)}")
    
#     # Load a small sample of training data
#     logging.info("\nLoading sample of training data...")
#     df = pd.read_csv(TRAINING_DATA, nrows=1000)
    
#     # Check unique poses in training data
#     unique_poses = df['l3_pose'].unique()
#     logging.info(f"\nUnique poses in training data sample:")
#     for pose in unique_poses:
#         count = len(df[df['l3_pose'] == pose])
#         logging.info(f"{pose}: {count} samples")
    
#     # Check encoded values
#     logging.info("\nEncoder mappings:")
#     for i, pose in enumerate(encoder.classes_):
#         logging.info(f"Index {i} -> {pose}")
    
#     # Create a sample prediction using actual training data
#     if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
#         sample = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']].iloc[0:1]
#         try:
#             prediction = model.predict(sample, verbose=0)
#             predicted_index = np.argmax(prediction[0])
#             predicted_pose = encoder.inverse_transform([predicted_index])[0]
#             actual_pose = df['l3_pose'].iloc[0]
            
#             logging.info("\nSample Prediction Test:")
#             logging.info(f"Actual pose: {actual_pose}")
#             logging.info(f"Predicted pose: {predicted_pose}")
#             logging.info("Prediction probabilities:")
#             for i, prob in enumerate(prediction[0]):
#                 pose = encoder.inverse_transform([i])[0]
#                 logging.info(f"{pose}: {prob:.4f}")
#         except Exception as e:
#             logging.error(f"Error making prediction: {e}")

# if __name__ == "__main__":
#     analyze_model_and_training()

#---------testing model and encoder compatibility

# import tensorflow as tf
# import joblib

# # Update the path to point to the correct file location
# model_path = '/home/ubuntu/v2/yoga/models/3dyoga90_pose_model.keras'
# model = tf.keras.models.load_model(model_path)

# # Extract the number of output units in the last layer of the model
# output_units = model.layers[-1].units

# # Load the label encoder
# label_encoder_path = '/home/ubuntu/v2/yoga/models/label_encoder.pkl'
# label_encoder = joblib.load(label_encoder_path)

# # Number of classes in the label encoder
# num_classes_label_encoder = len(label_encoder.classes_)

# # Display the comparison
# print(f"Model output units: {output_units}")
# print(f"Number of classes in label encoder: {num_classes_label_encoder}")

# if output_units == num_classes_label_encoder:
#     print("The model output layer matches the label encoder.")
# else:
#     print("The model output layer does NOT match the label encoder.")

#------- testing every step -keras parquet csv files


# import pandas as pd
# import os
# import logging
# import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # File paths
# BASE_DIR = '/home/ubuntu/v2/yoga'
# PARQUET_DIR = os.path.join(BASE_DIR, '3DYoga90/data_process/precomputed_skeleton/official_dataset')
# SKELETAL_CSV = os.path.join(BASE_DIR, 'csv_files/extracted_skeletal_data_with_sequence.csv')
# LANDMARK_CSV = os.path.join(BASE_DIR, 'csv_files/3dyoga90_landmarks/3DYoga90.csv')
# TRAINING_CSV = os.path.join(BASE_DIR, 'csv_files/top_20_poses_combined_data.csv')
# MODEL_FILE = os.path.join(BASE_DIR, 'models/3dyoga90_pose_model.keras')
# SCALER_FILE = os.path.join(BASE_DIR, 'models/scaler.pkl')
# ENCODER_FILE = os.path.join(BASE_DIR, 'models/label_encoder.pkl')

# def analyze_parquet_data():
#     """Analyze structure of parquet files"""
#     logging.info("\n=== Analyzing Parquet Data ===")
#     try:
#         sample_parquet = os.listdir(PARQUET_DIR)[0]
#         df_parquet = pd.read_parquet(os.path.join(PARQUET_DIR, sample_parquet))
        
#         logging.info(f"Parquet file sample: {sample_parquet}")
#         logging.info(f"Shape: {df_parquet.shape}")
#         logging.info(f"Columns: {df_parquet.columns.tolist()}")
#         logging.info("\nSample data:")
#         print(df_parquet.head())
#         logging.info("\nData types:")
#         print(df_parquet.dtypes)
        
#         return df_parquet.shape[1]  # Return number of columns for comparison
#     except Exception as e:
#         logging.error(f"Error analyzing parquet data: {e}")
#         return None

# def analyze_skeletal_data():
#     """Analyze structure of extracted skeletal data"""
#     logging.info("\n=== Analyzing Skeletal Data ===")
#     try:
#         df_skeletal = pd.read_csv(SKELETAL_CSV, nrows=5)
        
#         logging.info(f"Shape: {df_skeletal.shape}")
#         logging.info(f"Columns: {df_skeletal.columns.tolist()}")
#         logging.info("\nSample data:")
#         print(df_skeletal.head())
#         logging.info("\nData types:")
#         print(df_skeletal.dtypes)
        
#         return df_skeletal.shape[1]
#     except Exception as e:
#         logging.error(f"Error analyzing skeletal data: {e}")
#         return None

# def analyze_landmark_data():
#     """Analyze structure of landmark data"""
#     logging.info("\n=== Analyzing Landmark Data ===")
#     try:
#         df_landmarks = pd.read_csv(LANDMARK_CSV, nrows=5)
        
#         logging.info(f"Shape: {df_landmarks.shape}")
#         logging.info(f"Columns: {df_landmarks.columns.tolist()}")
#         logging.info("\nSample data:")
#         print(df_landmarks.head())
#         logging.info("\nData types:")
#         print(df_landmarks.dtypes)
        
#         # Analyze pose distribution
#         if 'l3_pose' in df_landmarks.columns:
#             logging.info("\nPose distribution:")
#             print(df_landmarks['l3_pose'].value_counts())
        
#         return df_landmarks.shape[1]
#     except Exception as e:
#         logging.error(f"Error analyzing landmark data: {e}")
#         return None

# def analyze_training_data():
#     """Analyze structure of final training data"""
#     logging.info("\n=== Analyzing Training Data ===")
#     try:
#         df_training = pd.read_csv(TRAINING_CSV, nrows=5)
        
#         logging.info(f"Shape: {df_training.shape}")
#         logging.info(f"Columns: {df_training.columns.tolist()}")
#         logging.info("\nSample data:")
#         print(df_training.head())
#         logging.info("\nData types:")
#         print(df_training.dtypes)
        
#         # Analyze pose distribution
#         if 'l3_pose' in df_training.columns:
#             logging.info("\nPose distribution (top 10):")
#             print(df_training['l3_pose'].value_counts().head(10))
        
#         return df_training.shape[1]
#     except Exception as e:
#         logging.error(f"Error analyzing training data: {e}")
#         return None

# def verify_model_and_encoder():
#     """Verify model input shape and encoder classes"""
#     logging.info("\n=== Analyzing Model and Encoder ===")
#     try:
#         import tensorflow as tf
#         import joblib
        
#         # Load model
#         model = tf.keras.models.load_model(MODEL_FILE)
#         logging.info("\nModel Summary:")
#         model.summary(print_fn=logging.info)
        
#         # Load encoder
#         encoder = joblib.load(ENCODER_FILE)
#         logging.info("\nEncoder classes:")
#         for i, cls in enumerate(encoder.classes_):
#             logging.info(f"{i}: {cls}")
        
#         # Load scaler
#         scaler = joblib.load(SCALER_FILE)
#         logging.info("\nScaler information:")
#         logging.info(f"Mean shape: {scaler.mean_.shape}")
#         logging.info(f"Scale shape: {scaler.scale_.shape}")
#         logging.info(f"Mean values: {scaler.mean_}")
#         logging.info(f"Scale values: {scaler.scale_}")
        
#     except Exception as e:
#         logging.error(f"Error analyzing model and encoder: {e}")

# def verify_data_pipeline():
#     """Verify entire data pipeline and check for consistency"""
#     logging.info("\n=== Starting Data Pipeline Verification ===")
    
#     # Analyze each step
#     parquet_cols = analyze_parquet_data()
#     skeletal_cols = analyze_skeletal_data()
#     landmark_cols = analyze_landmark_data()
#     training_cols = analyze_training_data()
    
#     # Verify model and encoder
#     verify_model_and_encoder()
    
#     # Compare column counts
#     logging.info("\n=== Data Pipeline Summary ===")
#     logging.info(f"Parquet columns: {parquet_cols}")
#     logging.info(f"Skeletal data columns: {skeletal_cols}")
#     logging.info(f"Landmark data columns: {landmark_cols}")
#     logging.info(f"Training data columns: {training_cols}")

# if __name__ == "__main__":
#     verify_data_pipeline()

#---------------------------testing training data

# import pandas as pd
# import logging
# import numpy as np

# logging.basicConfig(level=logging.INFO)

# # Load training data
# DATA_FILE = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
# df = pd.read_csv(DATA_FILE)

# # Analyze pose distribution
# logging.info("Pose Distribution in Training Data:")
# pose_dist = df['l3_pose'].value_counts()
# for pose, count in pose_dist.items():
#     logging.info(f"{pose}: {count} ({count/len(df)*100:.2f}%)")

# # Analyze feature ranges per pose
# logging.info("\nFeature Ranges per Pose:")
# for pose in df['l3_pose'].unique():
#     pose_data = df[df['l3_pose'] == pose]
#     logging.info(f"\n{pose}:")
#     for col in ['x', 'y', 'z']:
#         logging.info(f"{col}: min={pose_data[col].min():.3f}, max={pose_data[col].max():.3f}, mean={pose_data[col].mean():.3f}")

# # Check sequence_id distribution
# logging.info("\nSequence ID Distribution:")
# seq_dist = df['sequence_id'].value_counts().head()
# logging.info(f"Number of unique sequence_ids: {df['sequence_id'].nunique()}")
# logging.info("Sample of sequence_ids and their counts:")
# for seq, count in seq_dist.items():
#     logging.info(f"Sequence {seq}: {count} entries")

#--------checking combined_csv file data

# import pandas as pd
# import numpy as np

# # Load and analyze specific pose data
# df = pd.read_csv('/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv')

# # Get sample rows for downward-dog and tree
# poses_to_analyze = ['downward-dog', 'tree']
# samples = pd.DataFrame()

# for pose in poses_to_analyze:
#     pose_data = df[df['l3_pose'] == pose].groupby('sequence_id').first().reset_index()
#     samples = pd.concat([samples, pose_data.head(5)], ignore_index=True)

# # Save samples
# samples.to_csv('pose_samples.csv', index=False)

# # Print statistics
# print("\nPose Statistics:")
# for pose in poses_to_analyze:
#     pose_data = df[df['l3_pose'] == pose]
#     stats = pose_data[['x', 'y', 'z']].describe()
#     print(f"\n{pose} statistics:")
#     print(stats)

# print("\nTotal sequences per pose:")
# print(df.groupby('l3_pose')['sequence_id'].nunique())

#------ testing new model

# import cv2
# import mediapipe as mp
# import numpy as np
# import logging
# import os
# from collections import deque
# from gtts import gTTS
# import time

# logging.basicConfig(level=logging.INFO)

# BASE_DIR = '/home/ubuntu/v2/yoga'
# VIDEO_FILE = os.path.join(BASE_DIR, 'videos/sy-6.mp4')
# OUTPUT_FILE = os.path.join(BASE_DIR, 'videos/output.mp4')
# AUDIO_DIR = os.path.join(BASE_DIR, 'audio_feedback')
# os.makedirs(AUDIO_DIR, exist_ok=True)

# class PoseClassifier:
#     def __init__(self):
#         self.pose_history = deque(maxlen=5)
#         self.last_audio_time = 0
#         self.last_feedback = None
        
#     def calculate_angle(self, a, b, c):
#         a = np.array([a.x, a.y])
#         b = np.array([b.x, b.y])
#         c = np.array([c.x, c.y])
#         radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#         angle = np.abs(np.degrees(radians))
#         return angle if angle <= 180 else 360-angle

#     def get_body_orientation(self, landmarks):
#         spine_angle = self.calculate_angle(
#             landmarks[0],
#             landmarks[23],
#             landmarks[24]
#         )
#         hip_y = (landmarks[23].y + landmarks[24].y) / 2
#         feet_y = (landmarks[27].y + landmarks[28].y) / 2
#         is_inverted = hip_y < landmarks[0].y
        
#         if is_inverted and feet_y > hip_y:
#             return "inverted"
#         elif abs(hip_y - feet_y) < 0.2 and landmarks[0].y < hip_y:
#             return "horizontal"
#         else:
#             return "standing"

#     def is_balancing_table(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "horizontal":
#             return False
        
#         hip_y = (landmarks[23].y + landmarks[24].y) / 2
#         shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
#         wrist_y = (landmarks[15].y + landmarks[16].y) / 2
        
#         trunk_parallel = abs(hip_y - shoulder_y) < 0.1
#         arms_straight = abs(shoulder_y - wrist_y) < 0.1
#         leg_lifted = abs(landmarks[27].y - landmarks[28].y) > 0.2
        
#         return trunk_parallel and arms_straight and leg_lifted

#     def is_tree(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "standing":
#             return False
        
#         ankle_diff = abs(landmarks[27].y - landmarks[28].y)
#         spine_vertical = abs(landmarks[0].x - landmarks[24].x) < 0.2
#         hip_level = abs(landmarks[23].y - landmarks[24].y) < 0.1
        
#         return ankle_diff > 0.15 and spine_vertical and hip_level

#     def is_downward_dog(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "inverted":
#             return False
            
#         hip_y = (landmarks[23].y + landmarks[24].y) / 2
#         arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
#         leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
#         hip_height = abs(hip_y - landmarks[0].y)
        
#         return hip_height > 0.2 and arm_angle > 160 and leg_angle > 160

#     def is_staff(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "horizontal":
#             return False
            
#         spine_angle = abs(90 - self.calculate_angle(landmarks[0], landmarks[23], landmarks[24]))
#         leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
#         return spine_angle < 20 and leg_angle > 160

#     def is_shoulder_pressing(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "standing":
#             return False
            
#         shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
#         wrists_up = landmarks[15].y < landmarks[13].y
#         return shoulder_angle < 90 and wrists_up

#     def is_reclining_hero(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "horizontal":
#             return False
            
#         back_flat = abs(landmarks[11].y - landmarks[23].y) < 0.1
#         knees_bent = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27]) < 90
#         lying_back = landmarks[0].y < landmarks[23].y
        
#         return back_flat and knees_bent and lying_back

#     def is_cockerel(self, landmarks):
#         orientation = self.get_body_orientation(landmarks)
#         if orientation != "standing":
#             return False
            
#         leg_bend = min(
#             self.calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
#             self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
#         )
#         spine_vertical = abs(landmarks[0].x - landmarks[24].x) < 0.2
        
#         return leg_bend < 60 and spine_vertical

#     def get_pose_feedback(self, landmarks, pose_name):
#         if pose_name == "balancing-table":
#             trunk_angle = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
#             leg_lift = abs(landmarks[27].y - landmarks[28].y)
#             hip_level = abs(landmarks[23].y - landmarks[24].y)
            
#             if abs(trunk_angle - 90) > 20:
#                 return "Keep your spine parallel to the floor"
#             if leg_lift < 0.2:
#                 return "Lift your extended leg higher"
#             if hip_level > 0.1:
#                 return "Keep your hips level"

#         elif pose_name == "tree":
#             head_x = landmarks[0].x
#             hip_x = (landmarks[23].x + landmarks[24].x) / 2
#             standing_leg = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            
#             if abs(head_x - hip_x) > 0.2:
#                 return "Stack your head over your hips"
#             if standing_leg < 160:
#                 return "Press firmly into your standing foot"
#             if abs(landmarks[23].y - landmarks[24].y) > 0.1:
#                 return "Level your hips"

#         elif pose_name == "downward-dog":
#             hip_y = (landmarks[23].y + landmarks[24].y) / 2
#             head_y = landmarks[0].y
#             arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
#             leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            
#             if hip_y >= head_y:
#                 return "Lift your hips higher"
#             if arm_angle < 160:
#                 return "Straighten your arms"
#             if leg_angle < 160:
#                 return "Work on straightening your legs"

#         return self.get_additional_feedback(landmarks, pose_name)

#     def get_additional_feedback(self, landmarks, pose_name):
#         if pose_name == "staff":
#             spine_angle = self.calculate_angle(landmarks[0], landmarks[23], landmarks[24])
#             leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            
#             if abs(spine_angle - 90) > 20:
#                 return "Sit taller, lengthen your spine"
#             if leg_angle < 160:
#                 return "Engage your thighs, straighten your legs"

#         elif pose_name == "shoulder-pressing":
#             shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
#             wrist_position = landmarks[15].y - landmarks[13].y
            
#             if shoulder_angle >= 90:
#                 return "Bend your elbows more deeply"
#             if wrist_position > 0:
#                 return "Lift your forearms higher"

#         elif pose_name == "reclining-hero":
#             back_flat = abs(landmarks[11].y - landmarks[23].y)
#             knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            
#             if back_flat > 0.1:
#                 return "Keep your back flat on the ground"
#             if knee_angle > 90:
#                 return "Bend your knees more deeply"

#         elif pose_name == "cockerel":
#             leg_bend = min(
#                 self.calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
#                 self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
#             )
#             spine_vertical = abs(landmarks[0].x - landmarks[24].x)
            
#             if leg_bend > 60:
#                 return "Bend your leg more deeply"
#             if spine_vertical > 0.2:
#                 return "Keep your spine vertical"

#         return None

#     def generate_audio_feedback(self, pose_name, feedback_text=None):
#         current_time = time.time()
#         if feedback_text and feedback_text != self.last_feedback and current_time - self.last_audio_time > 5:
#             tts = gTTS(text=feedback_text, lang='en')
#             audio_file = os.path.join(AUDIO_DIR, f'feedback_{int(current_time)}.wav')
#             mp3_file = os.path.join(AUDIO_DIR, f'feedback_{int(current_time)}.mp3')
#             tts.save(mp3_file)
#             os.system(f'ffmpeg -i {mp3_file} -acodec pcm_s16le -ar 44100 {audio_file}')
#             os.remove(mp3_file)
#             self.last_audio_time = current_time
#             self.last_feedback = feedback_text
#             return audio_file
#         return None

#     def classify_pose(self, landmarks):
#         checks = {
#             "balancing-table": self.is_balancing_table,
#             "tree": self.is_tree,
#             "downward-dog": self.is_downward_dog,
#             "staff": self.is_staff,
#             "shoulder-pressing": self.is_shoulder_pressing,
#             "reclining-hero": self.is_reclining_hero,
#             "cockerel": self.is_cockerel
#         }
        
#         for pose_name, check_func in checks.items():
#             if check_func(landmarks):
#                 confidence = 0.9
#                 feedback = self.get_pose_feedback(landmarks, pose_name)
#                 audio_file = self.generate_audio_feedback(pose_name, feedback)
#                 return pose_name, confidence, audio_file, feedback
                
#         return "unknown", 0.0, None, None

# def main():
#     classifier = PoseClassifier()
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(min_detection_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils  # Fixed this line


#     cap = cv2.VideoCapture(VIDEO_FILE)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     temp_video = os.path.join(BASE_DIR, 'videos/temp_output.mp4')
#     out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), 
#                          fps, (frame_width, frame_height))

#     audio_files = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(image_rgb)

#         if results.pose_landmarks:
#             pose_name, confidence, audio_file, feedback = classifier.classify_pose(
#                 results.pose_landmarks.landmark
#             )
            
#             if audio_file:
#                 audio_files.append(audio_file)

#             if pose_name != "unknown":
#                 color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
#                 cv2.putText(
#                     frame,
#                     f'{pose_name}: {confidence:.2f}',
#                     (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     color,
#                     2
#                 )
                
#                 if feedback:
#                     cv2.putText(
#                         frame,
#                         feedback,
#                         (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 0, 255),
#                         2
#                     )

#             mp_drawing.draw_landmarks(
#                 frame,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS
#             )

#         out.write(frame)

#     cap.release()
#     out.release()

#     if audio_files:
#         with open(os.path.join(AUDIO_DIR, "audio_list.txt"), "w") as f:
#             for audio_file in audio_files:
#                 f.write(f"file '{audio_file}'\n")

#         combined_audio = os.path.join(AUDIO_DIR, "combined_audio.wav")
#         os.system(f'ffmpeg -f concat -safe 0 -i {os.path.join(AUDIO_DIR, "audio_list.txt")} -c copy {combined_audio}')

#         final_output = os.path.join(BASE_DIR, 'videos/output_with_audio.mp4')
#         os.system(f'ffmpeg -i {temp_video} -i {combined_audio} -c:v copy -c:a aac {final_output}')

#         os.remove(temp_video)
#         os.remove(combined_audio)
#         for audio_file in audio_files:
#             if os.path.exists(audio_file):
#                 os.remove(audio_file)

#     pose.close()

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed data
data_file = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
df = pd.read_csv(data_file)

# Prepare features and labels
features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]
labels = df['l3_pose']

# Split and evaluate
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Load trained model
model = tf.keras.models.load_model('/home/ubuntu/v2/yoga/models/3dyoga90_pose_model.keras')

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()