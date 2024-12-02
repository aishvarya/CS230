# STEP 1: Import the necessary modules.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = 'v2/CS230/posture_detection/mediapipe input/pose_landmarker_heavy.task'
# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

#prepare image
import mediapipe as mp

# Load the input image from an image file.
import os
dir_path='v2/CS230/posture_detection/datasets/pushup/test'
items=os.listdir(dir_path)
items.remove('annotations.json')
for item in items:
  
  mp_image = mp.Image.create_from_file('v2/CS230/posture_detection/datasets/pushup/test/' + item)

  pose_landmarker_result = detector.detect(mp_image)

  import cv2
  annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)

  cv2.imwrite('v2/CS230/mp/dataset/test/' + item,annotated_image)