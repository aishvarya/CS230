# STEP 1: Import the necessary modules.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import json
import csv

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
model_path = 'v2/CS230/mp/pose_landmarker_heavy.task'
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

dir_path='v2/CS230/posture_detection/datasets/pushup/test/'


import json
with open("v2/CS230/posture_detection/datasets/pushup/test/annotations.json", mode="r", encoding="utf-8") as read_file:
  annotations = json.load(read_file)


a = []

# Using a loop to create a list of dictionaries
for i in range(0,len(annotations)):
  
  item=(annotations[i]['file_name'])
  mp_image = mp.Image.create_from_file(dir_path + item)
  pose_landmarker_result = detector.detect(mp_image)

  yankle=pose_landmarker_result.pose_landmarks[0][27].y
  yhip=pose_landmarker_result.pose_landmarks[0][23].y
  xhip=pose_landmarker_result.pose_landmarks[0][23].x
  xankle=pose_landmarker_result.pose_landmarks[0][27].x
  yshoulder=pose_landmarker_result.pose_landmarks[0][11].y
  xshoulder=pose_landmarker_result.pose_landmarks[0][11].x
  body_angle=(((180/np.pi)*np.arctan2(abs(yankle-yhip), abs(xankle-xhip)))-((180/np.pi)*np.arctan2(abs(yshoulder-yhip), abs(xshoulder-xhip))))
  #print(annotations[i]['file_name'])
  #print(body_angle)
  if body_angle>10:
    a.append({"file_name": annotations[i]['file_name'], "label": annotations[i]['label'], "subclass":annotations[i]['subclass'], "biomechanics":"raised_hips"})
  elif body_angle <-10:
    a.append({"file_name": annotations[i]['file_name'], "label": annotations[i]['label'], "subclass":annotations[i]['subclass'], "biomechanics":"lower_back_touching_ground"})
  else:
    a.append({"file_name": annotations[i]['file_name'], "label": annotations[i]['label'], "subclass":annotations[i]['subclass'], "biomechanics":"none"})




N=0
for im in range(0, len(a)):
  if a[im]['subclass']==a[im]['biomechanics']:
    N+=1

print("Accuracy: " + str(N/len(a)))


myFile = open('demo_file.csv', 'w')
writer = csv.writer(myFile)
writer.writerow(['Name', 'label', 'subclass', 'biomech'])
for dictionary in a:
    writer.writerow(dictionary.values())
myFile.close()
print(a)