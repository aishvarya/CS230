import os
import cv2
import json
import random

# using https://www.kaggle.com/datasets/mohamadashrafsalama/pushup dataset
correct_dir = '/Users/aishvaryasingh/Downloads/pushup/Correct sequence'
incorrect_dir = '/Users/aishvaryasingh/Downloads/pushup/Wrong sequence'

output_train_dir = '/Users/aishvaryasingh/cs230_git/CS230/posture_detection/datasets/pushup/train'
output_test_dir = '/Users/aishvaryasingh/cs230_git/CS230/posture_detection/datasets/pushup/test'
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

train_annotations_path = os.path.join(output_train_dir, 'annotations.json')
test_annotations_path = os.path.join(output_test_dir, 'annotations.json')
train_annotations = []
test_annotations = []

def extract_frames(video_path, label, label_name, frame_interval=10, start_count=0):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_saved = 0
    temp_annotations = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i % frame_interval == 0:
            frame_filename = f"{label_name}_frame_{start_count + frames_saved}.jpg"
            
            temp_annotations.append({
                'file_name': frame_filename,
                'frame': frame,
                'label': label
            })
            frames_saved += 1

    cap.release()
    print(f"Extracted {frames_saved} frames from {video_path}")

    random.shuffle(temp_annotations)
    split_index = int(len(temp_annotations) * 0.8)
    train_data = temp_annotations[:split_index]
    test_data = temp_annotations[split_index:]
    
    for item in train_data:
        frame_path = os.path.join(output_train_dir, item['file_name'])
        cv2.imwrite(frame_path, item['frame'])
        train_annotations.append({
            'file_name': item['file_name'],
            'label': item['label']
        })
    
    for item in test_data:
        frame_path = os.path.join(output_test_dir, item['file_name'])
        cv2.imwrite(frame_path, item['frame'])
        test_annotations.append({
            'file_name': item['file_name'],
            'label': item['label']
        })

correct_frame_count = 0
incorrect_frame_count = 0

for video_file in os.listdir(correct_dir):
    video_path = os.path.join(correct_dir, video_file)
    if os.path.isfile(video_path) and video_file.endswith('.mp4'):
        extract_frames(video_path, label=1, label_name="correct_pushup", frame_interval=10, start_count=correct_frame_count)
        correct_frame_count += 1

for video_file in os.listdir(incorrect_dir):
    video_path = os.path.join(incorrect_dir, video_file)
    if os.path.isfile(video_path) and video_file.endswith('.mp4'):
        extract_frames(video_path, label=0, label_name="incorrect_pushup", frame_interval=10, start_count=incorrect_frame_count)
        incorrect_frame_count += 1

with open(train_annotations_path, 'w') as f:
    json.dump(train_annotations, f, indent=4)
with open(test_annotations_path, 'w') as f:
    json.dump(test_annotations, f, indent=4)
