import pandas as pd
import json
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to files
INPUT_FILE = '/home/ubuntu/v2/yoga/csv_files/top_10_poses_combined_data.csv'  # Your CSV file with training data
OUTPUT_JSON_FILE = '/home/ubuntu/v2/yoga/reference_poses.json'

# Load the dataset
logging.info(f"Loading dataset from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit()

# Extract the top 10 poses and average their landmark positions
logging.info("Calculating average landmark positions for each pose...")

# Dictionary to hold average landmark data for each pose
reference_poses = {}

# Grouping by 'l3_pose' to calculate average landmark positions for each pose
poses = df['l3_pose'].unique()
for pose_name in tqdm(poses, desc="Processing poses"):
    pose_data = df[df['l3_pose'] == pose_name]

    # Average the x, y, z coordinates for each landmark_index
    avg_landmarks = pose_data.groupby('landmark_index')[['x', 'y', 'z']].mean().reset_index()
    
    # Convert to a dictionary format
    landmark_positions = avg_landmarks[['landmark_index', 'x', 'y', 'z']].to_dict(orient='records')
    reference_poses[pose_name] = landmark_positions

# Save the average landmark positions to a JSON file
logging.info(f"Saving reference pose data to: {OUTPUT_JSON_FILE}")
try:
    with open(OUTPUT_JSON_FILE, 'w') as json_file:
        json.dump(reference_poses, json_file, indent=4)
    logging.info("Reference pose data saved successfully.")
except Exception as e:
    logging.error(f"Failed to save reference pose data: {e}")
    exit()
