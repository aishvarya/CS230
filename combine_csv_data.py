import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
POSES_DATA_PATH = '/home/ubuntu/v2/yoga/csv_files/all_poses_data_pose.csv'
ANGLES_DATA_PATH = '/home/ubuntu/v2/yoga/csv_files/4_angles_poses_angles.csv'
POSE_INDEX_PATH = '/home/ubuntu/v2/yoga/csv_files/3dyoga90_landmarks/pose-index.csv'
OUTPUT_PATH = '/home/ubuntu/v2/yoga/csv_files/combined_poses_data.csv'

CHUNK_SIZE = 10000  # Adjust chunk size based on system memory capacity

# Load pose index data (small enough to load fully)
logging.info(f"Loading pose index data from '{POSE_INDEX_PATH}'...")
try:
    pose_index_data = pd.read_csv(POSE_INDEX_PATH)
    logging.info(f"Pose index data loaded with shape: {pose_index_data.shape}")
except Exception as e:
    logging.error(f"Failed to load pose index data: {e}")
    exit()

# Load angles data (small enough to load fully)
logging.info(f"Loading angles data from '{ANGLES_DATA_PATH}'...")
try:
    angles_data = pd.read_csv(ANGLES_DATA_PATH)
    logging.info(f"Angles data loaded with shape: {angles_data.shape}")

    # Verify if 'row_id' column exists, if not, create it
    if 'row_id' not in angles_data.columns:
        logging.warning("'row_id' column is missing in angles data. Creating 'row_id'...")
        angles_data['row_id'] = [f"{i}-pose-{i}" for i in range(len(angles_data))]  # Ensuring row_id matches original format
        logging.info("'row_id' column created in angles data.")
    
    # Convert 'row_id' in angles_data to string type
    angles_data['row_id'] = angles_data['row_id'].astype(str)

except Exception as e:
    logging.error(f"Failed to load angles data: {e}")
    exit()

# Process pose data in chunks
logging.info(f"Loading pose data in chunks from '{POSES_DATA_PATH}'...")
chunk_list = []

try:
    for chunk in pd.read_csv(POSES_DATA_PATH, chunksize=CHUNK_SIZE):
        logging.info(f"Processing chunk with shape: {chunk.shape}")

        # Filter out rows where 'l3_pose_id' is NaN
        chunk = chunk[chunk['l3_pose_id'].notna()]
        if chunk.empty:
            logging.warning("Chunk has no valid 'l3_pose_id' values. Skipping this chunk.")
            continue

        # Ensure 'row_id' in pose chunk is of type string to match angles data
        chunk['row_id'] = chunk['row_id'].astype(str)

        # Merge current chunk with pose index data (inner join to ensure only matching l3_pose_id)
        merged_chunk = pd.merge(chunk, pose_index_data, how='inner', on='l3_pose_id')
        logging.info(f"Merged chunk with pose index data, resulting shape: {merged_chunk.shape}")

        # Log the rows that did not match to understand missing l3_pose_id values
        unmatched_rows = chunk[~chunk['l3_pose_id'].isin(pose_index_data['l3_pose_id'])]
        logging.warning(f"Number of unmatched rows based on 'l3_pose_id': {len(unmatched_rows)}")

        # Merge with angles data using 'row_id' (left join to retain all pose data)
        merged_chunk = pd.merge(merged_chunk, angles_data, how='left', on='row_id')
        logging.info(f"Chunk after merging with angles data, resulting shape: {merged_chunk.shape}")

        # Handle missing values in angle columns (e.g., fill with mean or set to a specific value)
        angle_columns = ['armpit_left', 'armpit_right', 'elbow_left', 'elbow_right',
                         'hip_left', 'hip_right', 'knee_left', 'knee_right',
                         'ankle_left', 'ankle_right']
        for col in angle_columns:
            if col in merged_chunk.columns:
                merged_chunk[col].fillna(merged_chunk[col].mean(), inplace=True)
                logging.info(f"Filled missing values in column '{col}' with mean value.")

        # Append processed chunk to list
        chunk_list.append(merged_chunk)

    # Concatenate all chunks into a final DataFrame
    if chunk_list:
        final_data = pd.concat(chunk_list, axis=0)
        logging.info(f"Final concatenated data shape: {final_data.shape}")

        # Save final combined dataset
        logging.info(f"Saving final combined dataset to '{OUTPUT_PATH}'...")
        final_data.to_csv(OUTPUT_PATH, index=False)
        logging.info("Final combined dataset saved successfully.")
    else:
        logging.error("No valid data chunks processed. Please check input files for completeness.")

except Exception as e:
    logging.error(f"An error occurred while processing pose data in chunks: {e}")
    exit()

logging.info("Data combination process completed.")
