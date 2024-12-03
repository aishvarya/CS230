import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to files
COMBINED_CSV_PATH = '/home/ubuntu/v2/yoga/csv_files/extracted_skeletal_data_with_sequence.csv'
POSE_INDEX_PATH = '/home/ubuntu/v2/yoga/csv_files/3dyoga90_landmarks/3DYoga90.csv'
OUTPUT_FILTERED_CSV_PATH = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'

# Load the skeletal dataset
logging.info("Loading skeletal dataset from '%s'...", COMBINED_CSV_PATH)
combined_df = pd.read_csv(COMBINED_CSV_PATH)

# Load the pose index dataset
logging.info("Loading pose index dataset from '%s'...", POSE_INDEX_PATH)
pose_index_df = pd.read_csv(POSE_INDEX_PATH)

# Merge the two datasets using the common column 'sequence_id'
logging.info("Merging skeletal data with pose data using 'sequence_id'...")
merged_df = pd.merge(combined_df, pose_index_df, on='sequence_id', how='inner')

# Verifying the available columns and identifying top poses
logging.info("Available columns in merged dataset: %s", merged_df.columns.tolist())

# Analyze pose frequencies to identify the top 20 most common level 3 poses
try:
    logging.info("Analyzing pose frequencies to identify the top 20 level 3 poses...")
    pose_frequencies = merged_df['l3_pose'].value_counts()
    top_20_poses = pose_frequencies.head(20).index.tolist()
    logging.info("Top 20 poses identified: %s", top_20_poses)
except KeyError as e:
    logging.error("Column 'l3_pose' not found in the merged dataset. Available columns are: %s", merged_df.columns.tolist())
    raise e

# Filter the dataset to include only rows with the top 20 poses
logging.info("Filtering dataset to include only the top 20 poses...")
filtered_df = merged_df[merged_df['l3_pose'].isin(top_20_poses)]

# Display a few rows of the filtered dataset to verify correctness
logging.info("Displaying the first few rows of the filtered dataset:")
print(filtered_df.head(10))

# Adding a progress bar while saving the filtered dataset
logging.info("Saving filtered dataset with top 20 poses to '%s'...", OUTPUT_FILTERED_CSV_PATH)
with open(OUTPUT_FILTERED_CSV_PATH, mode='w') as output_csv:
    # Write the header only once
    header_written = False
    chunk_size = 5000

    num_chunks = len(filtered_df) // chunk_size + 1

    for i in tqdm(range(num_chunks), desc="Saving to CSV"):
        chunk = filtered_df[i * chunk_size: (i + 1) * chunk_size]
        if chunk.empty:
            continue

        if not header_written:
            chunk.to_csv(output_csv, index=False)
            header_written = True
        else:
            chunk.to_csv(output_csv, index=False, header=False)

logging.info("Filtering complete. Filtered dataset saved successfully.")
