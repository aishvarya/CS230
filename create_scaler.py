import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_scaler_from_training_logic():
    """Create a scaler that matches the original training data format"""
    # File paths
    DATA_FILE = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
    NEW_SCALER_FILE = '/home/ubuntu/v2/yoga/models/new_scaler.pkl'

    # Load dataset
    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    logging.info(f"Dataset loaded with shape: {df.shape}")

    # Filter dataset to include only top 10 Level 3 poses
    logging.info("Filtering dataset to include only the top 10 Level 3 poses...")
    top_10_poses = df['l3_pose'].value_counts().index[:10]
    df = df[df['l3_pose'].isin(top_10_poses)]
    logging.info(f"Filtered dataset shape: {df.shape}")

    # Prepare features exactly as in training
    logging.info("Preparing features...")
    features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]

    # Create and fit new scaler
    scaler = StandardScaler()
    scaler.fit(features)

    # Save the new scaler
    joblib.dump(scaler, NEW_SCALER_FILE)
    logging.info(f"New scaler saved to: {NEW_SCALER_FILE}")

    # Print scaler information
    logging.info("\nScaler Information:")
    logging.info(f"Number of features: {len(scaler.mean_)}")
    logging.info(f"Feature means: {scaler.mean_}")
    logging.info(f"Feature scales: {scaler.scale_}")

    return scaler

if __name__ == "__main__":
    scaler = create_scaler_from_training_logic()