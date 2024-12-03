import joblib
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
BASE_DIR = '/home/ubuntu/v2/yoga'
OLD_ENCODER_FILE = os.path.join(BASE_DIR, 'models/label_encoder.pkl')
NEW_ENCODER_FILE = os.path.join(BASE_DIR, 'models/label_encoder_fixed.pkl')
MODEL_FILE = os.path.join(BASE_DIR, 'models/3dyoga90_pose_model.keras')
TRAINING_DATA = os.path.join(BASE_DIR, 'csv_files/top_20_poses_combined_data.csv')

# Correct order from training data
CORRECT_ORDER = [
    'staff',
    'reclining-hero',
    'cockerel',
    'downward-dog',
    'frog',
    'reclining-cobbler',
    'wind-relieving',
    'shoulder-pressing',
    'balancing-table',
    'tree'
]

def verify_training_data():
    """Verify training data matches our assumptions"""
    logging.info("Verifying training data...")
    df = pd.read_csv(TRAINING_DATA)
    
    # Get pose frequencies
    pose_counts = df['l3_pose'].value_counts()
    top_10_poses = pose_counts.head(10)
    
    # Verify order matches
    actual_order = top_10_poses.index.tolist()
    order_correct = actual_order == CORRECT_ORDER
    
    logging.info("\nTraining Data Verification:")
    logging.info(f"Order matches expected: {order_correct}")
    
    if not order_correct:
        logging.warning("Mismatched poses:")
        for i, (expected, actual) in enumerate(zip(CORRECT_ORDER, actual_order)):
            if expected != actual:
                logging.warning(f"Position {i}: Expected '{expected}', Found '{actual}'")
    
    return order_correct

def verify_model_compatibility():
    """Verify model output layer matches number of poses"""
    logging.info("\nVerifying model compatibility...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Check output layer
    output_units = model.layers[-1].units
    logging.info(f"Model output units: {output_units}")
    logging.info(f"Number of poses: {len(CORRECT_ORDER)}")
    
    return output_units == len(CORRECT_ORDER)

def create_and_verify_encoder():
    """Create new encoder and verify its functionality"""
    # Create new encoder
    new_encoder = LabelEncoder()
    new_encoder.fit(CORRECT_ORDER)
    
    # Verify encoding/decoding
    test_indices = np.arange(len(CORRECT_ORDER))
    decoded_poses = new_encoder.inverse_transform(test_indices)
    
    logging.info("\nEncoder Verification:")
    for idx, pose in zip(test_indices, decoded_poses):
        logging.info(f"Index {idx} -> {pose}")
    
    return new_encoder

def save_encoder(encoder, verify_mapping=True):
    """Save encoder with verification step"""
    # Backup old encoder
    if os.path.exists(OLD_ENCODER_FILE):
        backup_file = OLD_ENCODER_FILE + '.backup'
        os.rename(OLD_ENCODER_FILE, backup_file)
        logging.info(f"\nOld encoder backed up to: {backup_file}")
    
    # Save new encoder
    joblib.dump(encoder, OLD_ENCODER_FILE)
    logging.info(f"New encoder saved to: {OLD_ENCODER_FILE}")
    
    # Verify saved encoder
    if verify_mapping:
        loaded_encoder = joblib.load(OLD_ENCODER_FILE)
        verification_indices = np.arange(len(CORRECT_ORDER))
        loaded_poses = loaded_encoder.inverse_transform(verification_indices)
        original_poses = encoder.inverse_transform(verification_indices)
        
        matches = np.array_equal(loaded_poses, original_poses)
        logging.info(f"\nSaved encoder verification: {'Passed' if matches else 'Failed'}")

def main():
    logging.info("Starting encoder verification and creation process...")
    
    # Step 1: Verify training data
    if not verify_training_data():
        logging.error("Training data verification failed!")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    # Step 2: Verify model compatibility
    if not verify_model_compatibility():
        logging.error("Model compatibility check failed!")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    # Step 3: Create and verify new encoder
    logging.info("\nCreating new encoder...")
    new_encoder = create_and_verify_encoder()
    
    # Step 4: Save encoder
    logging.info("\nSaving encoder...")
    save_encoder(new_encoder)
    
    logging.info("\nProcess complete!")

if __name__ == "__main__":
    main()