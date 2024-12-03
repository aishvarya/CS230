import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
DATA_FILE = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
MODEL_FILE = '/home/ubuntu/v2/yoga/models/3dyoga90_pose_model.keras'

# Our top 10 poses from implementation
TOP_10_POSES = [
    'tree', 
    'downward-dog', 
    'staff', 
    'shoulder-pressing', 
    'balancing-table', 
    'reclining-hero', 
    'cockerel', 
    'frog', 
    'reclining-cobbler', 
    'wind-relieving'
]

def evaluate_model():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Filter for top 10 poses
    df = df[df['l3_pose'].isin(TOP_10_POSES)]
    print(f"Data filtered to top 10 poses. Shape: {df.shape}")

    print("Preparing features and labels...")
    features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]
    labels = df['l3_pose']

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print("\nPose distribution:")
    for i, pose in enumerate(label_encoder.classes_):
        count = len(labels[labels == pose])
        print(f"{pose}: {count} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, test_size=0.2, random_state=42
    )

    print("\nMaking predictions...")
    model = tf.keras.models.load_model(MODEL_FILE)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_))

if __name__ == "__main__":
    evaluate_model()

