# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# # Paths
# DATA_FILE = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
# MODEL_FILE = '/home/ubuntu/v2/yoga/models/3dyoga90_pose_model.keras'

# # Our top 10 poses
# TOP_10_POSES = [
#     'tree', 'downward-dog', 'staff', 'shoulder-pressing', 
#     'balancing-table', 'reclining-hero', 'cockerel', 
#     'frog', 'reclining-cobbler', 'wind-relieving'
# ]

# def create_visualizations(model, X_test, y_test, y_pred_classes, label_encoder):
#     # Create confusion matrix
#     cm = confusion_matrix(y_test, y_pred_classes)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     plt.close()

#     print("Confusion matrix saved as 'confusion_matrix.png'")

# def evaluate_and_visualize():
#     print("Loading data...")
#     df = pd.read_csv(DATA_FILE)
    
#     # Filter for top 10 poses
#     df = df[df['l3_pose'].isin(TOP_10_POSES)]
#     print(f"Data filtered to top 10 poses. Shape: {df.shape}")

#     print("Preparing features and labels...")
#     features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]
#     labels = df['l3_pose']

#     # Standardize features
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     # Encode labels
#     label_encoder = LabelEncoder()
#     labels_encoded = label_encoder.fit_transform(labels)
    
#     print("\nPose distribution:")
#     for i, pose in enumerate(label_encoder.classes_):
#         count = len(labels[labels == pose])
#         print(f"{pose}: {count} samples")

#     X_train, X_test, y_train, y_test = train_test_split(
#         features_scaled, labels_encoded, test_size=0.2, random_state=42
#     )

#     print("\nLoading model and making predictions...")
#     model = tf.keras.models.load_model(MODEL_FILE)
#     y_pred = model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)

#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred_classes, 
#                               target_names=label_encoder.classes_))

#     print("\nGenerating visualizations...")
#     create_visualizations(model, X_test, y_test, y_pred_classes, label_encoder)

# if __name__ == "__main__":
#     evaluate_and_visualize()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of poses
def plot_pose_distribution(labels, save_path='pose_distribution.png'):
    plt.figure(figsize=(12, 6))
    labels.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Yoga Poses in Dataset')
    plt.savefig(save_path)
    plt.close()

# Precision/Recall Bar Chart
def plot_metrics_comparison(precision_scores, recall_scores, poses, save_path='metrics_comparison.png'):
    plt.figure(figsize=(15, 6))
    x = np.arange(len(poses))
    width = 0.35

    plt.bar(x - width/2, precision_scores, width, label='Precision')
    plt.bar(x + width/2, recall_scores, width, label='Recall')
    plt.xticks(x, poses, rotation=45)
    plt.ylabel('Score')
    plt.title('Precision and Recall by Pose')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()