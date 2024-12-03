# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import joblib
# import logging
# from tqdm import tqdm

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # File paths
# DATA_FILE = '/home/ubuntu/v2/yoga/csv_files/top_20_poses_combined_data.csv'
# MODEL_FILE = '/home/ubuntu/v2/yoga/models/3dyoga90_pose_model.keras'
# SCALER_FILE = '/home/ubuntu/v2/yoga/models/scaler.pkl'
# ENCODER_FILE = '/home/ubuntu/v2/yoga/models/label_encoder.pkl'

# # Load dataset
# logging.info("Loading dataset...")
# df = pd.read_csv(DATA_FILE)
# logging.info(f"Dataset loaded with shape: {df.shape}")

# # Filter dataset to include only top 10 Level 3 poses
# logging.info("Filtering dataset to include only the top 10 Level 3 poses...")
# top_10_poses = df['l3_pose'].value_counts().index[:10]
# df = df[df['l3_pose'].isin(top_10_poses)]
# logging.info(f"Filtered dataset shape: {df.shape}")

# # Prepare features and labels
# logging.info("Preparing features and labels...")
# features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]
# labels = df['l3_pose']

# # Encode labels
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)
# logging.info("Labels encoded.")

# # Normalize features
# scaler = StandardScaler()
# features_normalized = scaler.fit_transform(features)
# logging.info("Features normalized.")

# # Split dataset
# logging.info("Splitting dataset into training and testing sets...")
# X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels_encoded, test_size=0.2, random_state=42)
# logging.info(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# # Define the model
# logging.info("Defining the model...")
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(len(label_encoder.classes_), activation='softmax')
# ])

# # Compile the model
# logging.info("Compiling the model...")
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model with progress bar
# logging.info("Training the model...")
# for epoch in tqdm(range(50), desc="Training Progress"):
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64, verbose=0)
# logging.info("Model training completed.")

# # Evaluate the model
# logging.info("Evaluating the model...")
# loss, accuracy = model.evaluate(X_test, y_test)
# logging.info(f"Model accuracy: {accuracy:.4f}")

# # Save the model, scaler, and label encoder
# logging.info("Saving the model, scaler, and label encoder...")
# model.save(MODEL_FILE)  # Save using default Keras format
# joblib.dump(scaler, SCALER_FILE)
# joblib.dump(label_encoder, ENCODER_FILE)
# logging.info("Model, scaler, and label encoder saved successfully.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import joblib
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = '/home/ubuntu/v2/yoga'
DATA_FILE = os.path.join(BASE_DIR, 'csv_files/top_20_poses_combined_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_PREFIX = f'yoga_model_top10_{timestamp}'

MAX_SAMPLES_PER_POSE = 100000

def load_and_process_data():
    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    pose_counts = df['l3_pose'].value_counts()
    top_10_poses = pose_counts.nlargest(10).index
    logging.info("\nTop 10 poses selected:")
    for pose in top_10_poses:
        logging.info(f"{pose}: {pose_counts[pose]} samples")
    
    df = df[df['l3_pose'].isin(top_10_poses)]
    
    # Sample data for each pose
    df_sampled = df.groupby('l3_pose').apply(
        lambda x: x.sample(n=min(len(x), MAX_SAMPLES_PER_POSE), random_state=42)
    ).reset_index(drop=True)
    
    # Calculate class weights
    class_counts = df_sampled['l3_pose'].value_counts()
    class_weights = dict(zip(
        range(len(class_counts)),
        len(df_sampled) / (len(class_counts) * class_counts)
    ))
    
    logging.info("\nAfter sampling:")
    for pose in top_10_poses:
        count = len(df_sampled[df_sampled['l3_pose'] == pose])
        logging.info(f"{pose}: {count} samples")
    
    return df_sampled, class_weights

def create_model(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    df, class_weights = load_and_process_data()
    
    features = df[['x', 'y', 'z', 'landmark_index', 'sequence_id']]
    labels = df['l3_pose']
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_normalized, 
        labels_encoded, 
        test_size=0.2, 
        stratify=labels_encoded,
        random_state=42
    )
    
    model = create_model(X_train.shape[1], len(label_encoder.classes_))
    
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f'{MODEL_PREFIX}.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_encoder.pkl'))
    pd.DataFrame(history.history).to_csv(os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_history.csv'))

if __name__ == "__main__":
    main()