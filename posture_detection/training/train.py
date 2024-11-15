import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
from posture_detection.data_loader.data_loader import CustomPostureDataset
from tqdm import tqdm
import yaml
import logging
import json

# Set up logging
_logger = logging.getLogger("train")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

# Load configuration
with open("posture_detection/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load training dataset
train_dataset = CustomPostureDataset(
    annotations_file=os.path.join(config['train_dir'], "annotations.json"),
    img_dir=config['train_dir'],
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
_logger.info(f"Number of samples in training dataset: {len(train_dataset)}")

# Define the model, loss function, and optimizer
model = HybridCNNWithAttention() 
criterion = torch.nn.BCEWithLogitsLoss()  # Combines sigmoid activation with BCE loss
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
for epoch in tqdm(range(config['epochs']), desc="Training"):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)  # (batch_size x 1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = (outputs >= 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct_predictions / total_predictions
    _logger.info(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), config['model_output'])
