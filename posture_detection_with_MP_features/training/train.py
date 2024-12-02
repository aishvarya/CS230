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

_logger = logging.getLogger("train")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

with open("mp/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CustomPostureDataset(
    annotations_file=os.path.join(config['train_dir'], "annotations.json"),
    img_dir=config['train_dir'],
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
_logger.info(f"Number of samples in training dataset: {len(train_dataset)}")

num_classes = len(config['label_mapping'])
model = HybridCNNWithAttention(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for epoch in tqdm(range(config['epochs']), desc="Training"):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels, subclasses in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    _logger.info(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), config['model_output'])
