import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from posture_detection.data_loader.coco_loader import CocoDataset
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
from tqdm import tqdm
import yaml
import logging
import random


_logger = logging.getLogger("train")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

# Load config
with open("posture_detection/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define common transformations for both datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

squat_dataset = CocoDataset(
    annotations_file=config['squat_annotations'],
    img_dir=config['squat_img_dir'],
    transform=transform,
    exercise_type="squat",
    label_mapping=config['label_mapping']
)

curl_dataset = CocoDataset(
    annotations_file=config['curl_annotations'],
    img_dir=config['curl_img_dir'],
    transform=transform,
    exercise_type="curl",
    label_mapping=config['label_mapping']
)

# Combine datasets
combined_dataset = ConcatDataset([squat_dataset, curl_dataset])

# Following code is added to sample a small percentage of the dataset
# to speed up training for testing purposes. Remove this code for
# training on the entire dataset.
sample_percentage = 0.01
num_samples = int(len(combined_dataset) * sample_percentage)
sample_indices = random.sample(range(len(combined_dataset)), num_samples)
combined_dataset = Subset(combined_dataset, sample_indices)

# DataLoader for the combined dataset
train_loader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True)

_logger.info(f"Number of samples in the dataset: {len(combined_dataset)}")

# Model, Loss, and Optimizer
model = HybridCNNWithAttention()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training Loop
for epoch in tqdm(range(config['epochs']), desc="Training"):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1).squeeze(-1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    _logger.info(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), config['model_output'])
