import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from posture_detection.data_loader.data_loader import CustomPostureDataset
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
import yaml
import os
import logging
from tqdm import tqdm

# Set up logging
_logger = logging.getLogger("test")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(funcName)s() | %(message)s",
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

# Load test dataset
test_dataset = CustomPostureDataset(
    annotations_file=os.path.join(config['test_dir'], "annotations.json"),
    img_dir=config['test_dir'],
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize model
model = HybridCNNWithAttention()
model.load_state_dict(torch.load(config['model_output']))
model.eval()

_logger.info(f"Total number of test samples: {len(test_dataset)}")

# Define evaluation function
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images)
            predicted = (outputs >= 0.5).float().squeeze(1)  # Binary threshold at 0.5
            _logger.info(f"Predicted: {predicted}")
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    _logger.info(f"Test Accuracy: {accuracy:.2f}%")

# Evaluate the model
evaluate_model(model, test_loader)
