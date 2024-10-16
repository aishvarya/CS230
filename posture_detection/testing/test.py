import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from posture_detection.data_loader.coco_loader import CocoDataset
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
import yaml
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

# Load config
with open("posture_detection/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test datasets
test_squat_dataset = CocoDataset(
    annotations_file=config['test_squat_annotations'],
    img_dir=config['test_squat_img_dir'],
    transform=transform,
    exercise_type="squat",
    label_mapping=config['label_mapping']
)

test_curl_dataset = CocoDataset(
    annotations_file=config['test_curl_annotations'],
    img_dir=config['test_curl_img_dir'],
    transform=transform,
    exercise_type="curl",
    label_mapping=config['label_mapping']
)

# Combine test datasets and create DataLoader
test_combined_dataset = ConcatDataset([test_squat_dataset, test_curl_dataset])
test_loader = DataLoader(test_combined_dataset, batch_size=config['batch_size'], shuffle=False)

# Load the model
model = HybridCNNWithAttention()
model.load_state_dict(torch.load(config['model_output'], weights_only=True))
model.eval()

_logger.info(f"Total number of test samples: {len(test_combined_dataset)}")

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            labels = labels.float().unsqueeze(1).squeeze(-1)
            outputs = model(images)
            predicted = (outputs >= 0.5).float()  # Binary threshold at 0.5
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    _logger.info(f"Test Accuracy: {accuracy:.2f}%")

# Evaluate the model
evaluate_model(model, test_loader)
