import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from posture_detection.data_loader.data_loader import CustomPostureDataset
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
import yaml
import os
import logging
from tqdm import tqdm

_logger = logging.getLogger("test")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

with open("mp/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = CustomPostureDataset(
    annotations_file=os.path.join(config['test_dir'], "annotations.json"),
    img_dir=config['test_dir'],
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

num_classes = len(config['label_mapping'])
model = HybridCNNWithAttention(num_classes=num_classes)
model.load_state_dict(torch.load(config['model_output']))
model.eval()

_logger.info(f"Total number of test samples: {len(test_dataset)}")

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels, subclasses in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)  # Get predicted class index
            #_logger.info(f"Predicted: {predicted}, Actual: {labels}")

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    accuracy = 100 * correct / total
    _logger.info(f"Overall Test Accuracy: {accuracy:.2f}%")


evaluate_model(model, test_loader)
print(config['model_output'])
