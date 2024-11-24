import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from posture_detection.data_loader.data_loader import CustomPostureDataset
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
import yaml
import os
import logging
from tqdm import tqdm
import optuna

_logger = logging.getLogger("hyperparameter_tuning")
_logger.addHandler(logging.NullHandler())
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

with open("posture_detection/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_test_dataset():
    test_dataset = CustomPostureDataset(
        annotations_file=os.path.join(config['test_dir'], "annotations.json"),
        img_dir=config['test_dir'],
        transform=transform
    )
    return test_dataset

def initialize_model(num_classes):
    model = HybridCNNWithAttention(num_classes=num_classes)
    return model

def evaluate_model(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)  # Multiclass classification
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    overall_accuracy = 100 * correct / total
    _logger.info(f"Overall Test Accuracy: {overall_accuracy:.2f}%")

    return overall_accuracy

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = initialize_model(num_classes=config['num_classes'])
    model.load_state_dict(torch.load(config['model_output']))
    model.eval()

    accuracy = evaluate_model(model, test_loader)
    return accuracy

def run_hyperparameter_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    _logger.info("Best hyperparameters:")
    _logger.info(study.best_params)
    _logger.info(f"Best test accuracy: {study.best_value:.2f}%")

def main():
    test_dataset = load_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = initialize_model(num_classes=config['num_classes'])
    model.load_state_dict(torch.load(config['model_output']))
    model.eval()

    _logger.info(f"Total number of test samples: {len(test_dataset)}")

    evaluate_model(model, test_loader)

    _logger.info("Starting hyperparameter tuning...")
    run_hyperparameter_tuning()

if __name__ == "__main__":
    main()
