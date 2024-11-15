import torch
from torchvision import transforms
from PIL import Image
from posture_detection.models.hybrid_cnn_with_attention import HybridCNNWithAttention
import yaml

# Load configuration
with open("posture_detection/training/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize model
model = HybridCNNWithAttention()
model.load_state_dict(torch.load(config['model_output']))
model.eval()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Count parameters
total_params, trainable_params = count_parameters(model)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Define transformations (same as used during training/testing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output >= 0.5).float().item() 
    
    return "Correct Posture" if prediction == 1 else "Incorrect Posture"

ic_1 = "/Users/aishvaryasingh/Downloads/IMG_6958.JPG" 
c1 = "/Users/aishvaryasingh/Downloads/correct_3.jpg" 
result = predict_image(c1, model)
print(f"Prediction for {c1}: {result}")
