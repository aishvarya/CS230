import torch
import torch.nn as nn
from resnet_pytorch import ResNet


class HybridCNNWithAttention(nn.Module):
    def __init__(self, num_classes=1):
        super(HybridCNNWithAttention, self).__init__()
        self.cnn = ResNet.from_pretrained("resnet18")
        # Remove the last two layers of the ResNet model.
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.cnn(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        pooled_features = torch.mean(attended_features, dim=[2, 3])
        output = self.fc(pooled_features)
        return output
