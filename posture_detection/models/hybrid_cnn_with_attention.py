import torch
import torch.nn as nn
from resnet_pytorch import ResNet

class HybridCNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNWithAttention, self).__init__()
        self.cnn = ResNet.from_pretrained("resnet18")
        # Remove the last two layers of the ResNet model.
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.aux_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.cnn(x)

        attention_weights = self.attention(features)
        # Add attention weights layer.
        # attended_features = features * attention_weights
        # Add residual connection in addition to attention weights.
        attention_features = features + (features * attention_weights)

        pooled_features = torch.mean(attention_features, dim=[2, 3])
        output = self.fc(pooled_features)

        return output
