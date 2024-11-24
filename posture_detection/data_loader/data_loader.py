import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomPostureDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = img_info['label']
        subclass = img_info['subclass']
        return image, label, subclass
