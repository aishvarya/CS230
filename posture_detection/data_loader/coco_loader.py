import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, exercise_type=None, label_mapping=None):
        with open(annotations_file) as f:
            coco_data = json.load(f)
        self.annotations = coco_data["annotations"]
        self.image_id_to_file_name = {img["id"]: img["file_name"] for img in coco_data["images"]}
        self.img_dir = img_dir
        self.transform = transform
        self.exercise_type = exercise_type
        self.correct_label = label_mapping[exercise_type]["correct_label"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the annotation based on index
        ann = self.annotations[idx]
        
        # Retrieve the image file name and path
        img_file_name = self.image_id_to_file_name[ann['image_id']]
        img_path = os.path.join(self.img_dir, img_file_name)
        image = Image.open(img_path).convert("RGB")
        
        # Original label and binary mapping
        original_label = ann['category_id']
        label = 1 if original_label == self.correct_label else 0

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor([label], dtype=torch.float32)  # Float for BCELoss compatibility

        return image, label
