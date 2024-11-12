import os
import logging
import shutil
import random
import json

from roboflow import Roboflow


_logger = logging.getLogger("fetch_datasets")
_logger.addHandler(logging.NullHandler())

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s() | %(message)s",
    level=logging.INFO,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_dir = os.path.join(project_root, "datasets")
os.makedirs(base_dir, exist_ok=True)


rf = Roboflow(api_key="<>")
# Do not download if the folder already exists
# Curls dataset
curl_dataset_dir = os.path.join(base_dir, "Dumbbell-Biceps-Curl-Detection-12")
if not os.path.exists(curl_dataset_dir):
    _logger.info("Downloading Dumbbell-Biceps-Curl-Detection-12 dataset")
    project = rf.workspace("humanarm").project("dumbbell-biceps-curl-detection")
    version = project.version(12)
    dataset = version.download("coco")
else:
    _logger.info("Dumbbell-Biceps-Curl-Detection-12 dataset already exists")

# Squats dataset
squat_dataset_dir = os.path.join(base_dir, "Squat_Classification-4")
if not os.path.exists(squat_dataset_dir):
    _logger.info("Downloading Squat_Classification-4 dataset")
    project = rf.workspace("machine-learning-uejfw").project("squat_classification-2kw41")
    version = project.version(4)
    dataset = version.download("coco")
else:
    _logger.info("Squat_Classification-4 dataset already exists")

# Walking dataset (non-exercise)
non_exercise_dataset_dir = os.path.join(base_dir, "stuff-3")
if not os.path.exists(non_exercise_dataset_dir):
    _logger.info("Downloading Non-Exercise dataset")
    project = rf.workspace("bobbythedog-kbnfg").project("stuff-q9gky")
    version = project.version(3)
    dataset = version.download("coco")
    _logger.info("Non-Exercise dataset downloaded.")
else:
    _logger.info("Non-Exercise dataset already exists.")


# Label mapping specific to bicep curls
label_mapping = {
    "correct": ["Dumbbell Bicep Curl"],
    "incorrect": ["incorrect"]
}

class_labels = {
    "correct_posture": 1,
    "incorrect_posture": 0
}

# Define organized directories
organized_train_dir = os.path.join(base_dir, "organized/train")
organized_test_dir = os.path.join(base_dir, "organized/test")
os.makedirs(organized_train_dir, exist_ok=True)
os.makedirs(organized_test_dir, exist_ok=True)

train_annotations = []
test_annotations = []

def get_image_to_category_mapping(ann_data):
    image_to_category = {}
    for annotation in ann_data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        image_to_category[image_id] = category_id
    return image_to_category

def prepare_curl_images(base_path, label_mapping, train_annotations, test_annotations, split_ratio=0.8):
    images_dir = os.path.join(base_path, "train")
    annotations_path = os.path.join(base_path, "train", "_annotations.coco.json")

    # Load annotations
    with open(annotations_path, "r") as f:
        ann_data = json.load(f)

    image_to_category = get_image_to_category_mapping(ann_data)

    class_labels_mapping = {
        cat: class_labels["correct_posture"] if cat in label_mapping["correct"] else class_labels["incorrect_posture"]
        for cat in label_mapping["correct"] + label_mapping["incorrect"]
    }

    # Filter images based on correct and incorrect labels only
    filtered_images = [
        img for img in ann_data["images"]
        if any(
            c["name"] in label_mapping["correct"] + label_mapping["incorrect"]
            for c in ann_data["categories"]
            if c["id"] == image_to_category.get(img["id"])
        )
    ]

    random.shuffle(filtered_images)
    # pick 500 images for training and 100 images for testing
    train_images = filtered_images[:500]
    test_images = filtered_images[500:600]
    
    """
    split_index = int(len(filtered_images) * split_ratio)
    train_images = filtered_images[:split_index]
    test_images = filtered_images[split_index:]
    """

    # Process images and annotations for train and test sets
    for img_set, img_data, annotations, dir_ in [
        ("train", train_images, train_annotations, organized_train_dir),
        ("test", test_images, test_annotations, organized_test_dir),
    ]:
        for img_info in img_data:
            category_id = image_to_category.get(img_info["id"], None)
            if category_id is None:
                continue

            # Determine label
            original_class_name = next((c["name"] for c in ann_data["categories"] if c["id"] == category_id), None)
            label = class_labels_mapping.get(original_class_name, class_labels["incorrect_posture"])

            # Copy image to the organized directory
            src_path = os.path.join(images_dir, img_info["file_name"])
            dest_path = os.path.join(dir_, img_info["file_name"])
            shutil.copy(src_path, dest_path)

            # Append annotation
            annotations.append({
                "file_name": img_info["file_name"],
                "label": label
            })

_logger.info("Processing bicep curls for correct and incorrect postures")
prepare_curl_images(curl_dataset_dir, label_mapping, train_annotations, test_annotations, split_ratio=0.8)
_logger.info("Finished processing bicep curls")

train_annotations_file = os.path.join(organized_train_dir, "annotations.json")
test_annotations_file = os.path.join(organized_test_dir, "annotations.json")

with open(train_annotations_file, "w") as f:
    json.dump(train_annotations, f, indent=4)
with open(test_annotations_file, "w") as f:
    json.dump(test_annotations, f, indent=4)
