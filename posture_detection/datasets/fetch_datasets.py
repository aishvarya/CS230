import os
import logging

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


# Instantiate the Roboflow class with your API key.
rf = Roboflow(api_key="YOUR_API_KEY")

# Roboflow

# Do not download if the folder already exists
# Curls dataset
curl_dataset_dir = os.path.join(base_dir, "Dumbbell-Biceps-Curl-Detection-12")
print(curl_dataset_dir)
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
