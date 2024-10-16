# CS230
# Posture Detection Project

TODO

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

## Setup

```bash
poetry install
```

## Downloading the training data
Set your roboflow API key in `posture_detection/datasets/fetch_datasets.py` before running the following
```bash
poetry run python posture_detection/datasets/fetch_datasets.py
```

## Training the model
```bash
poetry run python posture_detection/training/train.py
```

# Testing the model
```bash
poetry run python posture_detection/testing/test.py
```