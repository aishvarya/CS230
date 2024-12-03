# CS230


# Posture Detection Project

This folder contains code for the development of a hybrid CNN model with attention for the classification of pushup postures. 

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

## Setup

```bash
poetry install
```

Assuming the dataset is already setup.

## Training the model
```bash
poetry run python posture_detection/training/train.py
```

# Testing the model
```bash
poetry run python posture_detection/testing/test.py
```


# Posture Detection with Media Pipe inputs Project

This folder contains code for the development of a hybrid CNN model with attention for the classification of pushup postures with mediapipe features used as additional inputs to the training of the model.  

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

## Setup

```bash
poetry install
```

Assuming the dataset is already setup.

## Training the model
```bash
poetry run python posture_detection/training/train.py
```

# Testing the model
```bash
poetry run python posture_detection/testing/test.py
```
