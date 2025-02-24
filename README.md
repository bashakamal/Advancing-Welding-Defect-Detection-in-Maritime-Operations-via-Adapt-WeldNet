# Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis

## Overview
This repository contains a PyTorch-based deep learning pipeline for image classification. It supports various pre-trained models, transfer learning strategies, hyperparameter optimization with Optuna, and early stopping to prevent overfitting.

## Features
- Supports multiple pre-trained models: ResNet, EfficientNet, DenseNet, MobileNet, and more.
- Different transfer learning modes:
  - **Transfer Learning**: Freezes all but the classifier layers.
  - **Freeze Early Layers**: Freezes initial layers and fine-tunes later ones.
  - **Fine-Tune All Layers**: Allows full network training.
- Hyperparameter tuning using **Optuna**.
- Multiple optimizer options: Adam, SGD, RMSprop, AdamW, and more.
- Implements **Early Stopping** to prevent overfitting.
- Saves the best model checkpoint based on validation performance.

## Installation
### Prerequisites
Ensure you have Python 3.7+ and PyTorch installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `torch`
- `torchvision`
- `optuna`
- `pandas`
- `scikit-learn`

## Usage

### 1. Load and Train a Model

```python
from torchvision import models
from train import train_and_evaluate, load_model

model = load_model('resnet18', num_classes=10)
best_val_acc, logs = train_and_evaluate(model, mode='transfer_learning', num_classes=10, train_loader, val_loader, device='cuda', patience=5, lr=0.001, optimizer_type='adam')
```

### 2. Transfer Learning Modes
You can configure the model using the `setup_mode` function:

```python
setup_mode(model, mode="fine_tune_all")  # Options: transfer_learning, freeze_early_layers, fine_tune_all
```

### 3. Hyperparameter Optimization with Optuna
You can use Optuna to find the best hyperparameters:

```python
import optuna
from optuna_integration import objective

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 4. Supported Models
- ResNet (`resnet18`)
- EfficientNet (`efficientnet_b0`, `efficientnet_v2_s`)
- MobileNet (`mobilenet_v2`, `mobilenet_v3_small`)
- DenseNet (`densenet121`)
- SqueezeNet (`squeezenet1_0`)
- ShuffleNet (`shufflenet_v2_x0_5`)
- Wide ResNet (`wide_resnet50_2`)
- Swin Transformer (`swin_t`)

### 5. Optimizer Choices
- `adam`
- `adamw`
- `adamax`
- `sgd`
- `rmsprop`
- `nadam`
- `adadelta`
- `adagrad`

## Training Logs
All training logs are saved in CSV format for analysis.

## Model Checkpointing
The best model is saved based on validation loss using early stopping.

## Contributing
Feel free to contribute by improving model selection, adding more optimization techniques, or extending dataset compatibility.

## License
This project is open-source under the MIT License.

