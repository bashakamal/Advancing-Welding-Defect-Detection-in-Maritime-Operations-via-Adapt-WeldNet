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

# Explainable AI (XAI) Metrics for Welding Defect Detection

## Introduction
This repository includes various deep learning models for welding defect detection using explainable AI (XAI) techniques. One of the key evaluation metrics implemented is **Region-based Recall** for Grad-CAM visualizations.

## XAI Metric: Region-based Recall

T### Region-based Recall

The **Region-based Recall** metric evaluates how well the Grad-CAM mask overlaps with the ground truth mask. It is defined as:

$$
\text{Recall} = \frac{| M_{\text{pred}} \cap M_{\text{gt}} |}{| M_{\text{gt}} |}
$$

Where:
- **M_pred**: The predicted Grad-CAM mask.
- **M_gt**: The ground truth mask.
- **| M_pred ∩ M_gt |**: The number of overlapping pixels between the predicted and ground truth masks.
- **| M_gt |**: The total number of pixels in the ground truth mask.

where:
- **CAM Mask** represents the Grad-CAM activation map.
- **GT Mask** represents the ground truth segmentation mask.
- **Intersection** measures the overlapping region between the two masks.

### Implementation in Python

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images
cam_mask = cv2.imread("path_to_gradcam_mask.jpg", cv2.IMREAD_GRAYSCALE)
real_mask = cv2.imread("path_to_ground_truth_mask.png", cv2.IMREAD_GRAYSCALE)

# Resize real_mask to match the dimensions of cam_mask
real_mask_resized = cv2.resize(real_mask, (cam_mask.shape[1], cam_mask.shape[0]))

# Region-based Recall Function
def compute_region_recall(pred_mask, gt_mask):
    pred_binary = (pred_mask > 0.5).astype(int)
    gt_binary = (gt_mask > 0.5).astype(int)
    intersection = np.sum(pred_binary * gt_binary)
    gt_area = np.sum(gt_binary)
    recall = intersection / gt_area if gt_area > 0 else 0
    return recall

# Compute recall
recall = compute_region_recall(cam_mask, real_mask_resized)
print("Recall:", recall)

# Plot the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cam_mask, cmap='gray')
axes[0].set_title("Grad-CAM Mask")
axes[0].axis('off')
axes[1].imshow(real_mask_resized, cmap='gray')
axes[1].set_title("Ground Truth Mask")
axes[1].axis('off')
overlay = np.maximum(cam_mask, real_mask_resized)
axes[2].imshow(overlay, cmap='gray')
axes[2].set_title(f"Overlay (Recall: {recall:.2f})")
axes[2].axis('off')
plt.tight_layout()
plt.show()
```

### How to Use
1. Replace `path_to_gradcam_mask.jpg` and `path_to_ground_truth_mask.png` with the actual file paths.
2. Run the script to compute the recall score.
3. The output will include the recall value and visualizations comparing the CAM mask with the ground truth.

### Future Improvements
- Extend XAI metrics by adding **Precision, F1-score, and IoU (Intersection over Union)**.
- Support additional explainability methods such as SHAP and Integrated Gradients.
- Apply to more welding defect datasets for robustness testing.

---
This metric is a step toward making deep learning models more interpretable and reliable in industrial defect detection tasks.



## License
This project is open-source under the MIT License.


## Cite 
```
@article{Basha2025AdvancingWD,
  title={Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis},
  author={Kamal Basha and Athira Nambiar},
  journal={ArXiv},
  year={2025},
  volume={abs/2508.00381},
  url={https://api.semanticscholar.org/CorpusID:280416743}
}
```

