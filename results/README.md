# Results Directory

This directory contains all training outputs and evaluation results.

## Generated Files

### Model Files
- `best_model.pth`: Best model weights based on validation loss

### Results Files
- `results.json`: Complete training and test metrics in JSON format

### Visualization Files
- `training_curves.png`: Training and validation loss/accuracy curves
- `confusion_matrix.png`: Test set confusion matrix
- `pr_curves.png`: Precision-Recall curves for each class
- `roc_curves.png`: ROC curves for each class

## Results JSON Structure

```json
{
    "test_loss": 0.1234,
    "test_accuracy": 0.9876,
    "test_precision": 0.9800,
    "test_recall": 0.9750,
    "test_f1": 0.9775,
    "test_auc": 0.9900,
    "train_losses": [...],
    "val_losses": [...],
    "train_accuracies": [...],
    "val_accuracies": [...],
    "config": {...}
}
```

The results are automatically generated during training and provide comprehensive evaluation metrics for analysis and comparison.