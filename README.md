# FadeNet: Fractional Attention Diffusion Enhanced Network

A PyTorch implementation of FadeNet, featuring the novel FadeAttn (Fractional Attention Diffusion Enhanced Attention) mechanism for medical image classification tasks.

## Project Structure

```
FadeNet/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration management
├── data/
│   └── README.md              # Data setup instructions
├── models/
│   ├── __init__.py
│   ├── fadenet.py             # Main FadeNet architecture
│   └── attention.py           # FadeAttn mechanism
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # Data loading utilities
│   ├── trainer.py             # Training logic
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Plot generation
│   └── utils.py               # General utilities
├── results/
│   └── README.md              # Results directory
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

```bash
git clone https://github.com/SachinPrasanth777/FadeFormer
cd FadeForner
pip install -r requirements.txt
```

## Data Setup

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── val/
│   ├── class_0/
│   └── class_1/
│   └── ...
└── test/
    ├── class_0/
    └── class_1/
    └── ...
```

## Usage

### Basic Training

```bash
python main.py --data_root ./data --save_dir ./results
```

### Advanced Configuration

```bash
python main.py \
    --data_root ./data \
    --save_dir ./results \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --dropout 0.5 \
    --seed 42
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_root` | str | `./data` | Root directory of dataset |
| `--save_dir` | str | `./results` | Directory to save results |
| `--batch_size` | int | `32` | Batch size for training |
| `--epochs` | int | `25` | Number of training epochs |
| `--learning_rate` | float | `1e-4` | Learning rate |
| `--weight_decay` | float | `1e-4` | Weight decay |
| `--dropout` | float | `0.5` | Dropout rate |
| `--seed` | int | `42` | Random seed |
| `--num_workers` | int | `2` | Data loading workers |

## Model Architecture

### FadeNet
- **Backbone**: Vision Transformer (ViT-Base)
- **Patch Size**: 16×16
- **Input Resolution**: 224×224
- **Novel Component**: FadeAttn layer

### FadeAttn Mechanism
- **Multi-head Attention**: 8 heads by default
- **Fractional Diffusion**: K=3 iterations
- **Learnable Mixing**: α parameter for attention/diffusion balance
- **Laplacian Graph**: Spatial relationship modeling

## Output Files

After training, the following files are saved to the results directory:

- `best_model.pth`: Best model weights
- `results.json`: Complete training metrics
- `training_curves.png`: Loss and accuracy plots
- `confusion_matrix.png`: Test set confusion matrix
- `pr_curves.png`: Precision-Recall curves
- `roc_curves.png`: ROC curves

## Key Features

### Data Handling
- **Weighted Sampling**: Handles class imbalance
- **Data Augmentation**: Random flips, rotations, color jitter
- **Normalization**: ImageNet statistics
- **Multiple Workers**: Efficient data loading

### Training Features
- **Early Stopping**: Prevents overfitting (patience=6)
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Mixed Precision**: Automatic when available
- **Progress Tracking**: Real-time training progress

### Evaluation Metrics
- **Multi-class Support**: Works with any number of classes
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Visual Analysis**: Multiple plot types for result interpretation

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{fadenet2024,
    title={FadeNet: Fractional Attention Diffusion Enhanced Network},
    author={Your Name},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions and support, please open an issue on GitHub.
