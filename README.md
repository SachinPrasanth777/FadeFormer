# FadeFormer: Fine-grained Attenion-Diffusion Encoder for Enhanced Medical Image Classification

A PyTorch implementation of FadeNet, featuring the novel FadeAttn (Fine-grained Attenion-Diffusion Encoder) mechanism for medical image classification tasks.
Vision Transformers (ViTs) have emerged as one of the most powerful backbones for medical imaging because of their ability to capture global features very effectively, which is vital for complex anatomical understanding and pathological variations. However, ViTs often struggle to maintain spatial coherence, which is an essential feature for medical images where local tissue patterns and fine details are critical for diagnosis. Recent variants of ViT, such as GraphViTs, encode spatial coherence by graph-based relations, but they often hardcode the entire local neighborhood structure without any adaptivity. On the other hand, diffusion models help in iterative feature refinement to better represent subtle abnormalities, but they lack integration with global features, which hinders their ability to balance global context with local structure.To overcome these shortcomings, in this paper we introduce FadeFormer, a novel architecture that unifies graph Laplacian diffusion operator with Transformer self-attention through a learnable mechanism. This enables an adaptive integration of local spatial smoothness and global contextual reasoning for improved medical image analysis. The proposed architecture was tested on publicly available MedMNIST v2, ISIC-2019 and NIH Chest X-Ray 14 medical image datasets.

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
cd FadeFormer
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

<img width="1018" height="1258" alt="image" src="https://github.com/user-attachments/assets/44421b3d-3062-4e39-b6b1-c2cc74021f00" />


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

### Results

Table 1. Comparison of AUC and Accuracy (%) on MedMNIST datasets
| Dataset     | Backbone      | AUC (%)   | Accuracy (%) |
| ----------- | ------------- | --------- | ------------ |
| OrganAMNIST | ResNet18      | 99.8      | 95.1         |
|             | AutoKeras     | 99.4      | 90.5         |
|             | AutoML Vision | 99.0      | 88.6         |
|             | MedViT        | 99.7      | 93.2         |
|             | **FadeFormer**      | **99.9**  | **98.5**     |
| OrganCMNIST | ResNet18      | 99.4      | 92.0         |
|             | AutoKeras     | 99.0      | 87.9         |
|             | AutoML Vision | 98.8      | 87.7         |
|             | MedViT        | 99.3      | 92.0         |
|             | **FadeFormer**      | **99.53** | **92.0**     |
| RetinaMNIST | ResNet18      | 71.0      | 49.3         |
|             | AutoKeras     | 71.9      | 50.3         |
|             | AutoML Vision | 75.0      | 53.1         |
|             | MedViT        | 82.1      | 59.4         |
|             | **FadeFormer**      | **75.3**  | **60.5**     |
| BloodMNIST  | ResNet18      | 99.8      | 96.3         |
|             | AutoKeras     | 99.8      | 96.1         |
|             | AutoML Vision | 99.8      | 96.6         |
|             | MedViT        | 99.7      | 96.8         |
|             | **FadeFormer**      | **99.9**  | **98.1**     |
| ChestMNIST  | ResNet18      | 77.3      | 94.7         |
|             | AutoKeras     | 74.2      | 93.7         |
|             | AutoML Vision | 77.8      | 94.8         |
|             | MedViT        | 55.0      | 94.7         |
|             | **FadeFormer**      | **77.9**  | **94.8**     |


Table 2. Comparison of BMCA and AUC (%) on the ISIC 2019 dataset
| Method       | Backbone          | BMCA (%)  | AUC (%)  |
| ------------ | ----------------- | --------- | -------- |
| ResNet18     | Baseline          | 65.7      | 91.2     |
| EfficientNet | Baseline          | 66.8      | 92.0     |
| ViT          | Transformer-based | 68.36     | 93.06    |
| MedViT       | Transformer-based | 68.29     | 93.34    |
| **FadeFormer**     | Proposed          | **68.43** | **93.5** |

Table 3. Comparison of accuracy and AUC (%) on the NIH ChestX-ray14 dataset
| Method       | Backbone          | Accuracy (%) | AUC (%)  |
| ------------ | ----------------- | ------------ | -------- |
| ResNet18     | Baseline          | 70.2         | 81.3     |
| EfficientNet | Baseline          | 72.9         | 83.5     |
| ViT          | Transformer-based | 74.0         | 85.0     |
| MedViT       | Transformer-based | 74.5         | 86.0     |
| **FadeFormer**     | Proposed          | **74.3**     | **85.8** |

<img width="1182" height="384" alt="image" src="https://github.com/user-attachments/assets/f3d132a4-1ba6-48f0-a285-3012cb1be7fd" />
<img width="1570" height="420" alt="image" src="https://github.com/user-attachments/assets/895549b7-abef-4918-9b40-6c1232170a4c" />

## FLOPS Table

| Model           | Architecture               | Parameters (M) | FLOPs (G) | Memory (GB) |
| --------------- | -------------------------- | -------------- | --------- | ----------- |
| **FADE (Ours)** | ViT-Base + Graph Attention | 86.6           | 569.16    | 3.2         |
| ViT-Base/16     | Vision Transformer         | 86.0           | 550.57    | 3.0         |
| ResNet-50       | Convolutional              | 25.6           | 130.40    | 1.8         |
| EfficientNet-B4 | Efficient CNN              | 19.3           | 18.50     | 1.4         |
| DeiT-Base/16    | Distilled ViT              | 86.0           | 550.57    | 3.0         |


## Citation

This paper is an official submission to WACV 2026

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
