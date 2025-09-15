# Data Directory

Place your dataset in this directory with the following structure:

```
data/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class_n/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class_n/
│       └── ...
└── test/
    ├── class_0/
    │   ├── image1.jpg
    │   └── ...
    ├── class_1/
    │   ├── image1.jpg
    │   └── ...
    └── class_n/
        └── ...
```

## Supported Image Formats
- `.jpg` / `.jpeg`
- `.png`

## Requirements
- Each split (train/val/test) must have the same class structure
- Class names should be consistent across all splits
- Images will be automatically resized to 224×224 during training