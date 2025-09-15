import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(self.root_dir) 
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(config):
    transform_train = transforms.Compose([
        transforms.Resize((config.transform_params['image_size'], 
                          config.transform_params['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.transform_params['rotation']),
        transforms.ColorJitter(
            brightness=config.transform_params['brightness'],
            contrast=config.transform_params['contrast'],
            saturation=config.transform_params['saturation']
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=config.transform_params['erasing_prob'],
            scale=config.transform_params['erasing_scale']
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((config.transform_params['image_size'], 
                          config.transform_params['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_test


def create_data_loaders(config):
    transform_train, transform_test = get_transforms(config)
    
    train_dataset = CustomDataset(config.data_root, 'train', transform=transform_train)
    val_dataset = CustomDataset(config.data_root, 'val', transform=transform_test)
    test_dataset = CustomDataset(config.data_root, 'test', transform=transform_test)
    
    num_classes = len(train_dataset.classes)
    
    targets = [sample[1] for sample in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    weights = [class_weights[target] for target in targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=sampler, 
        num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, num_classes