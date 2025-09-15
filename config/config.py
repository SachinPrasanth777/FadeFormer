import argparse
import os


class Config:
    def __init__(self):
        self.data_root = './data'
        self.save_dir = './results'
        self.batch_size = 32
        self.epochs = 25
        self.seed = 42
        self.num_workers = 2
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.dropout = 0.5
        self.early_stopping_patience = 6
        self.scheduler_patience = 2
        self.scheduler_factor = 0.5
        
        self.fade_attn_params = {
            'num_heads': 8,
            'K': 3,
            'alpha': 0.5,
            'dropout': 0.1
        }
        
        self.transform_params = {
            'image_size': 224,
            'brightness': 0.15,
            'contrast': 0.15,
            'saturation': 0.15,
            'rotation': 15,
            'erasing_prob': 0.4,
            'erasing_scale': (0.02, 0.12)
        }
    
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='FadeNet Training')
        parser.add_argument('--data_root', type=str, default='./data',
                           help='Root directory of the dataset')
        parser.add_argument('--save_dir', type=str, default='./results',
                           help='Directory to save results')
        parser.add_argument('--batch_size', type=int, default=32,
                           help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=25,
                           help='Number of training epochs')
        parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
        parser.add_argument('--num_workers', type=int, default=2,
                           help='Number of data loading workers')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                           help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                           help='Weight decay')
        parser.add_argument('--dropout', type=float, default=0.5,
                           help='Dropout rate')
        
        args = parser.parse_args()
        
        config = cls()
        for key, value in vars(args).items():
            setattr(config, key, value)
        
        os.makedirs(config.save_dir, exist_ok=True)
        return config