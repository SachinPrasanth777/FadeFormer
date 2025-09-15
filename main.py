import torch
from config import Config
from models import FadeNet
from utils import create_data_loaders, Trainer, set_seed


def main():
    config = Config.from_args()
    set_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(config)
    
    model = FadeNet(
        num_classes=num_classes, 
        dropout=config.dropout,
        fade_attn_params=config.fade_attn_params
    ).to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        config=config
    )
    
    results = trainer.train()
    return results


if __name__ == '__main__':
    main()