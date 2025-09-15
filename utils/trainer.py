import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import MetricsCalculator
from .visualization import Visualizer


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, num_classes, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.scheduler_factor, 
            patience=config.scheduler_patience
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        self.metrics_calc = MetricsCalculator(num_classes)
        self.visualizer = Visualizer(config.save_dir, num_classes)
    
    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(self.train_loader, desc='Training', leave=False)
        
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)
            y = y.view(-1).long()
            
            if y.numel() == 0:
                continue
            
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)
        
        return total_loss / len(self.train_loader.dataset), correct / total if total > 0 else 0.0
    
    def validate_epoch(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_targets, all_probs = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.view(-1).long()
                
                if y.numel() == 0:
                    continue
                
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
                
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
                all_targets.append(y.cpu())
                all_probs.append(probs.cpu())
        
        if total == 0:
            return float('inf'), 0.0, {}, [], []
        
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        metrics = self.metrics_calc.calculate_metrics(all_targets, all_probs)
        accuracy = correct / total
        avg_loss = total_loss / len(loader.dataset)
        
        return avg_loss, accuracy, metrics, all_targets, all_probs
    
    def save_results(self, results):
        results_path = os.path.join(self.config.save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    
    def train(self):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_path = os.path.join(self.config.save_dir, 'best_model.pth')
        
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print("-" * 80)
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_metrics, _, _ = self.validate_epoch(self.val_loader)
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"Precision {val_metrics.get('precision', 0):.4f} "
                  f"Recall {val_metrics.get('recall', 0):.4f} "
                  f"F1 {val_metrics.get('f1', 0):.4f} "
                  f"AUC {val_metrics.get('auc_macro', 0):.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"    → New best model saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        print("-" * 80)
        print("Training completed. Evaluating on test set...")
        
        self.model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_metrics, all_targets, all_probs = self.validate_epoch(self.test_loader)
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {test_metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {test_metrics.get('f1', 0):.4f}")
        print(f"  AUC: {test_metrics.get('auc_macro', 0):.4f}")
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_metrics.get('precision', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_f1': test_metrics.get('f1', 0),
            'test_auc': test_metrics.get('auc_macro', 0),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accs,
            'val_accuracies': self.val_accs,
            'config': vars(self.config)
        }
        
        self.save_results(results)
        
        if len(all_targets) > 0:
            self.visualizer.save_all_plots(
                self.train_losses, self.val_losses, 
                self.train_accs, self.val_accs,
                all_targets, all_probs, test_metrics.get('predictions', [])
            )
        
        print(f"\nAll results saved to: {self.config.save_dir}")
        return results