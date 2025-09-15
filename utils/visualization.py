import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import MetricsCalculator


class Visualizer:
    def __init__(self, save_dir, num_classes):
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.metrics_calc = MetricsCalculator(num_classes)
        os.makedirs(save_dir, exist_ok=True)
    
    def save_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc', linewidth=2)
        plt.plot(val_accs, label='Val Acc', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_confusion_matrix(self, all_targets, all_preds):
        cm = self.metrics_calc.get_confusion_matrix(all_targets, all_preds)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.colorbar()
        
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, [f'Class {i}' for i in range(self.num_classes)], rotation=45)
        plt.yticks(tick_marks, [f'Class {i}' for i in range(self.num_classes)])
        
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
        
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_pr_curves(self, all_targets, all_probs):
        pr_data = self.metrics_calc.get_pr_data(all_targets, all_probs)
        
        plt.figure(figsize=(12, 10))
        
        for precision_c, recall_c, ap_c, label in pr_data:
            plt.plot(recall_c, precision_c, label=f'{label} (AP={ap_c:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(ncol=2 if self.num_classes > 5 else 1, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'pr_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_roc_curves(self, all_targets, all_probs):
        roc_data = self.metrics_calc.get_roc_data(all_targets, all_probs)
        
        plt.figure(figsize=(12, 10))
        
        for fpr_c, tpr_c, roc_auc_c, label in roc_data:
            plt.plot(fpr_c, tpr_c, label=f'{label} (AUC={roc_auc_c:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='Random')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(ncol=2 if self.num_classes > 5 else 1, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'roc_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_plots(self, train_losses, val_losses, train_accs, val_accs,
                      all_targets, all_probs, all_preds):
        self.save_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        if len(all_targets) > 0:
            self.save_confusion_matrix(all_targets, all_preds)
            self.save_pr_curves(all_targets, all_probs)
            self.save_roc_curves(all_targets, all_probs)