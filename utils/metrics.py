import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


class MetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def calculate_metrics(self, all_targets, all_probs):
        if len(all_targets) == 0:
            return self._empty_metrics()
        
        preds = all_probs.argmax(axis=1)
        
        precision = precision_score(all_targets, preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
        
        y_true_bin = label_binarize(all_targets, classes=np.arange(self.num_classes))
        
        if self.num_classes == 2:
            y_true_bin = y_true_bin.flatten()
            all_probs_binary = all_probs[:, 1]
            auc_macro = roc_auc_score(y_true_bin, all_probs_binary)
        else:
            auc_per_class = []
            for c in range(self.num_classes):
                if np.sum(y_true_bin[:, c]) > 0 and np.sum(y_true_bin[:, c] == 0) > 0:
                    auc_per_class.append(roc_auc_score(y_true_bin[:, c], all_probs[:, c]))
            auc_macro = np.mean(auc_per_class) if len(auc_per_class) > 0 else float('nan')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_macro': auc_macro,
            'predictions': preds
        }
    
    def _empty_metrics(self):
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_macro': float('nan'),
            'predictions': []
        }
    
    def get_confusion_matrix(self, all_targets, all_preds):
        return confusion_matrix(all_targets, all_preds)
    
    def get_roc_data(self, all_targets, all_probs):
        y_true_bin = label_binarize(all_targets, classes=np.arange(self.num_classes))
        
        if self.num_classes == 2:
            y_true_bin = y_true_bin.flatten()
            fpr, tpr, _ = roc_curve(y_true_bin, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            return [(fpr, tpr, roc_auc, 'Binary')]
        
        roc_data = []
        for c in range(self.num_classes):
            fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], all_probs[:, c])
            roc_auc_c = auc(fpr_c, tpr_c)
            roc_data.append((fpr_c, tpr_c, roc_auc_c, f'Class {c}'))
        
        return roc_data
    
    def get_pr_data(self, all_targets, all_probs):
        y_true_bin = label_binarize(all_targets, classes=np.arange(self.num_classes))
        
        if self.num_classes == 2:
            y_true_bin = y_true_bin.flatten()
            precision_c, recall_c, _ = precision_recall_curve(y_true_bin, all_probs[:, 1])
            ap_c = auc(recall_c, precision_c)
            return [(precision_c, recall_c, ap_c, 'Binary')]
        
        pr_data = []
        for c in range(self.num_classes):
            precision_c, recall_c, _ = precision_recall_curve(y_true_bin[:, c], all_probs[:, c])
            ap_c = auc(recall_c, precision_c)
            pr_data.append((precision_c, recall_c, ap_c, f'Class {c}'))
        
        return pr_data