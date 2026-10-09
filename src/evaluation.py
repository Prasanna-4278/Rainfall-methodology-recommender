"""
Evaluation Metrics Module
Calculates standard and domain-critical rainfall forecasting metrics:
Accuracy, Rain Precision, Rain Recall, Rain F1, ROC-AUC, Cohen's Kappa,
Confusion Matrix, and Balanced Score.
"""
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix, brier_score_loss
)


def compute_metrics(y_true, y_pred, y_prob=None, train_time=0.0):
    """
    Computes complete performance evaluation metrics for binary rainfall prediction.
    Focuses specifically on Rain class (positive class = 1).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_prob : array-like, optional
        Predicted probabilities for Rain (class 1).
    train_time : float, optional
        Model training time in seconds.
        
    Returns
    -------
    metrics : dict
        Calculated metric dictionary.
    """
    acc = float(accuracy_score(y_true, y_pred))
    rain_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rain_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    rain_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # ROC-AUC and Brier Score if probabilities provided
    roc_auc = np.nan
    brier = np.nan
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                brier = float(brier_score_loss(y_true, y_prob))
        except Exception:
            pass
            
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    # Balanced Score (domain-oriented: 35% Recall, 35% F1, 20% ROC-AUC, 10% Accuracy)
    # Reflects meteorology priority: identifying precipitation events correctly
    auc_val = roc_auc if not np.isnan(roc_auc) else acc
    balanced_score = float(0.35 * rain_rec + 0.35 * rain_f1 + 0.20 * auc_val + 0.10 * acc)
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(rain_prec, 4),
        'recall': round(rain_rec, 4),
        'f1': round(rain_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'roc_auc': round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        'cohen_kappa': round(kappa, 4),
        'brier_score': round(brier, 4) if not np.isnan(brier) else None,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'confusion_matrix': cm.tolist(),
        'balanced_score': round(balanced_score, 4),
        'training_time_sec': round(train_time, 3)
    }
