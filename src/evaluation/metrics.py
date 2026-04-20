import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def evaluate_classification(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_proba, labels=['H', 'D', 'A']),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    return metrics


def print_evaluation_report(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = ['H', 'D', 'A']
    
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix")
    print("-" * 30)
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Predicted")
    print(f"              H    D    A")
    for i, row_name in enumerate(target_names):
        print(f"Actual {row_name}  {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")


def compute_class_distribution(y):
    if isinstance(y, pd.Series):
        return y.value_counts(normalize=True).to_dict()
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts / len(y)))
