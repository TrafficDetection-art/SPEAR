"""
Evaluation metrics module.

Provides accuracy, precision, recall, F1, and Attack Success Rate (ASR).
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_metrics_macro(y_true, y_pred, average="macro"):
    """Compute classification metrics (ACC, F1, Precision, Recall) for multi-class."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_metrics(y_true, y_pred):
    """Binary classification metrics focusing on phishing (label=1)."""
    # Filter invalid predictions (value 2) by treating them as benign (0)
    y_pred_filtered = [0 if pred == 2 else pred for pred in y_pred]

    return {
        "Accuracy": accuracy_score(y_true, y_pred_filtered),
        "Precision": precision_score(
            y_true, y_pred_filtered, average="binary", pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred_filtered, average="binary", pos_label=1, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred_filtered, average="binary", pos_label=1, zero_division=0
        ),
    }


def calculate_asr(y_true, y_pred):
    """
    Calculate Attack Success Rate (ASR).

    ASR = number of malicious samples that evaded detection / total malicious samples
    This is essentially the False Negative Rate for phishing emails.

    In the phishing email scenario:
      - True label = 1 (malicious phishing email)
      - Model prediction = 0 (misclassified as benign)
      - This means the attack successfully evaded detection

    Args:
        y_true: True label list (1=malicious, 0=benign)
        y_pred: Predicted label list (1=detected as malicious, 0=detected as benign)

    Returns:
        dict with ASR and related statistics.
    """
    # Filter invalid predictions (value 2) -> treat as benign (0)
    y_pred_filtered = [0 if pred == 2 else pred for pred in y_pred]

    total_malicious = sum(1 for label in y_true if label == 1)

    if total_malicious == 0:
        return {
            "ASR": 0.0,
            "successful_evasions": 0,
            "total_malicious": 0,
            "evasion_rate_percent": 0.0,
            "note": "No malicious samples in dataset",
        }

    # Count malicious samples that were misclassified as benign
    successful_evasions = sum(
        1 for true_label, pred_label in zip(y_true, y_pred_filtered)
        if true_label == 1 and pred_label == 0
    )

    asr = successful_evasions / total_malicious

    return {
        "ASR": asr,
        "successful_evasions": successful_evasions,
        "total_malicious": total_malicious,
        "evasion_rate_percent": asr * 100,
    }
