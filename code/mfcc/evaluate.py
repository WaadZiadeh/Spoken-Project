"""Spoof-positive metrics, interpolated ROC EER, and final test plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report,
)


def classification_metrics(y_true, predictions, scores):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def equal_error_rate(fpr, tpr):
    """Linearly interpolate the adjacent ROC points where FPR crosses FNR.

    This estimates equality on the empirical ROC polyline, including vertical
    segments. It does not claim a realizable single threshold at an interpolated point.
    """
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        return float(fpr[right])
    left = right - 1
    weight = -difference[left] / (difference[right] - difference[left])
    return float(fpr[left] + weight * (fpr[right] - fpr[left]))


def evaluate_test(model, scaler, x_test, y_test, output_dir):
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("SVC score orientation is not spoof-positive")
    scaled = scaler.transform(x_test)
    predictions = model.predict(scaled)
    scores = model.decision_function(scaled)
    metrics = classification_metrics(y_test, predictions, scores)
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1, drop_intermediate=False)
    metrics["eer"] = equal_error_rate(fpr, tpr)
    metrics["eer_percent"] = 100 * metrics["eer"]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=["Bonafide", "Spoof"], zero_division=0)
    print(report)
    print("Confusion matrix rows=true, columns=predicted; order=[Bonafide, Spoof]")
    print("[[True Bonafide, False Spoof], [False Bonafide, True Spoof]]")
    print(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Bonafide", "Spoof"]).plot(ax=ax, cmap="Blues")
    ax.set_title("MFCC + SVM — Test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"MFCC + SVM (ROC-AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Test ROC curve (positive class: Spoof)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)
    return metrics, cm, report
