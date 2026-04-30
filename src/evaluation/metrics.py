# src/evaluation/metrics.py

"""
Centralised evaluation metrics for Tenacious Judge project.
Provides accuracy, precision, recall, F1, and confusion matrix plotting.
"""

import os

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# -----------------------------
# Compute Metrics
# -----------------------------
def compute_classification_metrics(predictions, labels):
    """
    Compute accuracy, precision, recall, and F1 for binary classification.
    predictions: list/array of predicted labels (0/1)
    labels: list/array of true labels (0/1)
    """
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    metrics = {}
    metrics.update(accuracy.compute(predictions=predictions, references=labels))
    metrics.update(
        precision.compute(predictions=predictions, references=labels, average="binary")
    )
    metrics.update(
        recall.compute(predictions=predictions, references=labels, average="binary")
    )
    metrics.update(
        f1.compute(predictions=predictions, references=labels, average="binary")
    )
    return metrics


# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix(
    predictions,
    labels,
    output_dir,
    filename="confusion_matrix.png",
    title="Confusion Matrix",
):
    """
    Plot and save confusion matrix heatmap.
    predictions: list/array of predicted labels
    labels: list/array of true labels
    output_dir: directory to save plot
    filename: output file name
    title: plot title
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad Output", "Good Output"],
        yticklabels=["Bad Output", "Good Output"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath
