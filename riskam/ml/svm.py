"""
ml.svm

A simple one-class SVM model for OOD detection.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from riskam.data import ml_datasets


def _load_features(dataset_name: str, task: str, split: str) -> np.ndarray:
    """
    Load the features from the dataset.
    """

    # Load the dataset
    dataset = ml_datasets.get_dataset(
        name=dataset_name, transform=None, task=task, split=split
    )

    features = dataset[0]

    for i in range(1, len(dataset)):
        features = np.vstack((features, dataset[i]))

    return features


def train(dataset_name: str, task: str) -> OneClassSVM:
    """
    Train and eval the one class SVM model
    """

    X_train = _load_features(dataset_name, task, "train")
    X_val = _load_features(dataset_name, task, "val")
    X_ood = _load_features(dataset_name, task, "risk")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_ood_scaled = scaler.transform(X_ood)

    # One-Class SVM
    svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    svm.fit(X_train_scaled)

    # Evaluate
    val_scores = svm.decision_function(X_val_scaled)
    ood_scores = svm.decision_function(X_ood_scaled)

    # Threshold tuning
    threshold = np.percentile(val_scores, 5)
    ood_predictions = ood_scores > threshold

    y_true = np.zeros(len(X_ood))
    y_pred = ood_predictions

    # Metrics
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"ROC-AUC: {roc_auc:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Precision, Recall, F1-Score
    print("\nClassification Report:")
    print(
        classification_report(y_true, y_pred, target_names=["OOD", "In-Distribution"])
    )

    # Visualize Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["OOD", "In-Distribution"],
        yticklabels=["OOD", "In-Distribution"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return svm
