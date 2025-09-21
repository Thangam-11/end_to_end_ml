# src/model_evaluator.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from src.logging_system import MLProjectLogger


class ModelEvaluator:
    def __init__(self, target_col="is_safe", logger=None):
        self.logger = logger or MLProjectLogger("model_evaluator")
        self.target_col = target_col
        self.artifacts_dir = "artifacts/evaluation_results"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate all evaluation metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None and y_proba.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                metrics["roc_auc"] = 0.0
        return metrics

    def generate_classification_report(self, y_true, y_pred, model_name):
        """Generate and save classification report"""
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        report_path = os.path.join(self.artifacts_dir, f"{model_name}_classification_report.csv")
        df_report.to_csv(report_path)

        self.logger.logger.info(f"CLASSIFICATION_REPORT_SAVED | {report_path}")
        return df_report

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Unsafe", "Safe"], yticklabels=["Unsafe", "Safe"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")

        cm_path = os.path.join(self.artifacts_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        self.logger.logger.info(f"CONFUSION_MATRIX_SAVED | {cm_path}")
        return cm_path


# ------------------ Run as script ------------------
if __name__ == "__main__":
    # Load best model + test data
    model = joblib.load("artifacts/models/best_model.pkl")
    test_df = pd.read_csv("artifacts/data/test.csv")

    X_test, y_test = test_df.drop(columns=["is_safe"]), test_df["is_safe"]

    evaluator = ModelEvaluator()

    # Predictions
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None

    # Metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
    print("\n📊 Evaluation Metrics:", metrics)

    # Classification report
    report_df = evaluator.generate_classification_report(y_test, y_pred, "best_model")
    print("\n📄 Classification Report:\n", report_df)

    # Confusion matrix
    cm_path = evaluator.plot_confusion_matrix(y_test, y_pred, "best_model")
    print(f"\n📉 Confusion matrix saved at: {cm_path}")
    print("\n✅ Model evaluation completed successfully!")