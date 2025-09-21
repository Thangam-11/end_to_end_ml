# src/model_trainer.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from src.logging_system import MLProjectLogger
from src.exceptions import MLProjectException

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    def __init__(self, logger=None):
        self.logger = logger or MLProjectLogger("model_trainer")
        self.models = self._get_default_models()

        # make artifacts dir
        self.models_dir = "artifacts/models"
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_default_models(self):
        """Default set of models"""
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                random_state=42, eval_metric="logloss", use_label_encoder=False
            )
        return models

    def _evaluate_model(self, y_true, y_pred, y_proba=None):
        """Compute evaluation metrics"""
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

    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models, evaluate, and save best"""
        try:
            results = []

            for name, model in self.models.items():
                self.logger.log_pipeline_stage(f"TRAINING_{name}", "START")

                # Train
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Probabilities
                try:
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                except Exception:
                    y_train_proba, y_test_proba = None, None

                # Metrics
                train_metrics = self._evaluate_model(y_train, y_train_pred, y_train_proba)
                test_metrics = self._evaluate_model(y_test, y_test_pred, y_test_proba)

                # Save model
                model_path = os.path.join(self.models_dir, f"{name}.pkl")
                joblib.dump(model, model_path)

                self.logger.logger.info(f"MODEL_SAVED | {model_path}")
                self.logger.log_pipeline_stage(f"TRAINING_{name}", "SUCCESS")

                results.append({
                    "model_name": name,
                    "model": model,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                })

            # Pick best model (highest F1 score)
            best_result = max(results, key=lambda x: x["test_metrics"]["f1_score"])
            best_model_path = os.path.join(self.models_dir, "best_model.pkl")
            joblib.dump(best_result["model"], best_model_path)

            self.logger.logger.info(
                f"BEST_MODEL_SAVED | {best_result['model_name']} | {best_model_path}"
            )

            return results

        except Exception as e:
            self.logger.log_exception(e, "model_training")
            raise MLProjectException("Error during model training", e)


# ----------------- Run as script -----------------
if __name__ == "__main__":
    # Quick test with artifacts/data
    train_df = pd.read_csv("artifacts/data/train.csv")
    test_df = pd.read_csv("artifacts/data/test.csv")

    X_train, y_train = train_df.drop(columns=["is_safe"]), train_df["is_safe"]
    X_test, y_test = test_df.drop(columns=["is_safe"]), test_df["is_safe"]

    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)

    print("\n✅ Training finished. Results:")
    for r in results:
        print(r["model_name"], r["test_metrics"])
