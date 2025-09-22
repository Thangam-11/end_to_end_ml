import joblib
import pandas as pd
import os

class PredictionPipeline:
    def __init__(self, model_path="models/trained_models/best_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame):
        """
        Run predictions on given dataframe
        Returns: predictions, probabilities
        """
        preds = self.model.predict(data)
        
        try:
            probs = self.model.predict_proba(data)
        except:
            # For models without predict_proba (like SVM with linear kernel)
            probs = [[1-p, p] for p in preds]

        return preds, probs
# Example usage:
# pipeline = PredictionPipeline()