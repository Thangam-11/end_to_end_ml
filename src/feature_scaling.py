from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from src.logging_system import MLProjectLogger

class FeatureScaler:
    def __init__(self, method="standard", logger=None):
        self.logger = logger or MLProjectLogger("feature_scaling")
        scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
        self.scaler = scalers.get(method, StandardScaler())

    def fit_transform(self, X, cols=None):
        cols = cols or X.columns
        X_scaled = X.copy()
        X_scaled[cols] = self.scaler.fit_transform(X[cols])
        return X_scaled

    def transform(self, X, cols=None):
        cols = cols or X.columns
        X_scaled = X.copy()
        X_scaled[cols] = self.scaler.transform(X[cols])
        return X_scaled

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)
