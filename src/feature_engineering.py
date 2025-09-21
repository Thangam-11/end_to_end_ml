import numpy as np
import pandas as pd
import os
from src.exceptions import MLProjectException
from src.logging_system import MLProjectLogger  # ✅ Add logger


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, target_col="is_safe", logger=None):
        self.data = data
        self.target_col = target_col
        self.logger = logger or MLProjectLogger("feature_engineering")
        self.thresholds = {
            "aluminium": 2.8, "ammonia": 32.5, "arsenic": 0.01,
            "barium": 2, "cadmium": 0.005, "chloramine": 4,
            "chromium": 0.1, "copper": 1.3, "fluoride": 1.5,
            "bacteria": 0, "viruses": 0, "lead": 0.015,
            "nitrates": 10, "nitrites": 1, "mercury": 0.002,
            "perchlorate": 56, "radium": 5, "selenium": 0.5,
            "silver": 0.1, "uranium": 0.3
        }

    def danger_flags(self):
        self.logger.log_pipeline_stage("DANGER_FLAGS", "START")
        for col, th in self.thresholds.items():
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                self.data[f"{col}_high"] = (self.data[col] > th).astype(int)
        self.logger.log_pipeline_stage("DANGER_FLAGS", "SUCCESS")
        return self.data

    def danger_count(self):
        self.logger.log_pipeline_stage("DANGER_COUNT", "START")
        flags = [f"{c}_high" for c in self.thresholds if f"{c}_high" in self.data.columns]
        if flags:
            self.data["danger_count"] = self.data[flags].sum(axis=1)
        self.logger.log_pipeline_stage("DANGER_COUNT", "SUCCESS")
        return self.data

    def ratio_features(self):
        self.logger.log_pipeline_stage("RATIO_FEATURES", "START")
        if "nitrates" in self.data and "nitrites" in self.data:
            self.data["nitrate_nitrite_ratio"] = np.where(
                self.data["nitrites"] > 0,
                self.data["nitrates"] / self.data["nitrites"],
                0
            )
        self.logger.log_pipeline_stage("RATIO_FEATURES", "SUCCESS")
        return self.data

    def interaction_features(self):
        self.logger.log_pipeline_stage("INTERACTION_FEATURES", "START")
        if "bacteria" in self.data and "viruses" in self.data:
            self.data["bacteria_viruses"] = self.data["bacteria"] * self.data["viruses"]
        self.logger.log_pipeline_stage("INTERACTION_FEATURES", "SUCCESS")
        return self.data

    def engineer_features(self, save_path=None):
        try:
            self.data = self.danger_flags()
            self.data = self.danger_count()
            self.data = self.ratio_features()
            self.data = self.interaction_features()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.data.to_csv(save_path, index=False)
                self.logger.logger.info(f"ENGINEERED_DATA_SAVED | {save_path}")

            return self.data

        except Exception as e:
            self.logger.log_exception(e, "feature_engineering")
            raise MLProjectException("Error in feature engineering", e)


# ----------------- Run as script -----------------
if __name__ == "__main__":
    try:
        file_path = os.path.join("data", "raw_data", "project_data.csv")
        df = pd.read_csv(file_path)

        fe = FeatureEngineering(df, target_col="is_safe")
        final_data = fe.engineer_features(
            save_path=r"C:\Users\thang\Desktop\Water_classfiaction_app\data\raw_data\pre_procese_data.csv"
        )

        print("\n✅ Final engineered dataset shape:", final_data.shape)
        print(final_data.head())

    except MLProjectException as e:
        print(f"❌ Error: {e}")
