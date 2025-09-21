import pandas as pd
import os
import shutil
from src.logging_system import MLProjectLogger
from src.config_manager import ConfigManager


class DataIngestion:
    """Handles data ingestion from CSV/Excel/Database."""

    def __init__(self, logger=None):
        self.config = ConfigManager()
        # 👇 Make sure your config.yaml has paths: { raw_data: "data/raw_data/project_data.csv" }
        self.data_path = self.config.get("paths", {}).get("raw_data", "data/raw_data/project_data.csv")
        self.artifacts_dir = "artifacts/raw_data"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.logger = logger or MLProjectLogger("data_ingestion")

    def load_data(self):
        """Load dataset from a CSV file and save a copy into artifacts"""
        try:
            data = pd.read_csv(self.data_path)
            if data.empty:
                self.logger.logger.warning(f"DATA_LOADED_BUT_EMPTY | {self.data_path}")
            else:
                self.logger.logger.info(f"DATA_LOADED | {self.data_path} | Shape: {data.shape}")

            # Save a copy into artifacts/raw_data/
            artifact_path = os.path.join(self.artifacts_dir, "raw_data.csv")
            shutil.copy(self.data_path, artifact_path)
            self.logger.logger.info(f"RAW_DATA_SAVED | {artifact_path}")

            return data
        except Exception as e:
            self.logger.log_exception(e, "loading_data")
            raise


# ✅ Test code (only runs if you execute this file directly)
if __name__ == "__main__":
    ingestion = DataIngestion()
    data = ingestion.load_data()

    if data is not None and not data.empty:
        print("✅ Data loaded successfully!")
        print("Shape:", data.shape)
        print(data.head())
    else:
        print("⚠️ Data ingestion failed or empty dataset.")
