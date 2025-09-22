# src/data_splitter.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.exceptions import MLProjectException
from src.logging_system import MLProjectLogger


class DataSplitter:
    def __init__(self, data: pd.DataFrame, target_col="is_safe", test_size=0.2, random_state=42):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.logger = MLProjectLogger("data_splitter")

        # make artifacts dir
        self.artifacts_dir = "artifacts/data"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def split_and_save(self):
        """Split dataset into train/test, apply SMOTE, and save to artifacts."""
        try:
            self.logger.log_pipeline_stage("DATA_SPLITTING", "START")

            # Split
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # Apply SMOTE only on training
            sm = SMOTE(random_state=self.random_state)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

            # Save to artifacts
            train_df = pd.concat([X_train_res, y_train_res], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_path = os.path.join(self.artifacts_dir, "train.csv")
            test_path = os.path.join(self.artifacts_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self.logger.logger.info(f"TRAIN_DATA_SAVED | {train_path} | Shape: {train_df.shape}")
            self.logger.logger.info(f"TEST_DATA_SAVED  | {test_path} | Shape: {test_df.shape}")
            self.logger.log_pipeline_stage("DATA_SPLITTING", "SUCCESS")

            return train_df, test_df

        except Exception as e:
            self.logger.log_exception(e, "data_splitting")
            raise MLProjectException("Error splitting and saving data", e)


# ------------------ Run as script ------------------
if __name__ == "__main__":
    try:
        from src.compoents.data_ingestion import DataIngestion
        from src.compoents.data_cleaning import DataCleaning

        # 1. Load raw data
        df = DataIngestion().load_data()

        # 2. Clean data
        cleaner = DataCleaning(df)
        cleaned = cleaner.clean_data()

        # 3. Split into train/test
        splitter = DataSplitter(cleaned, target_col="is_safe")
        train_df, test_df = splitter.split_and_save()

        print("\n✅ Train shape:", train_df.shape)
        print("✅ Test shape:", test_df.shape)

    except MLProjectException as e:
        print(f"❌ Error: {e}")
