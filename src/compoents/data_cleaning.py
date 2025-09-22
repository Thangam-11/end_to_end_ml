import time
from functools import wraps
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exceptions import MLProjectException
from src.logging_system import MLProjectLogger
from src.compoents.data_ingestion import DataIngestion  # ✅ for standalone run


class DataCleaning:
    """Enhanced data cleaning with comprehensive logging"""
    
    def __init__(self, data: pd.DataFrame, logger=None, artifacts_path="artifacts/processed_data/cleaned_data.csv"):
        self.data = data
        self.logger = logger or MLProjectLogger("data_cleaning")
        self.artifacts_path = artifacts_path
        os.makedirs(os.path.dirname(self.artifacts_path), exist_ok=True)

    def convert_to_numeric(self):
        """Convert columns to numeric with logging"""
        self.logger.log_pipeline_stage("NUMERIC_CONVERSION", "START")
        try:
            numeric_columns = ['ammonia', 'is_safe']
            conversion_log = {}
            for col in numeric_columns:
                if col in self.data.columns:
                    original_dtype = self.data[col].dtype
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    conversion_log[col] = f"{original_dtype} -> {self.data[col].dtype}"
            if "is_safe" in self.data.columns:
                original_count = len(self.data)
                self.data = self.data.dropna(subset=["is_safe"])
                dropped_count = original_count - len(self.data)
                if dropped_count > 0:
                    self.logger.logger.info(f"DROPPED_NAN_TARGETS | Count: {dropped_count}")
            self.logger.logger.info(f"NUMERIC_CONVERSION | {conversion_log}")
            self.logger.log_pipeline_stage("NUMERIC_CONVERSION", "SUCCESS")
            return self.data
        except Exception as e:
            self.logger.log_exception(e, "numeric_conversion")
            raise MLProjectException("Error converting columns to numeric", e)

    def handle_missing_values(self):
        """Handle missing values with detailed logging"""
        self.logger.log_pipeline_stage("MISSING_VALUE_HANDLING", "START")
        try:
            features_with_na = [f for f in self.data.columns if self.data[f].isnull().sum() > 0]
            for feature in features_with_na:
                missing_count = self.data[feature].isnull().sum()
                missing_pct = round(missing_count / len(self.data) * 100, 2)
                self.logger.logger.warning(f"MISSING_VALUES | {feature}: {missing_count} ({missing_pct}%)")
            imputation_methods = {}
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:
                    if self.data[col].dtype == "object":
                        if len(self.data[col].mode()) > 0:
                            mode_val = self.data[col].mode()[0]
                            self.data[col].fillna(mode_val, inplace=True)
                            imputation_methods[col] = f"mode ({mode_val})"
                    else:
                        mean_val = self.data[col].mean()
                        self.data[col].fillna(mean_val, inplace=True)
                        imputation_methods[col] = f"mean ({mean_val:.4f})"
            self.logger.logger.info(f"IMPUTATION_METHODS | {imputation_methods}")
            self.logger.log_pipeline_stage("MISSING_VALUE_HANDLING", "SUCCESS")
            return self.data
        except Exception as e:
            self.logger.log_exception(e, "missing_value_handling")
            raise MLProjectException("Error handling missing values", e)

    def encode_categorical(self):
        """Encode categorical variables with logging"""
        try:
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                self.logger.logger.info(f"CATEGORICAL_ENCODING | Columns: {categorical_cols}")
                original_shape = self.data.shape
                self.data = pd.get_dummies(self.data, drop_first=True)
                new_shape = self.data.shape
                self.logger.logger.info(f"ENCODING_RESULT | {original_shape} -> {new_shape}")
            else:
                self.logger.logger.info("CATEGORICAL_ENCODING | No categorical columns found")
            return self.data
        except Exception as e:
            self.logger.log_exception(e, "categorical_encoding")
            raise MLProjectException("Error encoding categorical variables", e)

    def clean_target(self):
        """Clean target column with logging"""
        try:
            if "is_safe" in self.data.columns:
                original_count = len(self.data)
                self.data["is_safe"] = pd.to_numeric(self.data["is_safe"], errors="coerce")
                self.data = self.data[self.data["is_safe"].isin([0, 1])]
                removed_count = original_count - len(self.data)
                if removed_count > 0:
                    self.logger.logger.info(f"TARGET_CLEANING | Removed {removed_count} invalid values")
                target_dist = self.data["is_safe"].value_counts().to_dict()
                self.logger.logger.info(f"TARGET_DISTRIBUTION | {target_dist}")
            return self.data
        except Exception as e:
            self.logger.log_exception(e, "target_cleaning")
            raise MLProjectException("Error cleaning target column", e)

    def log_execution_time(logger):
        """Decorator to log execution time of methods"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.logger.info(f"EXECUTION_TIME | {func.__name__} took {elapsed_time:.2f} seconds")
                return result
            return wrapper
        return decorator

    @log_execution_time(MLProjectLogger("data_cleaning"))
    def clean_data(self):
        """Complete cleaning pipeline with logging + save to artifacts"""
        self.logger.log_pipeline_stage("DATA_CLEANING", "START")
        try:
            original_shape = self.data.shape
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                self.data = self.data.drop_duplicates()
                self.logger.logger.info(f"DUPLICATES_REMOVED | Count: {duplicates}")
            self.data = self.convert_to_numeric()
            self.data = self.clean_target()
            self.data = self.handle_missing_values()
            self.data = self.encode_categorical()
            final_shape = self.data.shape
            rows_removed = original_shape[0] - final_shape[0]
            cols_added = final_shape[1] - original_shape[1]
            self.logger.log_data_info("CLEANED_DATA", final_shape, 
                                    rows_removed=rows_removed, cols_added=cols_added)
            self.logger.log_pipeline_stage("DATA_CLEANING", "SUCCESS")

            # ✅ Save cleaned data to artifacts
            self.data.to_csv(self.artifacts_path, index=False)
            self.logger.logger.info(f"CLEANED_DATA_SAVED | {self.artifacts_path}")

            return self.data
        except Exception as e:
            self.logger.log_exception(e, "data_cleaning_pipeline")
            raise MLProjectException("Error in clean_data pipeline", e)


# -------------------- Test block --------------------
if __name__ == "__main__":
    try:
        data = DataIngestion().load_data()
        print("🔹 Raw Data Sample:")
        print(data.head())

        cleaner = DataCleaning(data)
        cleaned_data = cleaner.clean_data()
        print("\n✅ Cleaned Data Sample:")
        print(cleaned_data.head())
        print(f"\n📂 Cleaned data saved to: {cleaner.artifacts_path}")

    except MLProjectException as e:
        print(f"❌ Error: {e}")
