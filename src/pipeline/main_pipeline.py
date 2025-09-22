# src/main_pipeline.py
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.logging_system import MLProjectLogger
from src.config_manager import ConfigManager
from src.compoents.data_ingestion import DataIngestion
from src.compoents.data_cleaning import DataCleaning
from src.compoents.data_splitter import DataSplitter
from src.compoents.model_trainer import ModelTrainer
import pandas as pd


class WaterSafetyMLPipeline:
    """Complete ML pipeline orchestrator (without feature engineering/scaling)"""

    def __init__(self):
        self.logger = MLProjectLogger("main_pipeline")
        self.config = ConfigManager()

    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        try:
            self.logger.log_pipeline_stage("COMPLETE_PIPELINE", "START")

            # 1. Data Ingestion
            self.logger.log_pipeline_stage("DATA_INGESTION", "START")
            ingestion = DataIngestion()
            raw_data = ingestion.load_data()
            self.logger.log_data_info("RAW_DATA", raw_data.shape)

            # 2. Data Cleaning
            cleaner = DataCleaning(raw_data, self.logger)
            cleaned_data = cleaner.clean_data()

            # Save processed data
            os.makedirs("artifacts/processed_data", exist_ok=True)
            processed_path = "artifacts/processed_data/cleaned_data.csv"
            cleaned_data.to_csv(processed_path, index=False)
            self.logger.logger.info(f"PROCESSED_DATA_SAVED | {processed_path}")

            # 3. Data Splitting
            splitter = DataSplitter(cleaned_data, target_col="is_safe")
            train_df, test_df = splitter.split_and_save()
            
            # Separate features/target
            X_train, y_train = train_df.drop(columns=["is_safe"]), train_df["is_safe"]
            X_test, y_test = test_df.drop(columns=["is_safe"]), test_df["is_safe"]

            # 4. Model Training
            trainer = ModelTrainer(self.logger)
            
            results = trainer.train_models(X_train, y_train, X_test, y_test)
            if not results:
                self.logger.logger.error("NO_MODELS_TRAINED")
                raise Exception("No models were trained successfully.")

            # 5. Generate Results Summary
            self._generate_results_summary(results)

            self.logger.log_pipeline_stage("COMPLETE_PIPELINE", "SUCCESS")

            return results

        except Exception as e:
            self.logger.log_exception(e, "complete_pipeline")
            self.logger.log_pipeline_stage("COMPLETE_PIPELINE", "FAILED")
            raise

    def _generate_results_summary(self, results):
        """Generate results summary"""
        if not results:
            self.logger.logger.warning("NO_RESULTS_TO_SUMMARIZE")
            return

        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                "Model": result["model_name"],
                "Train_Accuracy": result["train_metrics"]["accuracy"],
                "Test_Accuracy": result["test_metrics"]["accuracy"],
                "Test_F1_Score": result["test_metrics"]["f1_score"],
                "Test_Precision": result["test_metrics"]["precision"],
                "Test_Recall": result["test_metrics"]["recall"],
                "Test_ROC_AUC": result["test_metrics"].get("roc_auc", "N/A"),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Test_F1_Score", ascending=False)

        # Save summary
        os.makedirs("artifacts/evaluation_results", exist_ok=True)
        summary_path = "artifacts/evaluation_results/model_comparison.csv"
        summary_df.to_csv(summary_path, index=False)

        # Log best model
        best_model = summary_df.iloc[0]
        self.logger.logger.info(
            f"BEST_MODEL | {best_model['Model']} | F1: {best_model['Test_F1_Score']:.4f}"
        )

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    pipeline = WaterSafetyMLPipeline()
    results = pipeline.run_complete_pipeline()
