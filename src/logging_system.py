# src/logging_system.py
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path

class MLProjectLogger:
    def __init__(self, logger_name="ml_project", log_dir="logs", log_level="INFO"):
        self.logger_name = logger_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        self._setup_handlers()

    def _setup_handlers(self):
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"{self.logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    # ---------------- Helper Logging Methods ----------------
    def log_exception(self, exception, context=""):
        self.logger.error(f"EXCEPTION | {context} | {type(exception).__name__}: {exception}")
        self.logger.debug(f"TRACEBACK | {traceback.format_exc()}")

    def log_pipeline_stage(self, stage, status):
        """Log pipeline stage with START/SUCCESS/FAILED"""
        self.logger.info(f"PIPELINE_STAGE | {stage} | STATUS: {status}")

    def log_data_info(self, name, shape, **kwargs):
        """Log dataset info like shape and additional metadata"""
        info = {"rows": shape[0], "columns": shape[1]}
        info.update(kwargs)
        self.logger.info(f"DATA_INFO | {name} | {json.dumps(info)}")

    def log_model_performance(self, model_name, metrics, dataset_type="test"):
        """Log model performance metrics"""
        self.logger.info(f"MODEL_PERFORMANCE | {model_name} | {dataset_type.upper()} | {json.dumps(metrics)}")

    def log_hyperparameters(self, model_name, params):
        """Log model hyperparameters"""
        self.logger.info(f"HYPERPARAMETERS | {model_name} | {json.dumps(params)}")
