# src/config_manager.py
import yaml
from pathlib import Path
from src.logging_system import MLProjectLogger

class ConfigManager:
    """
    Handles loading of configuration files (YAML/JSON) for models and pipeline.
    Keeps code clean by separating configs from implementation.
    """

    def __init__(self, config_path="configs/config.yaml", logger=None):
        self.config_path = Path(config_path)
        self.logger = logger or MLProjectLogger("config_manager")
        self.config = self._load_config()

    def _load_config(self):
        """Load YAML configuration file"""
        try:
            if not self.config_path.exists():
                self.logger.logger.warning(f"CONFIG_NOT_FOUND | Using defaults")
                return {}
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            self.logger.logger.info(f"CONFIG_LOADED | {self.config_path}")
            return config or {}
        except Exception as e:
            self.logger.log_exception(e, "loading_config")
            return {}

    def get(self, section, default=None):
        """Get a section of the config"""
        return self.config.get(section, default)

    def get_model_config(self, model_name):
        """Get configuration for a specific model"""
        return self.config.get("models", {}).get(model_name, {})

    def get_all_models_config(self):
        """Get configuration for all models"""
        return self.config.get("models", {})

    def get_training_config(self):
        """Get general training parameters (batch size, epochs, etc.)"""
        return self.config.get("training", {})

    def get_paths(self):
        """Get paths section"""
        return self.config.get("paths", {})
