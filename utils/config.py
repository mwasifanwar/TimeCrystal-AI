# utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            "model": {
                "architecture": "Transformer",
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "dropout": 0.1,
                "seq_len": 100,
                "pred_len": 30
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "batch_size": 32,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "forecasting": {
                "enable_uncertainty": True,
                "confidence_level": 0.95,
                "n_simulations": 1000,
                "multiple_horizons": [7, 30, 90, 180]
            },
            "data_preprocessing": {
                "handle_missing": True,
                "remove_outliers": True,
                "make_stationary": True,
                "feature_engineering": True
            },
            "deployment": {
                "framework": "FastAPI",
                "generate_dockerfile": True,
                "enable_monitoring": True
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "configs/default.yaml"):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_model_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("model", {})

def get_training_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("training", {})

def get_forecasting_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("forecasting", {})

def update_config(section: str, key: str, value: Any):
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_forecasting_params() -> Dict[str, Any]:
    config = load_config()
    forecasting = config.get("forecasting", {})
    
    return {
        "enable_uncertainty": forecasting.get("enable_uncertainty", True),
        "confidence_level": forecasting.get("confidence_level", 0.95),
        "n_simulations": forecasting.get("n_simulations", 1000)
    }