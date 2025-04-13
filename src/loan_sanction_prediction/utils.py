from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger


def setup_logger(name: str, log_dir:str="logs"):
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    logger.add(log_file)
    return logger


def parse_yaml(config_path: Path, log) -> dict:
    try:
        with config_path.open("r") as file:
            config = yaml.safe_load(file)
        log.success("Parsed config.yaml")
        return config
    except Exception:
        log.exception("Failed to parse config.yaml")
        raise
