from loguru import logger
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir="logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    logger.add(log_file)
    return logger
