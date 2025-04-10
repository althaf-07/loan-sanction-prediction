import pandas as pd
from utils import setup_logger
from pathlib import Path
import yaml

log = setup_logger(Path(__file__).stem)
log.info("Starting data processing pipeline")

log.info("Parsing config.yaml file...")
try:
    with open("src/loan_sanction_prediction/config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    useless_cols = config_data["useless_cols"]
    col_names = config_data["col_names"]
    num_cols = config_data["num"]["cols"]
    num_cols_with_nan = config_data["num"]["nan"]
    cat_cols = config_data["cat"]["cols"]
    cat_cols_with_nan = config_data["cat"]["nan"]
    target_col = config_data["target_col"]
    log.success("Successfully parsed config.yaml file")
except Exception:
    log.exception("Failed to parse config.yaml file")
    raise

log.info("Loading training dataset...")
try:
    df = pd.read_csv("data/raw/loan_sanction_train.csv")
    log.success(f"Loaded training dataset successfully with shape {df.shape}")
except Exception:
    log.exception("Failed to load training dataset")
    raise

log.info(f"Removing useless columns {useless_cols} from dataset...")
try:
    df.drop(columns=useless_cols, inplace=True)
    log.success(f"Removed useless columns successfully: {df.columns.to_list()}")
except Exception:
    log.exception("Failed to remove useless columns")
    raise

log.info("Clean column names...")
try:
    df.columns = col_names
    log.success(f"Successfully cleaned column names: {df.columns.to_list()}")
except Exception:
    log.exception("Failed to clean column names")
    raise

log.info("Format columns...")
try:
    df = df[num_cols + cat_cols + target_col]
    log.success(f"Successfully formatted columns: {df.columns.to_list()}")
except Exception:
    log.exception("Failed to format columns")
    raise
