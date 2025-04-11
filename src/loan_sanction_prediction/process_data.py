import pandas as pd
from utils import setup_logger
from pathlib import Path
import yaml
import time

start = time.time()
log = setup_logger(Path(__file__).stem)
log.info("Started data processing...")

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
    log.success("Parsed config.yaml")
except Exception:
    log.exception("Failed to parse config.yaml")
    raise

try:
    df = pd.read_csv("data/raw/loan_sanction_train.csv")
    log.success(f"Loaded training dataset with shape {df.shape}")
except Exception:
    log.exception("Failed to load training dataset")
    raise

try:
    df.drop(columns=useless_cols, inplace=True)
    log.success(f"Removed useless columns: {useless_cols}")
except Exception:
    log.exception("Failed to remove useless columns")
    raise

try:
    df.columns = col_names
    log.success(f"Cleaned column names: {df.columns[:3].to_list()}... (+{len(df.columns) - 3} more)")
except Exception:
    log.exception("Failed to clean column names")
    raise

try:
    df['dependents'] = df['dependents'].str.replace(r'\+$', '', regex=True)
    df["loan_status"] = df["loan_status"].replace({"Y": "yes", "N": "no"})
    log.success("Cleaned values in columns")
except Exception:
    log.exception("Failed to clean values in columns")
    raise

try:
    df['dependents'] = df['dependents'].astype("Int64")
    df['loan_amount_term'] = df['loan_amount_term'].astype("Int64")
    df['credit_history'] = df['credit_history'].astype("Int64")
    df[cat_cols] = df[cat_cols].astype("category")
    log.success("Converted data types")
except Exception:
    log.exception("Failed to convert data types")
    raise

log.info(f"Finished data processing in {time.time() - start:.2f}s")
