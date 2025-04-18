from pathlib import Path
from typing import Literal

import pandas as pd
from klib import clean_column_names

from loan_sanction_prediction.utils import parse_yaml, setup_logger


def data_preprocessor(df_type: Literal["train", "test"], config: dict, log, export_df: bool = True):
    useless_cols = config["useless_cols"]
    categorical_cols = config["features"]["discrete"]["categorical"]
    target = config["target"]

    # Load dataset
    try:
        interim_path = Path(config["data"]["interim"][df_type])
        df = pd.read_csv(interim_path).copy()
        log.success(f"Loaded {df_type} dataset")
    except Exception:
        log.exception(f"Failed to load {df_type} dataset")
        raise

    # Drop useless columns
    try:
        df.drop(columns=useless_cols, inplace=True)
        log.success(f"Dropped useless columns: {useless_cols}")
    except Exception:
        log.exception("Failed to drop useless columns")
        raise

    # Rename columns
    try:
        df = clean_column_names(df)
        log.success(
            f"Renamed columns: {df.columns[:3].to_list()}... (+{len(df.columns) - 3} more)"
        )
    except Exception:
        log.exception("Failed to rename columns")
        raise

    # Clean values
    try:
        for col in categorical_cols + [target]:
            df[col] = df[col].str.lower().str.replace(" ", "_")
        df[target] = df[target].replace({"y": "yes", "n": "no"})
        log.success("Cleaned values in columns")
    except Exception:
        log.exception("Failed to clean values in columns")
        raise

    if export_df:
        # Export dataset
        try:
            processed_path = Path(config["data"]["processed"][df_type])
            df.to_csv(processed_path, index=False)
            log.success(f"Exported {df_type} dataset")
        except Exception:
            log.exception(f"Failed to export {df_type} dataset")
            raise
    else:
        return df


def main():
    log = setup_logger(Path(__file__).stem)
    config = parse_yaml(log)
    data_preprocessor("train", config, log)
    data_preprocessor("test", config, log)


if __name__ == "__main__":
    main()
