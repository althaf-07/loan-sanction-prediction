from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from loan_sanction_prediction.utils import parse_yaml, setup_logger


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict, log):
        self.config = config
        self.log = log

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        useless_cols = self.config["useless_cols"]
        features = self.config["col_names"]["features"]

        # Drop useless columns
        try:
            X.drop(columns=useless_cols, inplace=True)
            self.log.success(f"Dropped useless columns: {useless_cols}")
        except Exception:
            self.log.exception("Failed to drop useless columns")
            raise

        # Rename columns
        try:
            X.columns = features
            self.log.success(
                f"Renamed columns: {X.columns[:3].to_list()}... (+{len(X.columns) - 3} more)"
            )
        except Exception:
            self.log.exception("Failed to rename columns")
            raise

        # Clean values
        try:
            X["dependents"] = X["dependents"].str.replace(r"\+$", "", regex=True)
            self.log.success("Cleaned values in columns")
        except Exception:
            self.log.exception("Failed to clean values in columns")
            raise

        return X


def main():
    log = setup_logger(Path(__file__).stem)
    config_path = Path(__file__).resolve().parent / "config.yaml"
    config = parse_yaml(config_path, log)

    try:
        train_df = pd.read_csv("data/raw/loan_sanction_train.csv")
        test_df = pd.read_csv("data/raw/loan_sanction_test.csv")
        log.success("Loaded datasets")
    except Exception:
        log.exception("Failed to load datasets")
        raise

    try:
        train_df.rename(columns={"Loan_Status": "loan_status"}, inplace=True)
        train_df["loan_status"].replace({"Y": "yes", "N": "no"}, inplace=True)
        log.success("Cleaned target column")
    except Exception:
        log.exception("Failed to clean target column")
        raise

    try:
        X_train = train_df.drop(columns=["loan_status"])
        y_train = train_df["loan_status"]
        dp = DataProcessor(config, log)
        X_train = dp.fit_transform(X_train)
        X_test = dp.transform(test_df)
        log.success("Applied DataProcessor on datasets")
    except Exception:
        log.exception("Failed to apply DataProcessor on datasets")
        raise

    try:
        X_train = pd.concat([X_train, y_train], axis=1)
        X_train.to_csv("data/processed/train.csv", index=False)
        X_test.to_csv("data/processed/test.csv", index=False)
        log.success("Exported preprocessed datasets")
    except Exception:
        log.exception("Failed to export preprocessed datasets")
        raise


if __name__ == "__main__":
    main()
