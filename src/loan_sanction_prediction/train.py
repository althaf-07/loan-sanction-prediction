from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from loan_sanction_prediction.utils import parse_yaml, setup_logger


def build_column_transformer(config: dict):
    num_cols = config["num"]["cols"]
    cat_ohe_cols = ["gender", "married", "education", "self_employed"]
    cat_oe_cols = ["dependents", "property_area"]

    num_pl = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    ohe_pl = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first")),
        ]
    )

    oe_pl = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    column_transformer = ColumnTransformer(
        [
            ("num", num_pl, num_cols),
            ("cat_ohe", ohe_pl, cat_ohe_cols),
            ("cat_oe", oe_pl, cat_oe_cols),
        ],
        remainder="passthrough",
    )

    return column_transformer


def main():
    log = setup_logger(Path(__file__).stem)
    config_path = Path(__file__).resolve().parent / "config.yaml"
    config = parse_yaml(config_path, log)

    try:
        df = pd.read_csv("data/processed/train.csv")
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]
        log.success("Loaded training dataset")
    except Exception:
        log.exception("Failed to load training dataset")
        raise

    try:
        pipeline = Pipeline([("preprocessor", build_column_transformer(config)), ("classifier", RandomForestClassifier())])
        pipeline.fit(X, y)
        joblib.dump(pipeline, "models/model.joblib")
        log.success("Model trained and saved")
    except Exception:
        log.exception("Model training failed")
        raise

if __name__ == "__main__":
    main()
