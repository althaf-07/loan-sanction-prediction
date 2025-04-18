from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
# import numpy as np
# from sklearn.exceptions 
# import warnings
from loan_sanction_prediction.utils import get_model, parse_yaml, setup_logger


def build_ct(config: dict):
    continuous_cols = config["features"]["continuous"]
    oe_cols = config["preprocessing"]["discrete"]["encode"]["oe"]
    ohe_cols = config["preprocessing"]["discrete"]["encode"]["ohe"]

    continuous_pl = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    ohe_pl = Pipeline(          
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    oe_pl = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder()),
        ]
    )

    ct = ColumnTransformer(
        [
            ("continuous", continuous_pl, continuous_cols),
            ("oe", oe_pl, oe_cols),
            ("ohe", ohe_pl, ohe_cols),
        ],
        remainder="passthrough",
    )

    return ct


def main():
    log = setup_logger(Path(__file__).stem)
    config = parse_yaml(log)
    target = config["target"]

    try:
        df = pd.read_csv(config["data"]["processed"]["train"])
        X_train = df.drop(target, axis=1)
        y_train = df[target]
        log.success("Loaded training dataset")
    except Exception:
        log.exception("Failed to load training dataset")
        raise
    
    ct = build_ct(config)
    clf = get_model(config)
    pl = Pipeline([("ct", ct), ("clf", clf)])
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    try:
        cv_scores = cross_val_score(pl, X_train, y_train, cv=5)
        log.success(
            f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )
    except Exception:
        log.exception("Cross-validation failed")
        raise

    try:
        pl.fit(X_train, y_train)
        model_dir = Path("models")
        joblib.dump(pl, model_dir / "pl.joblib")
        joblib.dump(le, model_dir / "le.joblib")
        log.success("Pipeline trained and saved to models/pl.joblib")
    except Exception:
        log.exception("Pipeline training and saving failed")
        raise


if __name__ == "__main__":
    main()
