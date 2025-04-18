import joblib
import pandas as pd
from loan_sanction_prediction.utils import setup_logger, parse_yaml
from pathlib import Path

def main():
    log = setup_logger(Path(__file__).stem)
    config = parse_yaml(log)

    clf = joblib.load("models/model.joblib")
    X_test = pd.read_csv("data/processed/test.csv")
    clf.predict(X_test)

if __name__ == "__main__":
    main()
