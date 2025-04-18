from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from loan_sanction_prediction.utils import parse_yaml, setup_logger


def main():
    log = setup_logger(Path(__file__).stem)
    config = parse_yaml(log)
    try:
        df = pd.read_csv(config["data"]["raw"])
        tts_params = config["train_test_split"]
        X_train, X_test = train_test_split(
            df,
            test_size=tts_params["test_size"],
            random_state=tts_params["random_state"],
            stratify=df[tts_params["stratify"]],
        )
        X_test.to_csv(config["data"]["interim"]["test"], index=False)
        X_train.to_csv(config["data"]["interim"]["train"], index=False)
        log.success(f"Successfully split train and test with parameters: {tts_params}")
    except Exception:
        log.exception("Failed to split train and test")
        raise


if __name__ == "__main__":
    main()
