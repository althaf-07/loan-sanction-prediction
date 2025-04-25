from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from loan_sanction_prediction.utils import parse_yaml, setup_logger


def plot_and_save(df, cols, plot, save_dir, log, figsize=(10, 6)):
    try:
        save_dir.mkdir(exist_ok=True, parents=True)
        for col in cols:
            save_path = save_dir / f"{col}.png"
            plt.figure(figsize=figsize)
            if plot == "hist":
                sns.histplot(df[col], kde=True)
            elif plot == "box":
                sns.boxplot(x=df[col])
            elif plot == "count":
                sns.countplot(x=df[col])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            log.success(f"✅ Saved: {save_path}")

    except Exception as e:
        log.exception(f"Failed to generate plot for column: {col}. Error: {e}")
        raise


def main():
    log = setup_logger(Path(__file__).stem)
    config = parse_yaml(log)
    plot: Literal["pie", "count", "hist", "box"] = "count"
    # continuous_cols = config["features"]["continuous"]
    discrete_cat_cols = config["features"]["discrete"]["categorical"]
    discrete_num_cols = config["features"]["discrete"]["numerical"]
    discrete_cols = discrete_cat_cols + discrete_num_cols
    dirs_for_plot = {"hist": "histograms", "box": "boxplots", "count": "countplots"}
    cols = discrete_cols
    # Load dataset
    try:
        df_path = Path(config["data"]["interim"]["train"])
        df = pd.read_csv(df_path, usecols=cols)
        log.success("Loaded dataset")
    except Exception:
        log.exception("Failed to load dataset")
        raise

    save_dir = Path(f"reports/figures/univariate/{dirs_for_plot[plot]}/")
    plot_and_save(df, cols, plot, save_dir, log)


if __name__ == "__main__":
    main()
