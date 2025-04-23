# Project Documentation

## 📌 Project Overview

- **Project Name**: Loan Sanction Prediction
- **Problem Statement**: ?
- **Objective**: Predict whether a loan get sanctioned or not.
- **Type of Problem**: Binary Classification
- **Target Column**: `loan_status`
- **Success Metrics**: Since no class needs to be prioritized (as of I know), Accuracy score (after handling class imbalance) and F1-Score are better candidates for evaluation metrics.

## 📂 Dataset Information

- **Source**: [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data) and [Google Drive](https://drive.google.com/drive/folders/1RxB5VRBdxYgGbEVorE8ysob29dVZ8CWE?usp=sharing)
- **Datapoints (Rows)**: 614
- **Features + Target (Columns)**: 11 + 1
- **Discrete Features**: 8
- **Continuous Features**: 3
- **Duplicate Records**: None
- **Useless Columns**: `Loan_ID`
- **Memory Usage**: 57.7+ KB
- **Target Distribution**:

## 🧾 Feature Summary

| Feature Name        | Type                    | Missing (%) | Original Dtype -> Final Dtype |
|---------------------|-------------------------|-------------|------------------------------ |
| applicant_income    | Continuous Numerical    | 0.00        | int64 -> float64              |
| coapplicant_income  | Continuous Numerical    | 0.00        | float64                       |
| loan_amount         | Continuous Numerical    | 3.46        | float64                       |
| gender              | Binary Categorical      | 1.02        | object -> category            |
| married             | Binary Categorical      | 0.61        | object -> category            |
| dependents          | Discrete Categorical    | 2.44        | object -> category            |
| self_employed       | Binary Categorical      | 5.30        | object -> category            |
| education           | Binary Categorical      | 0.00        | object -> category            |
| property_area       | Discrete Categorical    | 0.00        | object -> category            |
| loan_amount_term    | Discrete Numerical      | 2.24        | float64 -> int64 -> category  |
| credit_history      | Binary Numerical        | 7.94        | float64 -> int64 -> category  |

## 🧪 Experiment Results

| # | Model   | Accuracy | Precision | Recall | F1 Score | Notes                          |
|---|---------|----------|-----------|--------|----------|--------------------------------|
| 1 | log_reg | 0.81     | 0.79      | 0.84   | 0.81     | Baseline model                 |
| 2 | rfc     | 0.85     | 0.83      | 0.87   | 0.85     | Improved with tuned parameters |
| 3 | xgbc    | 0.87     | 0.85      | 0.89   | 0.87     | Best performance so far        |

## 🛠 Environment & Reproducibility

- **Python Version**: 3.12
- **Operating System**: This project is developed in an Ubuntu-based PC. I tried to maintain this project OS independent. But there is no guarantee that it is. Just a small heads-up.
- **Random Seed**: 37
- **Hardware Limitations**: No. This project is pretty simple and can be run from anywhere, whether it's Local, Google Colab, or Kaggle.
- **Python Dependency Manager**: `uv` 0.6.10
- **Libraries**: For information about the libraries and packages used in this project, checkout [pyproject.toml](../pyproject.toml) file.

Reprodue this project in Ubuntu-based PCs:

```bash
sudo apt update
sudo apt install python3 python3-pip git curl
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/althaf-07/loan-sanction-prediction.git # HTTPS or
git clone git@github.com:althaf-07/loan-sanction-prediction.git  # SSH
```

```bash
cd loan-sanction-prediciton
uv sync
```

Download dataset:

GUI:

1. Visit this [link](https://drive.google.com/file/d/1L87oVoCRJ-JWTHoq1t79vJXwFBEVPvPI/view?usp=drive_link) and download the .csv file to your PC.
2. Next, move the file from your browser downloads folder (usually ~/Downloads) to the project's raw data folder (loan-sanction-prediction/data/raw).

Terminal:

```bash
sudo apt install pipx
pipx install gdown
gdown --fuzzy https://drive.google.com/file/d/1L87oVoCRJ-JWTHoq1t79vJXwFBEVPvPI/view?usp=drive_link
mv entire_data.csv loan-sanction-prediction/data/raw/
```

## 📐 Project Conventions and Glossary

- Features -> The independent variables used to predict the dependent variable
- Target -> The dependent variable that needs to be predicted
- `pl` -> Pipeline
- `ct` -> Column Transformer
- `oe` -> Ordinal Encoder
- `ohe` -> One-Hot Encoder
- `le` -> Label Encoder
- `ss` -> Standard Scaler
- `mms` -> Min-Max Scaler
- `X` -> Features (dataframe)
- `y` -> Target variable (Series)
- `X_train`, `X_test`, etc. -> Split datasets
- `clf` -> Any classification model
- `df` -> This is considered train data unless there is also test data in the code. Then this is mean't to be the merge of both.
- `df_train` -> Train data
- `df_test` -> Test data
- `df_bak` -> Backup (copy)     of the df
- All column names (features + target) and values in categorical columns are expected to be in `snake_case` format.

## 📁 Project Directory Structure

```bash
loan-sanction-prediction/
├── .git/                        # Git version control data
├── .ruff_cache/                 # Cache directory for Ruff (linter)
├── .venv/                       # Local Python virtual environment
├── data/                        # All dataset-related files
│   ├── interim/                 # Intermediate data after partial processing
│   ├── processed/               # Cleaned and ready-to-use data
│   │   ├── test.csv             # Processed test set
│   │   └── train.csv            # Processed train set
│   └── raw/                     # Original unmodified data
│       ├── entire_data.csv      # Combined dataset
│       ├── test.csv             # Raw test data
│       └── train.csv            # Raw train data
├── logs/                        # Logs generated during preprocessing or training
├── models/                      # Trained models and encoders
│   ├── le.joblib                # Label Encoder
│   └── pl.joblib                # Pipeline model object
├── notebooks/                   # Jupyter notebooks for experimentation
│   └── experimentation.ipynb    # Exploratory and experimental notebook
├── reports/                     # Documentation and result summaries
│   └── experimentation.md       # Markdown summary of experimentation process
├── src/                         # Source code for the project
│   ├── loan_sanction_prediction/
│   │   ├── config.yaml          # Configuration file (paths, hyperparameters, etc.)
│   │   ├── evaluate.py          # Evaluation logic and metrics
│   │   ├── predict.py           # Model prediction script
│   │   ├── preprocess_data.py   # Data cleaning and preprocessing
│   │   ├── split_data.py        # Train-test split logic
│   │   ├── streamlit_app.py     # Streamlit app for UI/visualization
│   │   ├── train.py             # Model training script
│   │   └── utils.py             # Utility functions
│   └── loan_sanction_prediction.egg-info/  # Metadata for package distribution
├── tests/                       # Unit tests for each module
│   ├── test_predict.py
│   ├── test_preprocess_data.py
│   ├── test_train.py
│   └── test_utils.py
├── tmp/                         # Temporary files (e.g., intermediate outputs)
├── .env                         # Environment variables (e.g., secrets, config)
├── .envrc                       # Used with direnv for environment setup
├── .gitignore                   # Files and folders to ignore in Git
├── .python-version              # Python version used in the project
├── pyproject.toml               # Project and dependency configuration
├── README.md                    # Project overview and instructions
└── uv.lock                      # Dependency lock file for uv (ultrafast Python package manager)
```

## Others (Things that need to be cleaned)

- dtypes: float64(4), int64(1), object(7)
- Columns:
  - New Name: gender
  - Old Name: Gender
  - Continuous Features:
    - ?

- The target column (`loan_status`) doesn't have any missing values.
- Since we have only a few data points, we cannot afford to drop any of them.
- The percentage of missing values is low across all features, so there is no need to drop any columns.
- The missingness of values appears to be random.
- Some features are not in appropriate data types. So, adjust the data type accordingly after handling missing values
- There are many outliers on the upper side of all numerical features, while none are present on the lower side.
- Since the outliers appear to be valid and are not due to data entry issues, we don't have to drop them.
- None of the numerical features follow a normal distribution.
- The distributions of applicant income and loan amount are right-skewed (positively skewed).
- Feature transformation is required for all numerical features to address this skewness.
- It looks like people with a co-applicant income of 0 doesn't have a co-applicant. So, we should create a new binary feature called 'has_coapplicant'. For this feature, set the value to 'no' for individuals with a co-applicant income of 0, and 'yes' for those with a non-zero co-applicant income.
- A higher number of males apply for loans compared to females.
- Married individuals are more likely to apply for loans than unmarried individuals, with approximately twice as many married applicants.
- Individuals without dependents apply for loans more frequently than those with dependents.
- Graduates are more likely to apply for loans than non-graduates.
- Non-self-employed individuals apply for loans more than self-employed individuals.
- People whose property is located in semi-urban areas tend to apply for loans more than those with properties in rural or urban areas. Those with property in rural areas apply for the fewest loans, although these trends are not very strong.
- The majority of loan applicants prefer a loan term of 360 months (30 years), followed by 180 months (15 years). Other loan term durations are relatively rare.
- Individuals with a credit history of 1 are more likely to apply for loans compared to those with a credit history of 0.
- The classes in target is moderately imbalanced. Need to handle the class imbalance using SMOTE.
- None of the features show a strong linear relationship with each other. However, there is a moderate relationship between applicant income and loan amount. This makes sense because individuals with higher incomes often need larger loan amounts.
- Pearson, Spearman, and Kendall Tau's correlations show similar patterns, but their values are slightly different. Since the heatmaps from all of these are similar, the exact values are less important. In this case, Spearman's correlation is more suitable because the data isn't normally distributed, doesn't have a linear relationship between features, and has outliers.
- There are many outliers on the upper side of all numerical features, while none are present on the lower side.
- Since the outliers appear to be valid and are not due to data entry issues, we don't have to drop them.
- None of the numerical features follow a normal distribution.
- 'loan_amount_term' feature is highly imbalanced. Where only 360 and 180 values take up good amount, while others makes up literally nothing.

How to format this into a nice Markdown table for experimentation findings?
How to programmatically generate a report of the data?
What type of insights should I gather from a dataset? What types of analysis should I do on a dataset?

---

## Missing Value Analysis

## Outlier Analysis

---

## Handle Missing Values

- Numerical features: Use KNN Imputer or Iterative Imputer.
- Categorical features: Use classifier models to predict missing values.

## Handle Outliers

- Use Tree-Based Models. Because, Tree-Based Models are less sensitive to outliers, so handling outliers may not be necessary if we are using these models.
- Apply feature transformations to make the numerical features more normally distributed, which may reduce the impact of outliers.
- Use IQR-Based Capping to cap outliers to a specific range.

After applying these outlier handling methods, evaluate their impact on the model's performance to determine the most effective approach.
