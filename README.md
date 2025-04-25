# Loan Sanction Prediction

![Python](https://img.shields.io/badge/python-3.12-blue)
![Last Commit](https://img.shields.io/github/last-commit/althaf-07/loan-sanction-prediction)

## 📌 Project Overview

- **Project Name**: Loan Sanction Prediction
- **Problem Statement**: Loan applications can take time and are often rejected without clear reasons. Financial institutions need a reliable way to assess whether a loan should be sanctioned based on applicant data.
- **Objective**: Build a machine learning model to predict whether a loan will be sanctioned (approved) using applicant and loan-related features.
- **Type of Problem**: Binary Classification
- **Target Column**: `loan_status`
- **Success Metrics**: Since no class needs to be prioritized (as of I know), Accuracy Score (after handling class imbalance) and F1-Score are better candidates for evaluation metrics.

## 🚀 Getting Started

To set up the project and get it running locally, follow the steps in [reports/environment.md](reports/environment.md).

## 📁 Project Directory Structure

```bash
loan-sanction-prediction/
├── .git/                        # Git version control data
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
│   └── loan_sanction_prediction/
│       ├── config.yaml          # Configuration file (paths, hyperparameters, etc.)
│       ├── evaluate.py          # Evaluation logic and metrics
│       ├── predict.py           # Model prediction script
│       ├── preprocess_data.py   # Data cleaning and preprocessing
│       ├── split_data.py        # Train-test split logic
│       ├── streamlit_app.py     # Streamlit app for UI/visualization
│       ├── train.py             # Model training script
│       └── utils.py             # Utility functions
├── tests/                       # Unit tests for each module
│   ├── test_predict.py
│   ├── test_preprocess_data.py
│   ├── test_train.py
│   └── test_utils.py
├── tmp/                         # Temporary files (e.g., intermediate outputs)
├── .envrc                       # Used with direnv for environment setup
├── .gitignore                   # Files and folders to ignore in Git
├── .python-version              # Python version used in the project
├── pyproject.toml               # Project and dependency configuration
├── README.md                    # Project overview and instructions
└── uv.lock                      # Dependency lock file for uv (ultrafast Python package manager)
```

## 🔮 Future Enhancements

- [ ] ⚙️ Integrate SHAP for model explainability
- [ ] 📦 Add a `Dockerfile` for easier environment setup and deployment
- [ ] ☁️ Deploy the application on a GCP Virtual Machine or App Engine
- [ ] 🔁 Add CI with GitHub Actions to automate linting and tests
- [ ] 📊 Visualize training metrics and SHAP values in the Streamlit UI
