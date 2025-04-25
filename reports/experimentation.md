# Insights from Experimentation Documentation

## 📂 Dataset Information

- **Source**: [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data)
- **Rows**: 614
- **Columns**: 12 (11 features + 1 target)
- **Discrete Features**: 8
- **Continuous Features**: 3
- **Duplicate Records**: None
- **Useless Columns**: `Loan_ID`
- **Memory Usage**: ~57.7 KB

### Target Distribution

- **Y (Loan Approved)**: 422 (≈68.7%)
- **N (Loan Not Approved)**: 192 (≈31.3%)
- **Note**: Slight class imbalance addressed during training (e.g., using SMOTE).

---

## 📊 Feature Summary

| Feature Name        | Type                  | Missing (%) | Dtype Transformation             |
|---------------------|-----------------------|-------------|----------------------------------|
| applicant_income    | Continuous Numerical  | 0.00        | `int64` -> `float64`              |
| coapplicant_income  | Continuous Numerical  | 0.00        | `float64`                        |
| loan_amount         | Continuous Numerical  | 3.46        | `float64`                        |
| gender              | Binary Categorical    | 1.02        | `object` -> `category`            |
| married             | Binary Categorical    | 0.61        | `object` -> `category`            |
| dependents          | Discrete Categorical  | 2.44        | `object` -> `category`            |
| self_employed       | Binary Categorical    | 5.30        | `object` -> `category`            |
| education           | Binary Categorical    | 0.00        | `object` -> `category`            |
| property_area       | Discrete Categorical  | 0.00        | `object` -> `category`            |
| loan_amount_term    | Discrete Numerical    | 2.24        | `float64` -> `int64` -> `category` |
| credit_history      | Binary Numerical      | 7.94        | `float64` -> `int64` -> `category` |

---

## 🧪 Experiment Results

| # | Model   | Hyperparameters | Accuracy | F1 Score | Notes                          |
|---|---------|-----------------|----------|----------|--------------------------------|
| 1 | log_reg | default         | 0.81     | 0.81     | Baseline model                 |
| 2 | rfc     | default         | 0.85     | 0.85     | OK                             |
| 3 | xgbc    | default         | 0.87     | 0.87     | Best performance so far        |

---

## 🔄 Project Conventions & Glossary

### Dataset Variables

- `df`: Main dataset (or combined train + test)
- `df_train`, `df_test`: Training and testing data
- `df_bak`: Backup (copy) of the df
- `X`, `y`: Features and target variable
- `X_train`, `X_test`: Train and test splits of `X`
- `y_train`, `y_test`: Train and test splits of `y`
- `y_pred`: Prediction done by the model

### Model Abbreviations

- `log_reg`: Logistic Regression
- `rfc`: Random Forest Classifier
- `knnc`: KNeighbors Classifier
- `svc`: Support Vector Classifier
- `dtc`: Decision Tree Classifier
- `gbc`: Gradient Boosting Classifier
- `etc`: Extra Trees Classifier

### Preprocessing Terms

- `pl`: Pipeline
- `ct`: Column Transformer
- `oe`: Ordinal Encoder
- `ohe`: One-Hot Encoder
- `le`: Label Encoder
- `ss`: Standard Scaler
- `mms`: Min-Max Scaler
- `clf`: Classifier model

> All column names and values in categorical columns follow `snake_case` format.

---

## 📌 Exploratory Data Analysis (EDA) Insights

### Univariate Analysis

#### Numerical

- None of the numerical features follow a normal distribution.
- The distributions of applicant income and loan amount are right-skewed (positively skewed).
- Feature transformation is required for all numerical features to address this skewness.
- There are many outliers on the upper side of all numerical features, while none are present on the lower side.
- Since the outliers appear to be valid and are not due to data entry issues, we don't have to drop them.
- It looks like people with a co-applicant income of 0 doesn't have a co-applicant. So, we should create a new binary feature called 'has_coapplicant'. For this feature, set the value to 'no' for individuals with a co-applicant income of 0, and 'yes' for those with a non-zero co-applicant income.
- Use Tree-Based Models. Because, Tree-Based Models are less sensitive to outliers, so handling outliers may not be necessary if we are using these models.
- Apply feature transformations to make the numerical features more normally distributed, which may reduce the impact of outliers.
- Use IQR-Based Capping to cap outliers to a specific range.
- After applying these outlier handling methods, evaluate their impact on the model's performance to determine the most effective approach.

#### Categorical

- More male applicants than female.
- Married applicants are more than unmarried applicants (~2x more).
- Most applicants have no dependents.
- More graduates than non-graduates.
- Most applicants are non-self-employed than self-employed.
- Semi-urban areas dominate in `property_area`. Although other categories are also close.
- Loan terms are mostly 360 months (30 years), followed by 180 months (15 years). While others makes up literally nothing.
- Credit history of 1 is common among applicants.

### Bivariate Analysis

#### Numerical

- None of the features show a strong linear relationship with each other. However, there is a moderate relationship between applicant income and loan amount. This makes sense because individuals with higher incomes often need larger loan amounts.
- Pearson, Spearman, and Kendall Tau's correlations show similar patterns, but their values are slightly different. Since the heatmaps from all of these are similar, the exact values are less important. In this case, Spearman's correlation is more suitable because the data isn't normally distributed, doesn't have a linear relationship between features, and has outliers.

## Missing Value Analysis

- The target column (`loan_status`) doesn't have any missing values.
- Since we have only a few data points, we cannot afford to drop any of them.
- The percentage of missing values is low across all features, so there is no need to drop any columns.
- The missingness of values appears to be random.

---
