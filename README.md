# Loan Sanction Prediction

- There is one column ("Loan_ID") that is useless for model building.
- There are no duplicate data points in the dataframe.
- There are 614 rows and 12 columns.

### Conclusions

- There are no duplicate data points in the dataset
- The dataset contains 614 data points, 11 feature columns, and `loan_status` as the target column.
- Some features are not in appropriate data types. So, adjust the data type accordingly after handling missing values:
  - `applicant_income`: `float`
  - `loan_amount_term` and `credit_history`: `int` then, `object`
  - categorical features: `category` (This will be helpful in analysis)
  - There are 3 numerical features:
  - `applicant_income`
  - `coapplicant_income`
  - `loan_amount`
- There are 8 categorical features:
  - `gender`
  - `married`
  - `dependents`
  - `education`
  - `self_employed`
  - `property_area`
  - `loan_amount_term`
  - `credit_history`

### Changes that have made to the dataset in this notebook:

- `Loan_ID` is a useless feature for predictive model building. So, dropped it
- The column names were inconsistent and not in a standard format. So, standardized them using Klib
- Reordered column names to have numerical features at the start and categorical features second

#### Conclusions

- There are many outliers on the upper side of all numerical features, while none are present on the lower side.
- Since the outliers appear to be valid and are not due to data entry issues, we don't have to drop them.
- None of the numerical features follow a normal distribution.

---

## Outlier analysis

#### Handle Outliers

- Use Tree-Based Models. Because, Tree-Based Models are less sensitive to outliers, so handling outliers may not be necessary if we are using these models.

- Apply feature transformations to make the numerical features more normally distributed, which may reduce the impact of outliers.

- Use IQR-Based Capping to cap outliers to a specific range.

After applying these outlier handling methods, evaluate their impact on the model's performance to determine the most effective approach.

#### Conclusions:

- Only one numerical feature has missing values:
    - `loan_amount`
- Six categorical features have missing values:
    - `gender`
    - `married`
    - `dependents`
    - `self_employed`
    - `loan_amount_term`
    - `credit_history`
- The target column (`loan_status`) doesn't have any missing values.
- Since we have only a few data points, we cannot afford to drop any of them.
- The percentage of missing values is low across all features, so there is no need to drop any columns.
- The missingness of values appears to be random.

---
## Missing value analysis
#### Handle Missing Values:

- Numerical features: Use KNN Imputer or Iterative Imputer.
- Categorical features: Use classifier models to predict missing values.

# Data Cleaning Steps:

#### 1. Handle Missing Values:

- Numerical features: Use KNN Imputer or Iterative Imputer.
- Categorical features: Use classifier models to predict missing values.

#### 2. Adjust Data Types:

- `applicant_income`: `float`
- `loan_amount_term` and `credit_history`: `int`, then `str`
- categorical features: `category`

#### 3. Handle Outliers:

- Use Tree-Based Models. Because, Tree-Based Models are less sensitive to outliers, so handling outliers may not be necessary if we are using these models.

- Apply feature transformations to make the numerical features more normally distributed, which may reduce the impact of outliers.

- Use IQR-Based Capping to cap outliers to a specific range.

# Data Cleaning Steps:

#### 1. Handle Missing Values:

- Numerical features: Use KNN Imputer or Iterative Imputer.
- Categorical features: Use classifier models to predict missing values.

#### 2. Adjust Data Types:

- `applicant_income`: `float`
- `loan_amount_term` and `credit_history`: `int`, then `str`
- categorical features: `category`

#### 3. Handle Outliers:

- Use Tree-Based Models. Because, Tree-Based Models are less sensitive to outliers, so handling outliers may not be necessary if we are using these models.

- Apply feature transformations to make the numerical features more normally distributed, which may reduce the impact of outliers.

- Use IQR-Based Capping to cap outliers to a specific range.