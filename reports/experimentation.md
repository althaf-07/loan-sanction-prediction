## Data Assessment and Cleaning

- There is one column ("Loan_ID") that is useless for model building.
- There are no duplicate data points in the dataframe.
- There are 614 rows and 12 columns.

## EDA

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

### Conclusions

- There are many outliers on the upper side of all numerical features, while none are present on the lower side.
- Since the outliers appear to be valid and are not due to data entry issues, we don't have to drop them.
- None of the numerical features follow a normal distribution.
- The distributions of applicant income and loan amount are right-skewed (positively skewed).
- Feature transformation is required for all numerical features to address this skewness.
- It looks like people with a co-applicant income of 0 doesn't have a co-applicant. So, we should create a new feature called 'has_coapplicant'. For this feature, set the value to 'no' for individuals with a co-applicant income of 0, and 'yes' for those with a non-zero co-applicant income.

---

- A higher number of males apply for loans compared to females.
- Married individuals are more likely to apply for loans than unmarried individuals, with approximately twice as many married applicants.
- Individuals without dependents apply for loans more frequently than those with dependents.
- Graduates are more likely to apply for loans than non-graduates.
- Non-self-employed individuals apply for loans more than self-employed individuals.
- People whose property is located in semi-urban areas tend to apply for loans more than those with properties in rural or urban areas. Those with property in rural areas apply for the fewest loans, although these trends are not very strong.
- The majority of loan applicants prefer a loan term of 360 months (30 years), followed by 180 months (15 years). Other loan term durations are relatively rare.
- Individuals with a credit history of 1 are more likely to apply for loans compared to those with a credit history of 0.

---

- The classes in target is moderately imbalanced. Need to handle the class imbalance using SMOTE.

---

### Conclusions

- None of the features show a strong linear relationship with each other. However, there is a moderate relationship between applicant income and loan amount. This makes sense because individuals with higher incomes often need larger loan amounts.
- Pearson, Spearman, and Kendall
Tau's correlations show similar patterns, but their values are slightly different. Since the heatmaps from all of these are similar, the exact values are less important. In this case, Spearman's correlation is more suitable because the data isn't normally distributed, doesn't have a linear relationship between features, and has outliers.
