# Install inflecion and flash library
!pip install inflection git+https://github.com/flash-lib/flash.git -q

# Standard Libraries
import os

# Data Manipulation
import numpy as np
import pandas as pd

# Data Preprocessing
import inflection
import flash as fz

# Data Transformation
from sklearn.impute import SimpleImputer

# Model evaluation
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Change directory to current working directory
%cd /content/drive/MyDrive/Projects/loan-sanction-prediction

# Load the dataset
df = pd.read_csv('data/raw/loan_sanction_train.csv')

# Create a backup of the original dataset
df_copy = df.copy()

df.drop(columns=['Loan_ID'], inplace=True)

# Test
print(df.columns)

df.columns = [inflection.underscore(col) for col in df.columns]

# Test
print(df.columns)

# # Extract numerical, categorical, and other features from the dataset
# num_cols, cat_cols, other_cols = fz.extract_features(df, 'all', ignore_cols=['loan_status'])

# # Test
# print(num_cols)
# print(cat_cols)

num_cols = ['applicant_income', 'coapplicant_income', 'loan_amount']
cat_cols = [
    'gender', 'married', 'dependents', 'education', 'self_employed', 'property_area',
    'loan_amount_term', 'credit_history'
    ]
target_col = ['loan_status']

df = df[num_cols + cat_cols + target_col]

# Test
df.head()

def cap_outliers_std(data, cap=3):
    mean = np.mean(data)
    std = np.std(data)

    lower_bound = mean - cap * std
    upper_bound = mean + cap * std

    capped_data = np.clip(data, lower_bound, upper_bound)
    return capped_data

df_1 = pd.DataFrame()
df_1['applicant_income'] = cap_outliers_std(df['applicant_income'])
df_1['loan_amount'] = cap_outliers_std(df['loan_amount'])

fz.hist_box_viz(df_1, ['applicant_income', 'loan_amount'])

X = df.drop(columns=['loan_status'])

y = df['loan_status']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y = pd.Series(y_encoded, name=y.name)

random_state = 42
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=random_state),
    'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
    'Support Vector Machine': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Trees': DecisionTreeClassifier(random_state=random_state),
    'Xgboost': XGBClassifier(),
    'Extra Trees': ExtraTreesClassifier(random_state=random_state)
}

X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

from typing import Optional, Literal

model_mapping = {
            'linear': LogisticRegression(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'tree': DecisionTreeClassifier(random_state=42),
            'forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, random_state=42)
        }

def impute(
        df: pd.DataFrame,
        column: str,
        type: Literal['num', 'cat'] = 'num',
        method = None,
        n_neighbors = 5
        ) -> pd.DataFrame:

    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if method == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    elif method == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    elif method == 'mode':
        df[column].fillna(df[column].mode()[0], inplace=True)
    elif method == 'constant':
        df[column].fillna('missing', inplace=True)
    elif method in ['ffill', 'bfill']:
        df[column] = df[column].ffill().bfill() if method == 'ffill' else df[column].bfill().ffill()
    elif method == 'knn_imputer':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[column] = imputer.fit_transform(df[[column]])
    else:
        raise ValueError(
            f"Method '{method}' not recognized. Use 'mean', 'median', 'ffill', 'bfill',\
            'knn_imputer', 'iterative_imputer', or an instance of a ML model."
            )

        train_data = df[df[column].notna()]
        test_data = df[df[column].isna()]

        if test_data.empty:
            print(f"No missing values to impute in {column}.")
            return df

        feature_columns = df.drop(columns=[column]).columns
        X_train = train_data[feature_columns]
        y_train = train_data[column]
        X_test = test_data[feature_columns]

        model.fit(X_train, y_train)
        df.loc[df[column].isna(), column] = model.predict(X_test)

    return df

    for feature in cat_cols:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

    df['loan_amount'].fillna(df['loan_amount'].mean(), inplace=True)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

def eval_nan_value(
        X,
        y,
        feature = None,
        type: Literal['num', 'cat'] = 'num',
        methods = None,
        models = None,
        n_neighbors = 5
        ):

    # Create a copy of the original DataFrame and concatenate with the target
    df = pd.concat([X.copy(), y], axis=1)

    # Default methods if not provided
    if methods is None:
        if type == 'num':
            methods = ['mean', 'median', 'constant', 'ffill', 'bfill', 'knn_imputer', 'iterative_imputer']
        elif type == 'cat':
            methods = ['mode', 'constant', 'ffill', 'bfill', 'knn_imputer', 'iterative_imputer']

    # Initialize results dictionary and accuracies
    results_dict = {'Model': list(models.keys()), **{method: [] for method in methods}}

    for method in methods:
        X_imputed = impute_numerical(df, feature, method=method, n_neighbors=n_neighbors).drop(y.name, axis=1)

        # Evaluate each model with cross-validation
        for model in models.values():
            cv_scores = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy')
            results_dict[method].append(cv_scores.mean() * 100)

    # Convert the results_dict to a DataFrame
    results = pd.DataFrame(results_dict)
    results.loc[len(results)] = ['Mean Accuracy'] + results.iloc[:, 1:].mean().tolist()
    results = results.set_index('Model')

    return results

    def evaluate_models(X, y, feature = None, methods = None, models=None, n_neighbors = 5):
    # Create a copy of the original DataFrame and concatenate with the target
    df = pd.concat([X.copy(), y], axis=1)

    # Default methods if not provided
    if methods is None:


    # Initialize results dictionary and accuracies
    results_dict = {'Model': list(models.keys()), **{method: [] for method in methods}}

    for method in methods:
        X_imputed = impute_categorical(df, feature, method=method, n_neighbors=n_neighbors).drop(y.name, axis=1)
        print(X_imputed)

        # Evaluate each model with cross-validation
        for model in models.values():
            cv_scores = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy')
            results_dict[method].append(cv_scores.mean() * 100)

    # Convert the results_dict to a DataFrame
    results = pd.DataFrame(results_dict)
    results.loc[len(results)] = ['Mean Accuracy'] + results.iloc[:, 1:].mean().tolist()
    results = results.set_index('Model')

    return results

evaluate_models(X, y, 'loan_amount', models=models)

# # Categorical features with misssing values
# cat_features_with_na = fz.calc_na_values(df, cat_cols).index.tolist()

# print(cat_features_with_na)

cat_features_with_na = [
    'gender', 'married', 'dependents', 'self_employed', 'loan_amount_term',
    'credit_history'
    ]

X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)





evaluate_models(X, y, 'gender', models=models)

# Test
if df.isna().sum().sum() == 0:
    print("There are no missing values left in the DataFrame.")
else:
    print("There are still missing values in the DataFrame.")

df['applicant_income'] = df['applicant_income'].astype(float)

# Converting numerical categorical features to int
df['loan_amount_term'] = df['loan_amount_term'].astype(int)
df['credit_history'] = df['credit_history'].astype(int)

# Test
print(df.dtypes)

def export(df, filename, force_overwrite=False):
    if force_overwrite:
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
    else:
        # Check if the file already exists
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
        else:
            print(f"File {filename} already exists. Choose a different name or use force_overwrite=True to overwrite.")

export(df, 'data/interim/cleaned_data_v1.csv')
