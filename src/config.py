# Configuration for numerical and categorical features

# Numerical features
NUMERIC_FEATURES = {
    "cols": ["applicant_income", "coapplicant_income", "loan_amount"],
    "nan": ["loan_amount"]
}

# Categorical features
CATEGORICAL_FEATURES = {
    "cols": [
        "gender", "married", "dependents", "education", 
        "self_employed", "property_area", "loan_amount_term", 
        "credit_history", "has_coapplicant"
    ],
    "nan": ["gender", "married", "dependents", "self_employed", "loan_amount_term", "credit_history"]
}
