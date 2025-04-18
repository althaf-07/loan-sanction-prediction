import requests
import numpy as np
import streamlit as st


def main():
    st.title("Loan Sanction Prediction")
    min_value = float(np.finfo(float).tiny)
    max_value = float(np.finfo(float).max)

    # Collect user inputs
    gender = st.radio("Gender", ("Male", "Female"), horizontal=True)
    married = st.radio("Marital Status", ("Married", "Unmarried"), horizontal=True)
    dependents = st.radio(
        "Number of Dependents", ("0", "1", "2", "3+"), horizontal=True
    )
    education = st.radio("Education", ("Graduate", "Not Graduate"))
    self_employed = st.radio("Are you self-employed?", ("Yes", "No"))
    applicant_income = st.number_input(
        "Applicant's Income", min_value=min_value, max_value=max_value
    )
    coapplicant_income = st.number_input(
        "Co-applicant's Income", min_value=min_value, max_value=max_value
    )
    loan_amount = st.number_input(
        "Loan Amount", min_value=min_value, max_value=max_value
    )
    loan_amount_terms = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    loan_amount_term = st.selectbox("Select Loan Amount Term", loan_amount_terms)
    credit_history = st.radio("Credit History", (0, 1))
    property_area = st.radio("Property Area", ("Rural", "Semi-urban", "Urban"))

    data = {
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "dependents": dependents,
        "property_area": property_area,
        "gender": gender,
        "married": married,
        "education": education,
        "self_employed": self_employed,
        "loan_amount_term": float(loan_amount_term),
        "credit_history": float(credit_history)
    }

    if st.button("Predict"):
        with st.spinner("Predicting... please wait"):
            try:
                response = requests.post("http://localhost:8000/prediction", json=data)
                response.raise_for_status()
                if response.status_code == 200:
                    if response.json()["result"]:
                        st.success("The loan can be sanctioned ✅")
                    else:
                        st.error("The loan cannot be sanctioned ❌")
                else:
                    st.error("Something went wrong ❗")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to prediction server. Make sure the API is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
