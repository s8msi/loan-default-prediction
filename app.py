import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['balance_salary_ratio'] = X['bank_balance'] / (X['annual_salary'] + 1)
        X['debt_indicator'] = (X['bank_balance'] < 5000).astype(int)
        X['low_income_flag'] = (X['annual_salary'] < 30000).astype(int)

        X['risk_score'] = (
            (X['bank_balance'] < 5000)*2 +
            (X['annual_salary'] < 30000)*1 +
            (X['employed'] == 0)*2
        )

        X['employed_balance'] = X['employed'] * X['bank_balance']
        X['log_balance'] = np.log1p(X['bank_balance'])
        X['log_salary'] = np.log1p(X['annual_salary'])

        return X

model = joblib.load("credit_risk_model.pkl")

st.title("💳 Credit Risk Prediction App")
st.write("Enter customer details to predict default risk")

employed = st.selectbox("Employment Status (0 = No, 1 = Yes)", [0, 1])
bank_balance = st.number_input("Bank Balance", min_value=0.0)
annual_salary = st.number_input("Annual Salary", min_value=0.0)

if st.button("Predict Risk"):

	input_data = pd.DataFrame([{
	    'employed': employed,
	    'bank_balance': bank_balance,
	    'annual_salary': annual_salary
	}])

	prob = model.predict_proba(input_data)[0][1]

	st.write("Raw Probability:", prob)
	st.subheader(f"Default Probability: {prob:.4f}")

	if prob > 0.07:
	    st.error("🔴 High Risk - Reject Loan")
	elif prob > 0.02:
	    st.warning("🟡 Medium Risk - Review Required")
	else:
	    st.success("🟢 Low Risk - Approve Loan")

	if prob < 0.02:
	    st.info("Typical customer range (low risk group)")
	elif prob < 0.07:
	    st.info("Above average risk compared to population")
	else:
	    st.info("Top risk segment (highest risk group)")
