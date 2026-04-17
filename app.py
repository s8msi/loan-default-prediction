import streamlit as st
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X):
	    X = X.copy()
    
	    X['balance_salary_ratio'] = X['bank_balance'] / X['annual_salary']
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

	# Create dataframe
	input_data = pd.DataFrame([{
		'employed': employed,
		'bank_balance': bank_balance,
		'annual_salary': annual_salary
	}])

	prob = model.predict_proba(input_data)[0][1]

	st.subheader(f"Default Probability: {prob:.2f}")

	if prob > 0.7:
	    st.error("🔴 High Risk - Reject Loan")
	elif prob > 0.3:
	    st.warning("🟡 Medium Risk - Review Required")
	else:
	    st.success("🟢 Low Risk - Approve Loan")
