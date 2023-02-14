import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

st.write("# Loan Prediction")
st.write("Enter the following information to predict if a loan would be approved or not")

loan_type = st.selectbox("Select the type of loan you want to apply for",
                         ["Personal Loans", "Auto Loans", "Student Loans",
                          "Mortgage Loans", "Home Equity Loans", "Credit-Builder Loans",
                          "Debt Consolidation Loans", "Payday Loans"])

credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=500)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Desired Loan Amount", min_value=0, value=10000)
repaid_loans = st.number_input("Number of Repaid Loans", min_value=0, value=0)
years_of_employment = st.number_input("Years of Employment", min_value=0, value=1)

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation="relu", input_shape=(5,)))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if st.button("Predict"):
    features = np.array([credit_score, annual_income, loan_amount, repaid_loans, years_of_employment]).reshape(1, -1)

    model = create_model()
    prediction = model.predict(features)

    if prediction > 0.5:
        st.write("Loan approved")
    else:
        st.write("Loan declined")
