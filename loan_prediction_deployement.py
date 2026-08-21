import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Loan_Prediction_model.pkl")
encoder_data = joblib.load("label_encoder_loan.pkl")

feature_encoders = encoder_data["feature_encoders"]
target_encoder = encoder_data["target_encoder"]
feature_columns = encoder_data["feature_columns"]

st.title("Loan Prediction")

gender = st.selectbox(
    "Gender",
    feature_encoders["Gender"].classes_
)

married = st.selectbox(
    "Married",
    feature_encoders["Married"].classes_
)

dependents = st.selectbox(
    "Dependents",
    feature_encoders["Dependents"].classes_
)

education = st.selectbox(
    "Education",
    feature_encoders["Education"].classes_
)

self_employed = st.selectbox(
    "Self Employed",
    feature_encoders["Self_Employed"].classes_
)

applicantincome = st.number_input("Applicant Income", min_value=0.0)
coapplicantincome = st.number_input("Coapplicant Income", min_value=0.0)
loanamount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0)
credit_history = st.number_input("Credit History", min_value=0.0, max_value=1.0)

property_area = st.selectbox(
    "Property Area",
    feature_encoders["Property_Area"].classes_
)

df_input = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicantincome],
    "CoapplicantIncome": [coapplicantincome],
    "LoanAmount": [loanamount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

if st.button("Predict Loan"):

    for col in feature_encoders:

        if col in df_input.columns:

            df_input[col] = feature_encoders[col].transform(
                df_input[col]
            )

    prediction = model.predict(df_input)

   
    result = target_encoder.inverse_transform(prediction)

    st.success(
        f"Predicted Loan Status: {result[0]}"
    )
