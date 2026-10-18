import streamlit as st
import pandas as pd
import pickle

# Load saved model
model = pickle.load(open("cancerGuard.pkl", "rb"))



st.title("🧬 Tumor Growth Rate Prediction")

st.write("Enter patient details below to predict tumor growth rate:")

# User Inputs with Sliders
age = st.slider("Age", 1, 100, 40)
gender = st.radio("Gender", ["Male", "Female"])
previous_size = st.slider("Previous Tumor Size (cm)", 0.0, 20.0, 5.0, step=0.1)
time_since = st.slider("Time Since Diagnosis (months)", 0, 120, 12)
stage = st.slider("Stage (I=1, II=2, III=3, IV=4)", 1, 4, 2)
genetic_risk = st.slider("Genetic Risk (0=Low, 1=Medium, 2=High)", 0, 2, 1)
smoking = st.radio("Smoking History", ["No", "Yes"])
alcohol = st.slider("Alcohol Consumption (0=Low, 1=Medium, 2=High)", 0, 2, 1)
bmi = st.slider("BMI", 10.0, 50.0, 22.0, step=0.1)

treatment_chemo = st.checkbox("Chemotherapy")
treatment_radio = st.checkbox("Radiotherapy")
treatment_surgery = st.checkbox("Surgery")

# Convert categorical to numeric
gender_val = 0 if gender == "Male" else 1
smoking_val = 1 if smoking == "Yes" else 0

# Create dataframe for model
user_data = pd.DataFrame([[
    age, gender_val, previous_size, time_since, stage, genetic_risk,
    smoking_val, alcohol, bmi,
    int(treatment_chemo), int(treatment_radio), int(treatment_surgery)
]], columns=model.feature_names_in_)

# Predict
if st.button("Predict Tumor Growth Rate"):
    prediction = model.predict(user_data)[0]
    st.success(f"🎯 Predicted Tumor Growth Rate: {round(prediction, 2)}")