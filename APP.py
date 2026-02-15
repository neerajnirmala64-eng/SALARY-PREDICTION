import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary Prediction App")

age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender",encoder["Gender"].classes_)
education = st.selectbox("Education Level",encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title",encoder["Job Title"].classes_)
years_of_exp = st.number_input("Years of Experience", 0, 40)

# Create dataframe correctly
df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [years_of_exp]
})

if st.button("Predict"):
    # Create a copy so we don't overwrite the original numeric data by accident
    input_df = df.copy()

    # Apply encoders only to the categorical columns
    for col, transformer in encoder.items():
        if col in input_df.columns:
            input_df[col] = transformer.transform(input_df[col])
    
    # Ensure the column order matches what the model saw during training
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
