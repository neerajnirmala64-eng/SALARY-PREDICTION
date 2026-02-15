import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Salary Prediction App")

# ---- DEBUG (optional but very useful) ----
# Uncomment if you ever get errors again
# st.write("Encoder keys:", encoder.keys())
# st.write("Encoder type:", type(encoder))

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    encoder["gender"].classes_ if "gender" in encoder else []
)

education = st.selectbox(
    "Education Level",
    encoder["education"].classes_ if "education" in encoder else []
)

job_title = st.selectbox(
    "Job Title",
    encoder["job_title"].classes_ if "job_title" in encoder else []
)

years_of_exp = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

# Build DataFrame (column names MUST match training)
df = pd.DataFrame({
    "Age": [age],
    "gender": [gender],
    "education": [education],
    "job_title": [job_title],
    "Years of Experience": [years_of_exp]
})

if st.button("Predict"):

    # Apply encoders safely
    for col, enc in encoder.items():
        if col in df.columns:
            df[col] = enc.transform(df[col])

    prediction = model.predict(df)

    st.success(f"Predicted Salary: {prediction[0]:,.2f}")
