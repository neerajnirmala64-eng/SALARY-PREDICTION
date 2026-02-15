import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")   # expected: dict of encoders

st.title("Salary Prediction App")

# ---------------- DEBUG BLOCK ----------------
# Uncomment if needed
# st.write("Encoder object:", encoder)
# st.write("Encoder type:", type(encoder))
# if isinstance(encoder, dict):
#     st.write("Encoder keys:", encoder.keys())

# ---------------- USER INPUTS ----------------
age = st.number_input("Age", min_value=18, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    encoder["gender"].classes_ if isinstance(encoder, dict) and "gender" in encoder else []
)

education = st.selectbox(
    "Education Level",
    encoder["education"].classes_ if isinstance(encoder, dict) and "education" in encoder else []
)

job_title = st.selectbox(
    "Job Title",
    encoder["job_title"].classes_ if isinstance(encoder, dict) and "job_title" in encoder else []
)

years_of_exp = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

# ---------------- BUILD DATAFRAME ----------------
df = pd.DataFrame({
    "Age": [age],
    "gender": [gender],
    "education": [education],
    "job_title": [job_title],
    "Years of Experience": [years_of_exp]
})

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    if isinstance(encoder, dict):

        for col, enc in encoder.items():
            if col in df.columns:

                try:
                    df[col] = enc.transform(df[col])

                except ValueError as e:
                    st.error(f"Unknown category in column '{col}': {e}")
                    st.stop()

    try:
        prediction = model.predict(df)
        st.success(f"Predicted Salary: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
