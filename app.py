import streamlit as st
import numpy as np
import joblib

# Load trained pipeline (IMPORTANT: save full pipeline, not just model)
model = joblib.load("addiction_pipeline.pkl")

st.title("Smartphone Addiction Prediction")

st.write("Input user behavior data")

# ---- Inputs (based on your dataset structure) ----

daily_screen_time_hours = st.number_input("Daily Screen Time (hours)", 0.0, 24.0, 5.0)
weekend_screen_time = st.number_input("Weekend Screen Time (hours)", 0.0, 24.0, 6.0)
social_media_hours = st.number_input("Social Media Hours", 0.0, 24.0, 3.0)
gaming_hours = st.number_input("Gaming Hours", 0.0, 24.0, 1.0)
work_study_hours = st.number_input("Work/Study Hours", 0.0, 24.0, 6.0)
sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
age = st.number_input("Age", 10, 80, 20)
app_opens_per_day = st.number_input("App Opens Per Day", 0, 500, 50)

# Encoded fields (based on your mapping)
academic_work_impact = st.selectbox("Academic Work Impact", ["No", "Yes"])
stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])

# ---- Encoding (must match notebook exactly) ----
academic_work_impact = 1 if academic_work_impact == "Yes" else 0

stress_map = {"Low": 0, "Medium": 1, "High": 2}
stress_level = stress_map[stress_level]

# ---- Prediction ----
if st.button("Predict"):

    input_data = np.array([[
        daily_screen_time_hours,
        weekend_screen_time,
        social_media_hours,
        gaming_hours,
        work_study_hours,
        sleep_hours,
        age,
        app_opens_per_day,
        academic_work_impact,
        stress_level
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error("Addicted User")
    else:
        st.success("Non-Addicted User")

    st.write(f"Confidence: {max(prob)*100:.2f}%")