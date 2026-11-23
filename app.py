import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Cardion — 10-year CHD risk",
    page_icon="❤️",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent
MODELS_DIR = DATA_DIR / "Models"

st.markdown(
    """
    <div style="
        border: 3px solid #0b3d91;
        border-radius: 12px;
        padding: 28px;
        background-color: white;
        margin-bottom: 25px;
    ">
        <h1 style="
            color: #DC143C;
            font-size: 60px;
            font-weight: 900;
            text-align: center;
            margin: 0;
        ">
            Cardion
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with (MODELS_DIR / "rf_tuned2.pkl").open("rb") as file:
        model = pickle.load(file)

    with (MODELS_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)

    with (MODELS_DIR / "feature_columns.json").open(
        "r", encoding="utf-8"
    ) as file:
        feature_cols = json.load(file)

except FileNotFoundError as error:
    st.error(f"Missing file: {error.filename}")
    st.stop()

st.write("Machine Learning Based 10-Year Heart Disease Risk Prediction System")

left_col, right_col = st.columns(2)

with left_col:
    image_path = DATA_DIR / "heart.png"

    if image_path.is_file():
        with Image.open(image_path) as image:
            st.image(image.copy(), use_container_width=True, caption="Cardion")
    else:
        st.info("Place heart.png in the same folder as app.py.")

with right_col:
    st.subheader("Enter your data to predict the risk")

    with st.form("cardion_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=20, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox(
                "Highest education level",
                [
                    "School Leaver",
                    "GCE O/L Passed",
                    "GCE A/L Passed",
                    "Graduated and above",
                ],
            )
            current_smoker = st.selectbox("Currently smoking?", ["No", "Yes"])
            cigarettes_per_day = st.number_input(
                "Cigarettes per day", min_value=0, max_value=100, value=0
            )
            bp_medication = st.selectbox("On BP medication?", ["No", "Yes"])
            prevalent_stroke = st.selectbox("Prevalent stroke?", ["No", "Yes"])
            prevalent_hypertension = st.selectbox(
                "Prevalent hypertension?", ["No", "Yes"]
            )

        with col2:
            diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
            total_cholesterol = st.number_input(
                "Total cholesterol", min_value=50, max_value=800, value=200
            )
            systolic_bp = st.number_input(
                "Systolic BP", min_value=60, max_value=300, value=120
            )
            diastolic_bp = st.number_input(
                "Diastolic BP", min_value=30, max_value=200, value=80
            )
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
            heart_rate = st.number_input(
                "Heart rate", min_value=30, max_value=220, value=70
            )
            glucose = st.number_input(
                "Glucose", min_value=10, max_value=500, value=85
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        education_levels = {
            "School Leaver": 1,
            "GCE O/L Passed": 2,
            "GCE A/L Passed": 3,
            "Graduated and above": 4,
        }
        education_level = education_levels[education]

        input_values = {
            "male": int(gender == "Male"),
            "age": age,
            "currentSmoker": int(current_smoker == "Yes"),
            "cigsPerDay": cigarettes_per_day,
            "BPMeds": int(bp_medication == "Yes"),
            "prevalentStroke": int(prevalent_stroke == "Yes"),
            "prevalentHyp": int(prevalent_hypertension == "Yes"),
            "diabetes": int(diabetes == "Yes"),
            "totChol": total_cholesterol,
            "sysBP": systolic_bp,
            "diaBP": diastolic_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose,
            "edu_1.0": int(education_level == 1),
            "edu_2.0": int(education_level == 2),
            "edu_3.0": int(education_level == 3),
            "edu_4.0": int(education_level == 4),
            "PulsePressure": systolic_bp - diastolic_bp,
            "CholAgeRatio": total_cholesterol / (age + 1e-6),
        }

        missing_features = [
            column for column in feature_cols if column not in input_values
        ]

        if missing_features:
            st.error(
                "The app cannot supply these model features: "
                + ", ".join(missing_features)
            )
        else:
            input_row = pd.DataFrame([input_values], columns=feature_cols)
            scaled_values = scaler.transform(input_row)
            scaled_row = pd.DataFrame(scaled_values, columns=feature_cols)

            prediction = model.predict(scaled_row)[0]
            positive_class_index = model.classes_.tolist().index(1)
            score = model.predict_proba(scaled_row)[0, positive_class_index]

            if prediction == 1:
                st.warning("Model result: High Risk")
            else:
                st.success("Model result: Low Risk")

            st.metric("Model score", f"{score:.2%}")

st.markdown(
    "<footer>Cardion 2025 Nov © </footer>",
    unsafe_allow_html=True,
)
