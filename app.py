
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CancerGuard",
    page_icon="🧬",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 40px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ================= LOAD MODEL =================
MODEL_PATH = Path(__file__).parent / "cancerGuard.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file cancerGuard.pkl was not found. "
        "Keep it in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error("Unable to load the saved model.")
    st.code(str(error))
    st.stop()

# ================= TITLE =================
st.markdown(
    '<p class="title">🧬 CancerGuard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Cancer Growth Risk Prediction System</p>',
    unsafe_allow_html=True
)

st.info(
    "Enter the details below to obtain a prediction from your "
    "saved machine-learning model. For educational use only."
)

# ================= PATIENT DETAILS =================
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Patient Details")

    age = st.slider(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=40,
        key="patient_age"
    )

    previous_size = st.slider(
        "Previous Tumor Size (cm)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1,
        key="previous_tumor_size"
    )

    time_since = st.slider(
        "Time Since Diagnosis (months)",
        min_value=0,
        max_value=120,
        value=12,
        key="time_since_diagnosis"
    )

    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=22.0,
        step=0.1,
        key="patient_bmi"
    )

with right_col:
    st.subheader("Medical Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        key="patient_gender"
    )

    stage = st.selectbox(
        "Cancer Stage",
        ["Stage I", "Stage II", "Stage III", "Stage IV"],
        key="cancer_stage"
    )

    genetic_risk = st.selectbox(
        "Genetic Risk",
        ["Low", "Medium", "High"],
        key="genetic_risk"
    )

    smoking = st.selectbox(
        "Smoking History",
        ["No", "Yes"],
        key="smoking_history"
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["Low", "Medium", "High"],
        key="alcohol_consumption"
    )

    st.subheader("Treatment Information")

    chemotherapy = st.selectbox(
        "Chemotherapy",
        ["No", "Yes"],
        key="chemotherapy"
    )

    radiotherapy = st.selectbox(
        "Radiotherapy",
        ["No", "Yes"],
        key="radiotherapy"
    )

    surgery = st.selectbox(
        "Surgery",
        ["No", "Yes"],
        key="surgery"
    )

# ================= INPUT DATA =================
def build_input_data():
    return {
        "Age": age,
        "Gender": 0 if gender == "Male" else 1,
        "Previous_Tumor_Size": previous_size,
        "Time_Since_Diagnosis": time_since,
        "Stage": {
            "Stage I": 1,
            "Stage II": 2,
            "Stage III": 3,
            "Stage IV": 4
        }[stage],
        "Genetic_Risk": {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[genetic_risk],
        "Smoking_History": 0 if smoking == "No" else 1,
        "Alcohol_Consumption": {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[alcohol],
        "BMI": bmi,
        "Treatment_Chemotherapy": (
            1 if chemotherapy == "Yes" else 0
        ),
        "Treatment_Radiotherapy": (
            1 if radiotherapy == "Yes" else 0
        ),
        "Treatment_Surgery": (
            1 if surgery == "Yes" else 0
        )
    }

# ================= PREDICTION =================
st.divider()
st.subheader("Cancer Growth Prediction")

if st.button("Predict Tumor Growth Rate", type="primary"):

    input_data = pd.DataFrame([build_input_data()])

    # Find expected features from the saved model.
    expected_features = getattr(
        model, "feature_names_in_", None
    )

    # Also support common Scikit-learn pipelines.
    if expected_features is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            expected_features = getattr(
                step, "feature_names_in_", None
            )
            if expected_features is not None:
                break

    if expected_features is not None:
        expected_features = list(expected_features)

        missing = [
            feature for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing:
            st.error(
                "The saved model expects features that are not "
                "provided by this form."
            )
            st.write("Missing model features:", missing)
            st.info(
                "The input names and encoding must match the "
                "dataset and preprocessing used during training."
            )
            st.stop()

        input_data = input_data[expected_features]

    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Tumor Growth Rate: {prediction:.2f}"
            if isinstance(prediction, (int, float))
            else f"Model Output: {prediction}"
        )

        # Show probabilities only for classification models.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            st.subheader("Model Class Probabilities")

            probability_data = pd.DataFrame({
                "Class": model.classes_,
                "Probability (%)": [
                    round(float(probability) * 100, 2)
                    for probability in probabilities
                ]
            })

            st.dataframe(
                probability_data,
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))

st.divider()

st.caption(
    "CancerGuard | Educational Machine-Learning Project. "
    "This output is not a medical diagnosis or a validated "
    "clinical risk estimate."
)
