
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
=======
import streamlit as st
import pandas as pd
import pickle

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
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= LOAD MODEL =================
try:
    with open("cancerGuard.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error(
        "cancerGuard.pkl not found. "
        "Keep it in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load model: {e}")
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
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Patient Details")

    age = st.slider(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )

    previous_size = st.slider(
        "Previous Tumor Size",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

    time_since = st.slider(
        "Time Since Diagnosis",
        min_value=0,
        max_value=120,
        value=12
    )

    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=22.0,
        step=0.1
    )

with right_col:
    st.subheader("Medical Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    stage = st.selectbox(
        "Cancer Stage",
        ["Stage I", "Stage II", "Stage III", "Stage IV"]
    )

    genetic_risk = st.selectbox(
        "Genetic Risk",
        ["Low", "Medium", "High"]
    )

    smoking = st.selectbox(
        "Smoking History",
        ["No", "Yes"]
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    st.subheader("Treatment Information")

    chemotherapy = st.selectbox(
        "Chemotherapy",
        ["No", "Yes"]
    )

    radiotherapy = st.selectbox(
        "Radiotherapy",
        ["No", "Yes"]
    )

    surgery = st.selectbox(
        "Surgery",
        ["No", "Yes"]
    )

# ================= PREDICTION =================
st.divider()
st.subheader("Cancer Growth Risk Prediction")

if st.button("Predict Risk", type="primary"):

    # Create input with the feature names identified so far.
    input_data = pd.DataFrame([{
        "Age": age,
        "Previous_Tumor_Size": previous_size,
        "Time_Since_Diagnosis": time_since,
        "BMI": bmi,
        "Gender": 1 if gender == "Female" else 0,
        "Stage": {
          "Stage I": 1,
          "Stage II": 2,
          "Stage III": 3,
          "Stage IV": 4
        }[stage],
        "Genetic_Risk": {
          "Low": 1,
          "Medium": 2,
          "High": 3
        }[genetic_risk],
        "Smoking_History": 1 if smoking == "Yes" else 0,
        "Alcohol_Consumption": 1 if alcohol == "Yes" else 0,
        "Treatment_Chemotherapy": (
            1 if chemotherapy == "Yes" else 0
        ),
        "Treatment_Radiotherapy": (
            1 if radiotherapy == "Yes" else 0
        ),
        "Treatment_Surgery": (
            1 if surgery == "Yes" else 0
        )
    }])

    # Find the model's expected feature names, if available.
    expected_features = getattr(
        model, "feature_names_in_", None
    )

    if expected_features is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            expected_features = getattr(
                step, "feature_names_in_", None
            )
            if expected_features is not None:
                break

    if expected_features is not None:
        expected_features = list(expected_features)

        missing_features = [
            col for col in expected_features
            if col not in input_data.columns
        ]

        extra_features = [
            col for col in input_data.columns
            if col not in expected_features
        ]

        if missing_features or extra_features:
            st.error("Model feature mismatch detected.")

            if missing_features:
                st.write("Missing features:", missing_features)

            if extra_features:
                st.write("Unexpected features:", extra_features)

            st.info(
                "The input fields must match the training dataset. "
                "Add missing fields and use the same encoding as "
                "the model training code."
            )
            st.stop()

        input_data = input_data[expected_features]

    # ================= RUN PREDICTION =================
    try:
        prediction = model.predict(input_data)

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True
        )

        st.subheader("Prediction Result")
        st.write("Model output:", prediction[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            st.write("Model class probabilities:")

            for class_name, probability in zip(
                model.classes_, probabilities
            ):
                st.write(
                    f"{class_name}: {probability * 100:.2f}%"
                )

        st.markdown("</div>", unsafe_allow_html=True)

        st.warning(
            "This is an educational model output, not a medical "
            "diagnosis or a validated clinical risk estimate."
        )

    except Exception as e:
        st.error("Prediction failed. Check the model input format.")
        st.code(str(e))

# ================= FOOTER =================
st.divider()

st.caption(
    "CancerGuard | Educational Machine Learning Project"
)

