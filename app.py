import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# -----------------------------
# Load model, scaler, features
# -----------------------------
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("rf_pipeline.pkl")
    return pipeline["model"], pipeline["scaler"], pipeline["features"]

model, scaler, features = load_pipeline()

st.title("❤️ Heart Disease Prediction")
st.write(
    "Fill in the patient's details below. The model (Random Forest) will estimate "
    "the probability of heart disease based on the trained dataset."
)

st.divider()

# -----------------------------
# Input form
# -----------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - Typical angina",
                1: "1 - Atypical angina",
                2: "2 - Non-anginal pain",
                3: "3 - Asymptomatic",
            }[x],
        )
        resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
        cholesterol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        ecg = st.selectbox(
            "Resting ECG Result",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 - Normal",
                1: "1 - ST-T wave abnormality",
                2: "2 - Left ventricular hypertrophy",
            }[x],
        )

    with col2:
        max_heart_rate = st.slider("Max Heart Rate Achieved", 71, 202, 150)
        exercise_induced_chest_pain = st.selectbox(
            "Exercise-Induced Chest Pain",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        st_depression = st.slider("ST Depression (exercise vs rest)", 0.0, 6.2, 1.0, step=0.1)
        st_slope = st.selectbox(
            "ST Slope",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x],
        )
        stained_blood_vessels = st.selectbox("Number of Major Vessels Stained (0-4)", options=[0, 1, 2, 3, 4])
        blood_disorder = st.selectbox(
            "Blood Disorder (Thalassemia)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - Normal",
                1: "1 - Fixed defect",
                2: "2 - Reversible defect",
                3: "3 - Other",
            }[x],
        )

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_dict = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "ecg": ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_chest_pain": exercise_induced_chest_pain,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "stained_blood_vessels": stained_blood_vessels,
        "blood_disorder": blood_disorder,
    }

    # Ensure column order matches training features exactly
    input_df = pd.DataFrame([input_dict], columns=features)

    # The scaler was only fit on the continuous numeric columns during training —
    # categorical/binary columns were left unscaled. Replicate that here.
    numeric_cols = list(scaler.feature_names_in_)
    input_scaled_df = input_df.copy()
    input_scaled_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Reorder columns to match exactly what the model was trained on
    input_scaled_df = input_scaled_df[list(model.feature_names_in_)]

    prediction = model.predict(input_scaled_df)[0]
    probability = model.predict_proba(input_scaled_df)[0]

    prob_no_disease = probability[0]
    prob_disease = probability[1]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High likelihood of Heart Disease — probability: {prob_disease:.1%}")
    else:
        st.success(f"✅ Low likelihood of Heart Disease — probability: {prob_no_disease:.1%}")

    col_a, col_b = st.columns(2)
    col_a.metric("Probability: No Disease", f"{prob_no_disease:.1%}")
    col_b.metric("Probability: Disease", f"{prob_disease:.1%}")

    st.progress(float(prob_disease))

    st.caption(
        "This tool is for educational purposes only and is not a substitute for "
        "professional medical diagnosis."
    )