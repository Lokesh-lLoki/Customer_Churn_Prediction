import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "test_metrics.json"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.write("Enter customer service and billing details to estimate churn risk.")
st.caption("Educational demonstration using IBM's sample Telco Customer Churn dataset. This is not a guarantee of customer behavior.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run Customer_Churn_Prediction.ipynb through the save-model section first.")
    st.stop()

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

model = load_pipeline(str(MODEL_PATH))

with st.form("customer_details"):
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

    with c2:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    with c3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])

    st.subheader("Prediction result")
    if prediction == 1:
        st.error("The model estimates that this customer is at risk of churn.")
    else:
        st.success("The model estimates that this customer is less likely to churn.")

    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("Use this result as a decision-support signal, not as a definitive conclusion about an individual customer.")

if METRICS_PATH.exists():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    with st.expander("View saved test-set evaluation metrics"):
        st.json(metrics)
