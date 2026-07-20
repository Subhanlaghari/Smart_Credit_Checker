import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Credit Scoring System",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# FONT SIZE FIX
# -----------------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 14px !important;
    }
    h1 { font-size: 24px !important; }
    h2 { font-size: 20px !important; }
    h3 { font-size: 17px !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(BASE_DIR, "credit_scoring_model.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
except Exception as e:
    model = None
    st.error(f"Error Loading Files: {e}")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💳 AI Credit Scoring")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("CodeAlpha Machine Learning Project")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.title("🏦 AI Credit Scoring System")

    st.write(
        """
Predict whether a customer has **Good Credit** or **Bad Credit**
using Machine Learning.
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "1000")

    with col2:
        st.metric("Accuracy", "78%")

    with col3:
        st.metric("Algorithm", "Logistic Regression")

    st.markdown("---")

    st.subheader("✨ Features")

    st.write("✅ Machine Learning Prediction")
    st.write("✅ Credit Risk Analysis")
    st.write("✅ Loan Approval Recommendation")
    st.write("✅ Confidence Score")
    st.write("✅ Professional Dashboard")

    st.markdown("---")

    st.info(
        "This application predicts customer credit risk using the German Credit Dataset."
    )
    # -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Customer Credit Prediction")

    st.write("Please enter the customer information below.")

    col1, col2 = st.columns(2)

    with col1:

        credit_amount = st.number_input(
            "💰 Credit Amount",
            min_value=100,
            max_value=20000,
            value=5000
        )

        duration = st.slider(
            "📅 Loan Duration (Months)",
            min_value=4,
            max_value=72,
            value=24
        )

        age = st.slider(
            "🎂 Age",
            min_value=18,
            max_value=75,
            value=30
        )

        employment = st.selectbox(
            "💼 Employment Since",
            [
                "Unemployed",
                "<1 Year",
                "1-4 Years",
                "4-7 Years",
                "7+ Years"
            ]
        )

        housing = st.selectbox(
            "🏠 Housing",
            [
                "Own",
                "Rent",
                "Free"
            ]
        )

    with col2:

        savings = st.selectbox(
            "💵 Savings Account",
            [
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )

        credit_history = st.selectbox(
            "📄 Credit History",
            [
                "Excellent",
                "Good",
                "Average",
                "Poor",
                "Critical"
            ]
        )

        purpose = st.selectbox(
            "🎯 Loan Purpose",
            [
                "Car",
                "Furniture",
                "Education",
                "Business",
                "Repairs",
                "Vacation",
                "Others"
            ]
        )

        existing_credits = st.slider(
            "🏦 Existing Credits",
            min_value=1,
            max_value=4,
            value=1
        )

        installment_rate = st.slider(
            "💳 Installment Rate",
            min_value=1,
            max_value=4,
            value=2
        )

    st.markdown("---")

    predict = st.button(
        "🚀 Predict Credit Score",
        use_container_width=True
    )

    if predict:

        employment_map = {
            "Unemployed": "A71",
            "<1 Year": "A72",
            "1-4 Years": "A73",
            "4-7 Years": "A74",
            "7+ Years": "A75"
        }

        housing_map = {
            "Own": "A152",
            "Rent": "A151",
            "Free": "A153"
        }

        savings_map = {
            "Low": "A61",
            "Medium": "A62",
            "High": "A63",
            "Very High": "A64"
        }

        credit_history_map = {
            "Excellent": "A30",
            "Good": "A31",
            "Average": "A32",
            "Poor": "A33",
            "Critical": "A34"
        }

        purpose_map = {
            "Car": "A40",
            "Furniture": "A42",
            "Education": "A46",
            "Business": "A49",
            "Repairs": "A45",
            "Vacation": "A410",
            "Others": "A410"
        }

        raw_input = {
            "Status_checking_account": "A11",
            "Duration": duration,
            "Credit_history": credit_history_map[credit_history],
            "Purpose": purpose_map[purpose],
            "Credit_amount": credit_amount,
            "Savings_account": savings_map[savings],
            "Employment_since": employment_map[employment],
            "Installment_rate": installment_rate,
            "Personal_status_sex": "A93",
            "Other_debtors": "A101",
            "Residence_since": 2,
            "Property": "A123",
            "Age": age,
            "Other_installment_plans": "A143",
            "Housing": housing_map[housing],
            "Existing_credits": existing_credits,
            "Job": "A173",
            "Num_dependents": 1,
            "Telephone": "A192",
            "Foreign_worker": "A201"
        }

        if model is not None:

            row = []

            for col in feature_columns:
                value = raw_input[col]
                if col in encoders:
                    value = encoders[col].transform([value])[0]
                row.append(value)

            row_scaled = scaler.transform([row])
            prediction = model.predict(row_scaled)[0]
            probability = model.predict_proba(row_scaled)[0]

            st.markdown("---")

            if prediction == 1:
                confidence = probability[0] * 100
                st.success("## ✅ GOOD CREDIT")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "Good Credit")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.metric("Risk Level", "LOW")
                    st.metric("Loan Status", "APPROVED ✅")

                st.subheader("Confidence Level")
                st.progress(float(probability[0]))
                st.info("This customer has a very good credit profile. Loan approval is recommended.")

            else:
                confidence = probability[1] * 100
                st.error("## ❌ BAD CREDIT")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "Bad Credit")
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col2:
                    st.metric("Risk Level", "HIGH")
                    st.metric("Loan Status", "REJECTED ❌")

                st.subheader("Risk Level")
                st.progress(float(probability[1]))
                st.warning("High credit risk detected. Loan approval is not recommended.")

        else:
            st.error("❌ Model files could not be loaded. Please check all .pkl files.")

# -----------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------
elif page == "📊 Model Performance":

    st.title("📊 Model Performance Dashboard")

    st.write(
        "Performance of the trained Logistic Regression model on the German Credit Dataset."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "78.00%")
        st.metric("Precision", "80.89%")

    with col2:
        st.metric("Recall", "90.07%")
        st.metric("F1 Score", "85.23%")

    with col3:
        st.metric("ROC AUC", "81.70%")
        st.metric("Dataset Size", "1000")

    st.markdown("---")

    st.subheader("📈 Model Summary")

    st.success("""
✔ Logistic Regression Model

✔ German Credit Dataset

✔ 1000 Customer Records

✔ 20 Input Features

✔ Binary Classification

✔ Suitable for Credit Risk Prediction
""")

# -----------------------------
# ABOUT PAGE
# -----------------------------
else:

    st.title("ℹ️ About Project")

    st.markdown("""
# 💳 AI Credit Scoring System

This project predicts whether a customer is likely to have **Good Credit**
or **Bad Credit** using Machine Learning.

The system is built using **Python**, **Streamlit**, **Scikit-Learn**
and the **German Credit Dataset**.
""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Pandas")
    st.write("• Scikit-Learn")
    st.write("• Joblib")

    st.markdown("---")

    st.subheader("🤖 Machine Learning Algorithm")

    st.info("Logistic Regression")

    st.markdown("---")

    st.subheader("📂 Dataset")

    st.info("German Credit Dataset (1000 Records)")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.success("Subhan Laghari")

    st.caption("CodeAlpha Machine Learning Internship Project")

    st.markdown("---")

    st.caption("© 2026 AI Credit Scoring System | All Rights Reserved")
