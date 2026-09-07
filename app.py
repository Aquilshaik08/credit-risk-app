import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap

# Configure cross-device responsive UI layout
st.set_page_config(page_title="Intelligent Credit Risk Assessment System", layout="wide")

st.title("🛡️ Intelligent Credit Risk System")
st.write("Enter the applicant's financial metrics below to evaluate risk and compute AI explanations.")

# Load backend AI model assets safely
@st.cache_resource
def load_model():
    with open('credit_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- 1. USER INPUT DASHBOARD CONTAINER ---
st.subheader("📋 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
    existing_debt_payment = st.number_input("Total Monthly Debt ($)", min_value=0, value=1000)

with col2:
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000)
    employment_years = st.number_input("Years of Employment", min_value=0, value=3)
    prior_defaults = st.selectbox("Prior Defaults?", [0, 1])

# Feature engineering application parameters
dti_ratio = existing_debt_payment / (monthly_income + 1)

# Assemble input vector into structured DataFrame format
input_data = pd.DataFrame([{
    'credit_score': credit_score,
    'monthly_income': monthly_income,
    'existing_debt': existing_debt_payment,    
    'loan_amount': loan_amount,
    'employment_years': employment_years,
    'dti_ratio': dti_ratio,
    'prior_defaults': prior_defaults,
}])

# --- 2. EVALUATION & DECISION ENGINE ---
if st.button("Evaluate Application"):
    st.divider()
    st.subheader("📊 System Analysis & Verdict")
    
    # Tier 1: System Business Logic Rules
    if credit_score < 500:
        st.error("❌ Application Rejected")
        st.info("Reason: Credit Score falls below absolute minimum fallback threshold (500).")
    elif prior_defaults == 1 and dti_ratio > 0.45:
        st.error("❌ Application Rejected")
        st.info("Reason: High-risk indicators (Prior Defaults combined with DTI > 45%).")
        
    # Tier 2: Machine Learning Prediction Pipeline
    else:
        # Filter down to ONLY the 2 features your model was trained on
        model_features = ['credit_score', 'monthly_income']
        model_input = input_data[model_features]
      
        # Predict probability of default using the filtered data
        prob_default = model.predict_proba(model_input)[0][1]
      
        # Risk thresholds mapping logic
        if prob_default < 0.30:
            st.success("🎉 Application Approved")
            st.metric(label="Risk Assessment Score", value=f"{prob_default:.2%}")
            st.write("**Recommendation:** Low-risk profile. Proceed with standard onboarding.")
        elif prob_default < 0.65:
            st.warning("⚠️ Application Flagged for Manual Review")
            st.metric(label="Risk Assessment Score", value=f"{prob_default:.2%}")
            st.write("**Recommendation:** Medium-risk profile. Secondary underwriter review required.")
        else:
            st.error("❌ Application Denied")
            st.metric(label="Risk Assessment Score", value=f"{prob_default:.2%}")
            st.write("**Reason:** High default probability calculated by predictive model.")
          
        # --- 3. EXPLAINABLE AI INTERFACE (SHAP) ---
        st.divider()
        st.subheader("🔍 Explainable AI Diagnostics")
        st.write("Factors driving this risk rating (SHAP feature attribution analysis):")
      
        # Use the same filtered data here so SHAP doesn't crash
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(model_input)
      
        # Safe extraction for binary classification targets
        if isinstance(shap_values, list):
            # For older SHAP versions mapping multi-class/binary list outputs
            val_to_use = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            # For newer SHAP formats yielding (samples, features, classes)
            val_to_use = shap_values[0, :, 1]
        else:
            val_to_use = shap_values[0]
            
        # Structure SHAP factors dynamically into readable list
        feature_impact = pd.DataFrame({
            'Financial Indicator': model_features,
            'Impact Score': val_to_use
        }).sort_values(by='Impact Score', ascending=False)
      
        st.dataframe(feature_impact)