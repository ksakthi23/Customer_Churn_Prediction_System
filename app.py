

"""
Customer Churn Prediction - Streamlit Frontend (designed UI)
Run:  streamlit run app.py
Needs: pip install streamlit pandas scikit-learn joblib
"""
import os
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATHS = [
    "models/churn_model.pkl",
    "models/churn_model.joblib",
    "outputs/churn_model.joblib",
    "churn_model.pkl",
    "churn_model.joblib",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 20px; padding: 2.2rem 2.5rem; color: #fff;
    box-shadow: 0 12px 30px rgba(79,70,229,.30); margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #fff; }
.hero p { margin: .5rem 0 0 0; font-size: 1.05rem; opacity: .92; color: #fff; }

.section-title {
    font-size: 1.05rem; font-weight: 600; margin: .2rem 0 .8rem 0;
    padding-left: .7rem; border-left: 4px solid #7c3aed;
}

.kpi-row { display: flex; gap: 1rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 150px; padding: 1rem 1.2rem; border-radius: 14px;
    background: rgba(124,58,237,.08); border: 1px solid rgba(124,58,237,.25);
}
.kpi .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }
.kpi .value { font-size: 1.45rem; font-weight: 700; margin-top: .15rem; }

.result-card {
    border-radius: 20px; padding: 1.5rem 1.5rem 1.2rem 1.5rem; text-align: center;
    border: 2px solid var(--c); background: var(--bg);
}
.result-card .badge {
    display: inline-block; padding: .35rem 1rem; border-radius: 999px;
    font-weight: 600; font-size: .9rem; color: #fff; background: var(--c);
}
.result-card h2 { margin: .8rem 0 0 0; font-size: 1.35rem; font-weight: 700; }
.result-card .sub { opacity: .75; font-size: .9rem; margin-top: .2rem; }

.tip {
    padding: .75rem 1rem; margin-bottom: .6rem; border-radius: 12px;
    background: rgba(124,58,237,.07); border-left: 4px solid #7c3aed; font-size: .93rem;
}
.tip.good { background: rgba(16,185,129,.09); border-left-color: #10b981; }
.tip.bad  { background: rgba(239,68,68,.08);  border-left-color: #ef4444; }

div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; border: none; border-radius: 12px; padding: .8rem 1rem;
    font-weight: 600; font-size: 1.05rem; color: #fff;
    background: linear-gradient(90deg, #4f46e5, #db2777);
    box-shadow: 0 6px 16px rgba(79,70,229,.35); transition: transform .15s ease;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px); color: #fff; border: none;
}
.footer { text-align: center; opacity: .6; font-size: .85rem; margin-top: 2.5rem; }
</style>
"""


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Find and load the trained model from any of the usual locations."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None


def risk_level(prob):
    """Return (label, colour, background, headline) for a churn probability."""
    if prob >= 0.50:
        return "HIGH RISK", "#ef4444", "rgba(239,68,68,.08)", "Customer is likely to CHURN"
    if prob >= 0.35:
        return "MEDIUM RISK", "#f59e0b", "rgba(245,158,11,.09)", "Customer may leave soon"
    return "LOW RISK", "#10b981", "rgba(16,185,129,.09)", "Customer is likely to STAY"


def gauge_svg(prob, color):
    """Semi-circle gauge drawn with pure SVG (no extra libraries needed)."""
    length = 251.33  # length of the half circle with radius 80
    offset = length * (1 - prob)
    return (
        '<svg viewBox="0 0 200 125" style="width:100%;max-width:320px;">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(128,128,128,.25)" '
        'stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" '
        f'stroke-width="16" stroke-linecap="round" stroke-dasharray="{length}" '
        f'stroke-dashoffset="{offset:.2f}"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="30" font-weight="700" '
        f'fill="{color}">{prob * 100:.1f}%</text>'
        '<text x="100" y="112" text-anchor="middle" font-size="9" fill="gray">'
        'CHURN PROBABILITY</text>'
        '</svg>'
    )


def analyse(c):
    """Rule-based explanation: risk factors and retention actions for one customer."""
    risks, good = [], []

    if c["Contract"] == "Month-to-month":
        risks.append("Month-to-month contract: customers can leave at any time. "
                     "Offer a discount for a 1-year or 2-year plan.")
    else:
        good.append(f"{c['Contract']} contract keeps the customer committed.")

    if c["tenure"] < 12:
        risks.append("New customer (under 12 months): run an onboarding call and a welcome offer.")
    elif c["tenure"] >= 36:
        good.append("Long-term customer (3+ years) with strong loyalty.")

    if c["InternetService"] == "Fiber optic":
        risks.append("Fiber optic users churn more often: check speed, price and service quality.")

    if c["InternetService"] != "No":
        if c["OnlineSecurity"] == "No" or c["TechSupport"] == "No":
            risks.append("No Online Security / Tech Support: offer a free trial of these add-ons.")
        else:
            good.append("Uses support and security add-ons, which improves retention.")

    if c["PaymentMethod"] == "Electronic check":
        risks.append("Pays by electronic check: encourage automatic payment with a small incentive.")

    if c["MonthlyCharges"] > 80:
        risks.append("High monthly bill (above 80): review the plan or offer a loyalty discount.")

    if c["SeniorCitizen"] == 1:
        risks.append("Senior citizen: provide a dedicated, easy support channel.")

    if c["Partner"] == "Yes" or c["Dependents"] == "Yes":
        good.append("Has a partner or dependents, so the household is more stable.")

    return risks, good


def build_customer(**v):
    """Make a one-row DataFrame in the exact format the model was trained on."""
    row = dict(v)
    # keep the data consistent, like the original dataset
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    if row["InternetService"] == "No":
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
            row[col] = "No internet service"
    return pd.DataFrame([row])


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    model, model_path = load_model()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("## 📊 ChurnGuard")
        st.caption("AI-powered customer retention assistant")
        st.divider()
        st.markdown("### How it works")
        st.markdown(
            "1. Enter customer details\n"
            "2. The ML model analyses the profile\n"
            "3. Get churn probability and retention tips"
        )
        st.divider()
        st.markdown("### Model status")
        if model is not None:
            st.success("Model loaded")
            st.caption(f"File: `{model_path}`")
        else:
            st.error("Model file not found")
        st.divider()
        st.caption("Built with Python, scikit-learn and Streamlit")

    # ---------------- Hero ----------------
    st.markdown(
        '<div class="hero"><h1>📊 Customer Churn Prediction</h1>'
        '<p>Identify customers who are about to leave, and learn how to keep them.</p></div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "Could not find the trained model. Run `python churn_project.py` first, then place "
            "the saved file in a `models/` folder (e.g. `models/churn_model.pkl`)."
        )
        st.stop()

    tab_predict, tab_about = st.tabs(["🔮  Predict", "ℹ️  About the project"])

    # ================= PREDICT TAB =================
    with tab_predict:
        with st.form("customer_form"):
            st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            gender = c1.selectbox("Gender", ["Male", "Female"])
            senior = c2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = c3.selectbox("Has Partner?", ["Yes", "No"])
            dependents = c4.selectbox("Has Dependents?", ["Yes", "No"])

            st.markdown('<div class="section-title">📡 Services</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            phone = s1.selectbox("Phone Service", ["Yes", "No"])
            lines = s2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            tenure = s4.slider("Tenure (months)", 0, 72, 12)

            a1, a2, a3 = st.columns(3)
            security = a1.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = a2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device = a3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            b1, b2, b3 = st.columns(3)
            support = b1.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = b2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = b3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            st.markdown('<div class="section-title">💳 Contract and Billing</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            contract = d1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = d2.selectbox("Paperless Billing", ["Yes", "No"])
            payment = d3.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            e1, e2 = st.columns(2)
            monthly = e1.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
            total = e2.number_input("Total Charges", min_value=0.0, value=840.0, step=10.0)

            submitted = st.form_submit_button("🔍  Predict Customer Churn")

        if submitted:
            customer = build_customer(
                gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
                tenure=tenure, PhoneService=phone, MultipleLines=lines, InternetService=internet,
                OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=device,
                TechSupport=support, StreamingTV=tv, StreamingMovies=movies, Contract=contract,
                PaperlessBilling=paperless, PaymentMethod=payment,
                MonthlyCharges=monthly, TotalCharges=total,
            )
            prob = float(model.predict_proba(customer)[0][1])
            label, color, bg, headline = risk_level(prob)
            row = customer.iloc[0].to_dict()
            risks, good = analyse(row)

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

            left, right = st.columns([1, 1.3], gap="large")
            with left:
                st.markdown(
                    f'<div class="result-card" style="--c:{color};--bg:{bg};">'
                    f'<span class="badge">{label}</span>'
                    f'{gauge_svg(prob, color)}'
                    f'<h2>{headline}</h2>'
                    f'<div class="sub">Stay probability: {(1 - prob) * 100:.1f}%</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    '<div class="kpi-row">'
                    f'<div class="kpi"><div class="label">Tenure</div><div class="value">{tenure} mo</div></div>'
                    f'<div class="kpi"><div class="label">Monthly bill</div><div class="value">{monthly:,.0f}</div></div>'
                    f'<div class="kpi"><div class="label">Contract</div><div class="value">{contract.split("-")[0]}</div></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**⚠️ Risk factors and recommended actions**")
                if risks:
                    for r in risks:
                        st.markdown(f'<div class="tip bad">{r}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="tip good">No major risk factors found.</div>',
                                unsafe_allow_html=True)
                if good:
                    st.markdown("**✅ Positive signals**")
                    for g in good:
                        st.markdown(f'<div class="tip good">{g}</div>', unsafe_allow_html=True)

            with st.expander("View the data sent to the model"):
                st.dataframe(customer, use_container_width=True)

    # ================= ABOUT TAB =================
    with tab_about:
        st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
        st.write(
            "Customer churn means a customer stops using a company's service. This app uses a "
            "machine learning model trained on the Telco Customer Churn dataset to estimate how "
            "likely a customer is to leave, so the business can act before it happens."
        )
        st.markdown(
            '<div class="kpi-row">'
            '<div class="kpi"><div class="label">Dataset</div><div class="value">Telco Churn</div></div>'
            '<div class="kpi"><div class="label">Customers</div><div class="value">7,043</div></div>'
            '<div class="kpi"><div class="label">Features</div><div class="value">19</div></div>'
            '<div class="kpi"><div class="label">Task</div><div class="value">Classification</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Project workflow**")
        st.markdown(
            "1. Data cleaning and preprocessing\n"
            "2. Exploratory data analysis\n"
            "3. Training Logistic Regression, Random Forest and Gradient Boosting\n"
            "4. Comparing models (Accuracy, Precision, Recall, F1, ROC-AUC)\n"
            "5. Saving the best model and deploying it with Streamlit"
        )
        st.info("The retention tips are rule-based suggestions drawn from common churn patterns. "
                "The probability itself comes from the trained model.")

    st.markdown('<div class="footer">Machine Learning Project | Customer Churn Prediction</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()