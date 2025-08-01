import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── 0) Page config & global CSS ────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.markdown(
    """
    <!-- seven decorative divs -->
    <div id="bg-circle"></div>
    <div id="bg-square"></div>
    <div id="bg-triangle"></div>
    <div id="bg-dots"></div>
    <div id="bg-stripe"></div>
    <div id="bg-waves"></div>
    <div id="bg-gradient-orbs"></div>

    <style>
      /* 1) App background colour only */
      [data-testid="stAppViewContainer"] {
        background-color: #f7fafc !important;
      }

      /* 2) Full-viewport circle */
      #bg-circle {
        position: fixed;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background-color: rgba(100, 150, 200, 0.3);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
      }

      /* 3) Full-viewport rotated square */
      #bg-square {
        position: fixed;
        bottom: -50px;
        left: -50px;
        width: 200px;
        height: 200px;
        background-color: rgba(255, 180, 100, 0.3);
        transform: rotate(45deg);
        pointer-events: none;
        z-index: 0;
      }

      /* 4) Large soft triangle */
      #bg-triangle {
        position: fixed;
        top: 20%;
        left: -150px;
        width: 0;
        height: 0;
        border-left: 250px solid rgba(200, 100, 255, 0.25);
        border-bottom: 400px solid transparent;
        pointer-events: none;
        z-index: 0;
      }

      /* 5) Subtle dot-pattern overlay */
      #bg-dots {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: radial-gradient(rgba(0,0,0,0.05) 1px, transparent 1px);
        background-size: 25px 25px;
        pointer-events: none;
        z-index: 0;
      }

      /* 6) Diagonal stripe accent */
      #bg-stripe {
        position: fixed;
        top: 40%;
        left: -30%;
        width: 200%;
        height: 100px;
        background: linear-gradient(
          45deg,
          transparent 30%,
          rgba(255,255,255,0.4) 40%,
          rgba(255,255,255,0.4) 60%,
          transparent 70%
        );
        pointer-events: none;
        z-index: 0;
      }

      /* 7) Wave pattern */
      #bg-waves {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 150px;
        background: linear-gradient(
          to right,
          rgba(72, 187, 120, 0.1) 0%,
          rgba(56, 178, 172, 0.1) 50%,
          rgba(129, 140, 248, 0.1) 100%
        );
        clip-path: polygon(0 60%, 15% 40%, 35% 50%, 50% 30%, 65% 45%, 80% 25%, 100% 35%, 100% 100%, 0 100%);
        pointer-events: none;
        z-index: 0;
      }

      /* 8) Floating gradient orbs */
      #bg-gradient-orbs {
        position: fixed;
        top: 10%;
        left: 20%;
        width: 180px;
        height: 180px;
        background: radial-gradient(
          circle at 30% 30%,
          rgba(236, 72, 153, 0.3) 0%,
          rgba(147, 51, 234, 0.2) 40%,
          transparent 70%
        );
        border-radius: 50%;
        filter: blur(2px);
        animation: float 6s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
      }

      /* Animation for floating orbs */
      @keyframes float {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
      }

      /* Scroll animations */
      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(30px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes slideInLeft {
        from {
          opacity: 0;
          transform: translateX(-50px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      @keyframes slideInRight {
        from {
          opacity: 0;
          transform: translateX(50px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      @keyframes scaleIn {
        from {
          opacity: 0;
          transform: scale(0.8);
        }
        to {
          opacity: 1;
          transform: scale(1);
        }
      }

      /* Apply animations with initial hidden state */
      .header-bar {
        background: linear-gradient(90deg,#4B7BEC,#3A5AAC);
        padding: 2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin: 2rem auto 1rem;
        max-width: 900px;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease-out forwards;
      }
      .header-bar h1 { margin: 0; font-size: 2.5rem; }

      form[data-testid="stForm"] {
        opacity: 0;
        transform: translateX(-50px);
        animation: slideInLeft 0.8s ease-out 0.2s forwards;
      }

      .card {
        opacity: 0;
        transform: scale(0.8);
        animation: scaleIn 0.6s ease-out 0.4s forwards;
      }

      .metric-card {
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.5s ease-out forwards;
      }

      .metric-card:nth-child(1) { animation-delay: 0.1s; }
      .metric-card:nth-child(2) { animation-delay: 0.2s; }
      .metric-card:nth-child(3) { animation-delay: 0.3s; }

      /* Smooth scroll behavior */
      html {
        scroll-behavior: smooth;
      }

      /* Hover animations */
      .metric-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }

      .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
      }

      form[data-testid="stForm"]:hover {
        transform: translateY(-2px);
        transition: transform 0.3s ease;
      }

      /* Background elements subtle animation */
      #bg-circle {
        animation: float 8s ease-in-out infinite;
      }

      #bg-square {
        animation: float 10s ease-in-out infinite reverse;
      }

      #bg-triangle {
        animation: float 12s ease-in-out infinite;
      }

      /* 7) Ensure Streamlit UI is on top */
      [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
      }

      /* 8) Centre the results card */
      .card {
        display: block !important;
        margin: 2rem auto !important;
        max-width: 800px !important;
        width: 90% !important;
        text-align: center !important;
      }
      .card h3 {
        margin-bottom: 1.5rem;
      }

      /* 9) Metric cards styling */
      .metric-card {
        background: transparent !important;
        text-align: center !important;
        padding: 1rem !important;
      }
      .metric-card h4 { 
        margin-bottom: 0.5rem; 
        color: #555; 
      }
      .metric-card h2 { 
        margin: 0; 
        font-size: 1.75rem; 
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── 1) Header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-bar animated-header"><h1>🩺 Diabetes Risk Predictor</h1></div>',
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; max-width:900px; margin:auto; font-size:1.1rem;'>"
    "Estimate a patient’s diabetes risk in seconds—no uploads needed. "
    "Fill in the fields below and hit Submit."
    "</p>",
    unsafe_allow_html=True
)

# Add gap between description and patient input box
st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)

# ─── 2) Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="diabetes_model_calibrated.pkl"):
    return joblib.load(path)
model = load_model()

# ─── 3) Patient Information Form ─────────────────────────────────────────────
with st.form("patient_form"):
    st.markdown("<h3 style='margin-bottom:0.5rem;'>Patient Information</h3>", unsafe_allow_html=True)
    st.markdown("""
    <style>
      .input-label {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        display: block;
      }
    </style>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("<span class='input-label'>Age (years) 📆</span>", unsafe_allow_html=True)
        age = st.number_input("Age (years) 📆", 0, 120, 50, label_visibility="collapsed")
        st.markdown("<span class='input-label'>Gender ♀️♂️</span>", unsafe_allow_html=True)
        gender = st.selectbox("Gender ♀️♂️", ["Male", "Female", "Other"], label_visibility="collapsed")
        st.markdown("<span class='input-label'>History of hypertension?</span>", unsafe_allow_html=True)
        hypertension = st.selectbox("History of hypertension?", ["No", "Yes"], label_visibility="collapsed")
        st.markdown("<span class='input-label'>History of heart disease?</span>", unsafe_allow_html=True)
        heart_disease = st.selectbox("History of heart disease?", ["No", "Yes"], label_visibility="collapsed")
        st.markdown("<span class='input-label'>Smoking history 🚬</span>", unsafe_allow_html=True)
        smoking_history = st.selectbox("Smoking history 🚬", ["never","former","current","not current"], label_visibility="collapsed")
    with cols[1]:
        st.markdown("<span class='input-label'>Height (cm) 📏</span>", unsafe_allow_html=True)
        height_cm = st.number_input("Height (cm) 📏", 50.0, 250.0, 170.0, step=0.5, label_visibility="collapsed")
        st.markdown("<span class='input-label'>Weight (kg) ⚖️</span>", unsafe_allow_html=True)
        weight_kg = st.number_input("Weight (kg) ⚖️", 20.0, 200.0, 70.0, step=0.5, label_visibility="collapsed")
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m**2)
        st.markdown(f"**Calculated BMI:** {bmi:.1f}", unsafe_allow_html=True)
        st.markdown("<span class='input-label'>A1c (Glycated Hemoglobin) % 🧂</span>", unsafe_allow_html=True)
        hba1c = st.number_input(
            "A1c (Glycated Hemoglobin) % ",
            3.0, 15.0, 6.0, step=0.1,
            help="Avg. blood sugar over the past 2–3 months; normal <5.7%.",
            label_visibility="collapsed"
        )
        st.caption("Reflects your average blood glucose over the last 2–3 months.")
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<span class='input-label'>Blood glucose (mg/dL) 🩸</span>", unsafe_allow_html=True)
        glucose = st.number_input("Blood glucose (mg/dL) 🩸", 50.0, 300.0, 100.0, label_visibility="collapsed")
    submitted = st.form_submit_button("🔬 Submit")

# ─── 4) Prediction Results ───────────────────────────────────────────────────
if submitted:
    df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "hypertension": int(hypertension=="Yes"),
        "heart_disease": int(heart_disease=="Yes"),
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    def engineer(d):
        d = d.copy()
        d["log_glucose"]  = np.log1p(d["blood_glucose_level"])
        d["log_bmi"]      = np.log1p(d["bmi"])
        d["age_squared"]  = d["age"]**2
        d["bmi_squared"]  = d["bmi"]**2
        d["age_bmi_interaction"]     = d["age"]*d["bmi"]
        d["bmi_glucose_interaction"] = d["bmi"]*d["blood_glucose_level"]
        d["glucose_hba1c_ratio"]     = d["blood_glucose_level"]/d["HbA1c_level"]
        d["high_glucose_flag"] = (d["blood_glucose_level"]>125).astype(int)
        d["high_hba1c_flag"]   = (d["HbA1c_level"]>5.7).astype(int)
        d["obese_flag"]        = (d["bmi"]>=30).astype(int)
        Y,M = 30,60
        d["age_bin_old"]    = (d["age"]>M).astype(int)
        d["age_bin_middle"] = ((d["age"]>Y)&(d["age"]<=M)).astype(int)
        def map_risk(r):
            if r.obese_flag and r.high_glucose_flag: return 2
            if r.obese_flag or r.high_glucose_flag:  return 1
            return 0
        d["risk_group"] = d.apply(map_risk, axis=1)
        return d

    df_eng = engineer(df)
    prob   = model.predict_proba(df_eng)[0,1]
    pred   = int(prob>=0.5)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Prediction Results</h3>', unsafe_allow_html=True)

    cols = st.columns(3, gap="large")
    with cols[0]:
        st.markdown(f'<div class="metric-card"><h4>Probability</h4><h2>{prob:.1%}</h2></div>', unsafe_allow_html=True)
    with cols[1]:
        lbl = "Diabetic" if pred else "Non-diabetic"
        st.markdown(f'<div class="metric-card"><h4>Classification</h4><h2>{lbl}</h2></div>', unsafe_allow_html=True)
    with cols[2]:
        lvl = "High" if prob>=0.75 else "Moderate" if prob>=0.40 else "Low"
        st.markdown(f'<div class="metric-card"><h4>Risk Level</h4><h2>{lvl}</h2></div>', unsafe_allow_html=True)

    a,m,b = st.columns([1,2,1], gap="small")
    with m:
        if prob>=0.75:
            st.error("🚨 High risk! Please visit your doctor right away.")
        elif prob>=0.40:
            st.warning("⚠️ Moderate risk. Consider scheduling a check-up.")
        else:
            st.success("✅ Low risk. Keep up the healthy habits!")
    st.markdown('</div>', unsafe_allow_html=True)
