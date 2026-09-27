import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration (Full Desktop, Clean View without Sidebar)
st.set_page_config(
    page_title="ABC Ltd. | Dispatch Operations Console",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Blue-White Corporate Theme Styling
st.markdown("""
<style>
    /* Force light blue-white background and readable text */
    .stApp {
        background-color: #f0f4f9;
        color: #1e293b;
    }
    
    /* Top Header Section */
    .app-header {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
    }
    .app-header h1 {
        margin: 0;
        font-size: 28px;
        color: #ffffff;
        font-weight: 700;
    }
    .app-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #dbeafe;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.06);
        text-align: center;
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #1e3a8a;
    }
    .kpi-sub {
        font-size: 12px;
        font-weight: 600;
        color: #2563eb;
        margin-top: 4px;
    }

    /* Form Container */
    .panel-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* Shipment Status Card */
    .shipment-card {
        background: #ffffff;
        border: 2px solid #93c5fd;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
    }
    .badge-delayed {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #f87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-ontime {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #6ee7b7;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Saved Pipeline Artifacts
@st.cache_resource
def load_models():
    lin_model = joblib.load('linear_model.joblib')
    log_model = joblib.load('logistic_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return lin_model, log_model, scaler, encoders, feature_names

lin_model, log_model, scaler, encoders, feature_names = load_models()

# 4. Blue Header Banner
st.markdown("""
<div class="app-header">
    <h1>🚚 ABC Ltd. — Dispatch Operations Intelligence Console</h1>
    <p>Operational decision-support tool evaluating delivery durations and SLA delay risks in real time.</p>
</div>
""", unsafe_allow_html=True)

# 5. Top 3 KPI Cards (Active Hub Removed)
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Primary SLA Bar</div>
        <div class="kpi-value">120 Mins</div>
        <div class="kpi-sub">Standard Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Logistic Model Accuracy</div>
        <div class="kpi-value">69.3%</div>
        <div class="kpi-sub">Test Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">System Priority</div>
        <div class="kpi-value">Recall (Type 2)</div>
        <div class="kpi-sub">Prevent Missed Delays</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Two-Column Desktop Workstation
col_left, col_right = st.columns([1, 1], gap="large")

# LEFT COLUMN: Parameters Entry
with col_left:
    st.markdown("### 📋 Dispatch Parameters")
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    
    distance = st.slider("Delivery Route Distance (km)", min_value=0.5, max_value=80.0, value=12.5, step=0.5)
    
    sub1, sub2 = st.columns(2)
    with sub1:
        agent_rating = st.slider("Partner Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
    with sub2:
        agent_age = st.number_input("Partner Age", min_value=18, max_value=60, value=30, step=1)
        
    sub3, sub4 = st.columns(2)
    with sub3:
        traffic = st.selectbox("Traffic Level", options=list(encoders['Traffic'].classes_))
        weather = st.selectbox("Weather Condition", options=list(encoders['Weather'].classes_))
    with sub4:
        vehicle = st.selectbox("Vehicle Type", options=list(encoders['Vehicle'].classes_))
        area = st.selectbox("Urban Classification", options=list(encoders['Area'].classes_))
        
    st.markdown('</div>', unsafe_allow_html=True)
    run_btn = st.button("🚀 Assess Dispatch Risk", type="primary", use_container_width=True)

# RIGHT COLUMN: Clean Live Assessment
with col_right:
    st.markdown("### 📦 Live Shipment Assessment")

    if run_btn:
        # Preprocess inputs
        input_dict = {
            'Agent_Age': agent_age,
            'Agent_Rating': agent_rating,
            'Distance_km': distance,
            'Weather': encoders['Weather'].transform([weather])[0],
            'Traffic': encoders['Traffic'].transform([traffic])[0],
            'Vehicle': encoders['Vehicle'].transform([vehicle])[0],
            'Area': encoders['Area'].transform([area])[0]
        }
        
        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        # Run model inference
        pred_minutes = lin_model.predict(input_scaled)[0]
        delay_prob = log_model.predict_proba(input_scaled)[0][1]
        is_delayed = log_model.predict(input_scaled)[0]

        # Badge selection
        if is_delayed == 1:
            badge_html = '<span class="badge-delayed">⚠️ HIGH RISK OF DELAY</span>'
        else:
            badge_html = '<span class="badge-ontime">✅ ON-TIME DISPATCH</span>'

        order_id = np.random.randint(100000, 999999)

        # Visual Shipment Card (Zero markdown-indentation bugs)
        st.markdown(f"""<div class="shipment-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
    <div>
        <div style="color: #64748b; font-size: 12px; font-weight: 700; text-transform: uppercase;">Shipment Ticket</div>
        <h3 style="margin: 0; color: #1e3a8a; font-size: 20px;">Order #ABC-{order_id}</h3>
    </div>
    <div>{badge_html}</div>
</div>
<div style="text-align: center; color: #2563eb; font-weight: 700; font-size: 15px; margin: 18px 0; background: #eff6ff; padding: 10px; border-radius: 8px;">
    Warehouse &nbsp; ─── &nbsp; Dark Store Hub &nbsp; ─── &nbsp; 🚚 &nbsp; ─── &nbsp; Customer
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
    <div>
        <div style="color: #64748b; font-size: 12px; font-weight: 600;">ESTIMATED TIME (LINEAR)</div>
        <div style="color: #0f172a; font-size: 26px; font-weight: 800;">{pred_minutes:.0f} mins</div>
    </div>
    <div style="text-align: right;">
        <div style="color: #64748b; font-size: 12px; font-weight: 600;">DELAY PROBABILITY (LOGISTIC)</div>
        <div style="color: #0f172a; font-size: 26px; font-weight: 800;">{delay_prob * 100:.1f}%</div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Clean Action Guidance Alerts
        if is_delayed == 1:
            st.error("""
            **⚠️ Recommended Interventions (SLA Breach Risk):**
            * Assign an expedited partner or switch to a faster vehicle type.
            * Flag the dark store staging station for high-priority dispatch.
            * Send an automated preemptive notification to manage customer expectations.
            """)
        else:
            st.success("""
            **✅ Standard Dispatch Approved:**
            * Trip duration aligns within the 120-minute SLA window.
            * Dispatch order can proceed without supervisory adjustments.
            """)
    else:
        st.info("👈 Set the dispatch values on the left and click **'Assess Dispatch Risk'** to see the assessment.")
