import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="ABC Ltd. | Dispatch Operations Console",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Complete CSS Fix: Explicit Dark Text Colors for All Widgets
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f1f5f9;
    }

    /* Force all widget labels, titles, and text to dark high-contrast color */
    label, p, span, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 600;
    }

    /* Slider values & tick marks */
    .stSlider div {
        color: #0f172a !important;
    }

    /* Input boxes, Selectboxes & Number inputs: White background with dark text */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div, 
    div[data-baseweb="base-input"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }

    /* Top Blue Header */
    .app-header {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .app-header h1 {
        margin: 0;
        font-size: 26px;
        color: #ffffff !important;
        font-weight: 700;
    }
    .app-header p {
        margin: 4px 0 0 0;
        font-size: 14px;
        color: #dbeafe !important;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.08);
        text-align: center;
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #1e3a8a !important;
    }
    .kpi-sub {
        font-size: 12px;
        font-weight: 600;
        color: #2563eb !important;
        margin-top: 4px;
    }

    /* Form Container */
    .panel-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.03);
    }

    /* Shipment Result Card */
    .shipment-card {
        background: #ffffff;
        border: 2px solid #93c5fd;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.1);
    }
    .badge-delayed {
        background-color: #fee2e2;
        color: #b91c1c !important;
        border: 1px solid #f87171;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-ontime {
        background-color: #ecfdf5;
        color: #047857 !important;
        border: 1px solid #6ee7b7;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models
@st.cache_resource
def load_models():
    lin_model = joblib.load('linear_model.joblib')
    log_model = joblib.load('logistic_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return lin_model, log_model, scaler, encoders, feature_names

lin_model, log_model, scaler, encoders, feature_names = load_models()

# 4. Header Banner
st.markdown("""
<div class="app-header">
    <h1>🚚 ABC Ltd. — Dispatch Operations Intelligence Console</h1>
    <p>Operational decision-support tool evaluating delivery durations and SLA delay risks in real time.</p>
</div>
""", unsafe_allow_html=True)

# 5. Top 3 KPI Cards
# 5. Top KPI Cards
kpi1, kpi2 = st.columns(2)

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
    
st.write("")

# 6. Two-Column Layout
col_left, col_right = st.columns([1, 1], gap="large")

# LEFT COLUMN: Parameters Entry
with col_left:
    st.markdown("### 📋 Dispatch Parameters")
    with st.container():
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
    
    st.write("")
    run_btn = st.button("🚀 Assess Dispatch Risk", type="primary", use_container_width=True)

# RIGHT COLUMN: Results
with col_right:
    st.markdown("### 📦 Live Shipment Assessment")

    if run_btn:
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

        pred_minutes = lin_model.predict(input_scaled)[0]
        delay_prob = log_model.predict_proba(input_scaled)[0][1]
        is_delayed = log_model.predict(input_scaled)[0]

        badge_html = '<span class="badge-delayed">⚠️ HIGH RISK OF DELAY</span>' if is_delayed == 1 else '<span class="badge-ontime">✅ ON-TIME DISPATCH</span>'
        order_id = np.random.randint(100000, 999999)

        st.markdown(f"""<div class="shipment-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
    <div>
        <div style="color: #64748b !important; font-size: 12px; font-weight: 700; text-transform: uppercase;">Shipment Ticket</div>
        <h3 style="margin: 0; color: #1e3a8a !important; font-size: 20px;">Order #ABC-{order_id}</h3>
    </div>
    <div>{badge_html}</div>
</div>
<div style="text-align: center; color: #2563eb !important; font-weight: 700; font-size: 14px; margin: 16px 0; background: #eff6ff; padding: 10px; border-radius: 8px;">
    Warehouse &nbsp; ─── &nbsp; Dark Store Hub &nbsp; ─── &nbsp; 🚚 &nbsp; ─── &nbsp; Customer
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
    <div>
        <div style="color: #64748b !important; font-size: 12px; font-weight: 600;">ESTIMATED TIME (LINEAR)</div>
        <div style="color: #0f172a !important; font-size: 26px; font-weight: 800;">{pred_minutes:.0f} mins</div>
    </div>
    <div style="text-align: right;">
        <div style="color: #64748b !important; font-size: 12px; font-weight: 600;">DELAY PROBABILITY (LOGISTIC)</div>
        <div style="color: #0f172a !important; font-size: 26px; font-weight: 800;">{delay_prob * 100:.1f}%</div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

        st.write("")

        if is_delayed == 1:
            st.error("""
            **⚠️ Recommended Interventions (SLA Breach Risk):**
            * Assign an expedited partner or switch to a faster vehicle type.
            * Flag dark store staging for priority dispatch.
            * Send proactive delay notification to the customer.
            """)
        else:
            st.success("""
            **✅ Standard Dispatch Approved:**
            * Trip duration aligns within the 120-minute SLA window.
            * Dispatch order can proceed without supervisory adjustments.
            """)
    else:
        st.info("👈 Set the dispatch values on the left and click **'Assess Dispatch Risk'** to see the assessment.")
