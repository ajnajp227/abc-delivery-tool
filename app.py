import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Desktop widescreen configuration
st.set_page_config(
    page_title="ABC Ltd. | Dispatch Operations Console",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Desktop Dashboard Styling
st.markdown("""
<style>
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .shipment-panel {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .badge-delayed {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #f87171;
    }
    .badge-ontime {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #4ade80;
    }
    .transit-bar {
        font-size: 20px;
        color: #ea580c;
        text-align: center;
        letter-spacing: 4px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Saved ML Pipeline Artifacts
@st.cache_resource
def load_models():
    lin_model = joblib.load('linear_model.joblib')
    log_model = joblib.load('logistic_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return lin_model, log_model, scaler, encoders, feature_names

lin_model, log_model, scaler, encoders, feature_names = load_models()

# 4. Top Header & Operational KPIs
st.title("🚚 ABC Ltd. — Dispatch Operations Intelligence Console")
st.markdown("Real-time decision support tool evaluating **Estimated Delivery Duration (Linear Regression)** and **SLA Delay Risk (Logistic Regression)**.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Primary SLA Bar", "120 Mins", "Delivery Benchmark")
with kpi2:
    st.metric("Logistic Model Accuracy", "69.3%", "Trained Baseline")
with kpi3:
    st.metric("Active Hub", "Zone 4 Hub", "North Region")
with kpi4:
    st.metric("System Priority", "Recall (Type 2)", "Prevent Missed Delays")

st.markdown("---")

# 5. Two-Column Desktop Layout
col_left, col_right = st.columns([1, 1], gap="large")

# LEFT COLUMN: Managerial Input Controls
with col_left:
    st.subheader("📋 Dispatch Parameter Entry")
    
    st.markdown("##### 📍 Route & Partner Metrics")
    distance = st.slider("Delivery Route Distance (km)", min_value=0.5, max_value=80.0, value=12.5, step=0.5)
    
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        agent_rating = st.slider("Agent Performance Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
    with c_sub2:
        agent_age = st.number_input("Delivery Partner Age", min_value=18, max_value=60, value=30, step=1)
        
    st.markdown("##### 🌤️ Operational & Ambient Conditions")
    c_sub3, c_sub4 = st.columns(2)
    with c_sub3:
        traffic = st.selectbox("Current Traffic Density", options=list(encoders['Traffic'].classes_))
        weather = st.selectbox("Current Weather Condition", options=list(encoders['Weather'].classes_))
    with c_sub4:
        vehicle = st.selectbox("Assigned Vehicle Type", options=list(encoders['Vehicle'].classes_))
        area = st.selectbox("Zone Urban Classification", options=list(encoders['Area'].classes_))

    run_btn = st.button("🚀 Assess Dispatch Risk", type="primary", use_container_width=True)

# RIGHT COLUMN: Live Shipment Status Panel (Inspired by reference dashboard)
with col_right:
    st.subheader("📦 Live Dispatch Assessment & SLA Diagnosis")

    if run_btn:
        # Vectorize & scale inputs
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

        # Run models
        pred_minutes = lin_model.predict(input_scaled)[0]
        delay_prob = log_model.predict_proba(input_scaled)[0][1]
        is_delayed = log_model.predict(input_scaled)[0]

        # Dynamic Status Badge
        badge_html = '<span class="badge-delayed">⚠️ HIGH RISK OF DELAY</span>' if is_delayed == 1 else '<span class="badge-ontime">✅ ON-TIME DISPATCH</span>'

        # Desktop Shipment Overview Card
        st.markdown(f"""
        <div class="shipment-panel">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin: 0; color: #64748b; font-size: 13px; font-weight: 600;">ACTIVE SHIPMENT TICKET</p>
                    <h3 style="margin: 0; color: #0f172a;">Order #ABC-{np.random.randint(100000, 999999)}</h3>
                </div>
                <div>{badge_html}</div>
            </div>
            
            <div class="transit-bar">
                Warehouse ─── Hub ─── 🚚 ─────── Customer
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 15px;">
                <div>
                    <p style="margin: 0; color: #64748b; font-size: 13px;">Estimated Duration (Linear)</p>
                    <h2 style="margin: 0; color: #0f172a;">{pred_minutes:.0f} mins</h2>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; color: #64748b; font-size: 13px;">Delay Probability (>120 SLA)</p>
                    <h2 style="margin: 0; color: #0f172a;">{delay_prob * 100:.1f}%</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Actionable Guidelines for Managers
        if is_delayed == 1:
            st.error("""
            **⚠️ Recommended Managerial Interventions (High SLA Risk):**
            - **Fleet Reassignment:** Assign a motorcycle or faster route partner if available.
            - **Customer Communication:** Send an automated advance delivery alert to set expectations and lower churn risk.
            - **Hub Priority:** Prioritize this package at the dark-store staging counter.
            """)
        else:
            st.success("""
            **✅ Standard Dispatch Approved:**
            - Route duration is well within the 120-minute SLA envelope.
            - Dispatch can proceed without manual supervisory intervention.
            """)
    else:
        st.info("👈 Set the operational parameters on the left and click **'Assess Dispatch Risk'** to generate predictions.")

# 6. Desktop Sidebar: User Study Context & Study Prompts
with st.sidebar:
    st.header("🏢 ABC Ltd. Pilot Overview")
    st.markdown("""
    This interface is deployed for a **qualitative managerial user study**:
    - **Step 1:** Select route parameters.
    - **Step 2:** Observe how Linear & Logistic models evaluate transit time and delay risk.
    - **Step 3:** Review recommended operational interventions.
    """)
    st.markdown("---")
    st.markdown("**Model Specs:**")
    st.caption("- Linear Regression ($R^2 \approx 0.28$, $\text{MAE} \approx 33.7$ min)")
    st.caption("- Logistic Regression (Accuracy $\approx 69.3\%$, Target $>120$ min)")
