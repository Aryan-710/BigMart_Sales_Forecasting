import numpy as np
import datetime as dt
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big Mart Sales Predictor",
    page_icon="🛒",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d0d; color: #f0ede6; }

h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 2.6rem !important;
    color: #f0ede6 !important; letter-spacing: -1px; line-height: 1.1;
}
h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    color: #ff6b35 !important; margin-bottom: 1rem !important;
}
label, .stSelectbox label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #999 !important; letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
input[type="number"], .stSelectbox > div > div {
    background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    border-radius: 6px !important; color: #f0ede6 !important;
}
.stButton > button {
    background: #ff6b35 !important; color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.9rem !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 6px !important; padding: 0.65rem 2rem !important;
    width: 100% !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #ff8c5a !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,53,0.3) !important;
}
hr { border-color: #1f1f1f !important; margin: 1.5rem 0 !important; }

.result-box {
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    border: 1px solid #2a2a2a; border-left: 3px solid #ff6b35;
    border-radius: 8px; padding: 1.5rem 2rem; margin-top: 1.5rem;
}
.result-label {
    font-size: 0.75rem; font-weight: 500; color: #666;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.3rem;
}
.result-value {
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 800; color: #ff6b35; letter-spacing: -1px;
}
.result-range { font-size: 0.85rem; color: #666; margin-top: 0.4rem; }

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-card {
    flex: 1; background: #141414; border: 1px solid #1f1f1f;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-card-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px; }
.metric-card-value {
    font-family: 'Syne', sans-serif; font-size: 1.3rem;
    font-weight: 700; color: #f0ede6; margin-top: 0.2rem;
}
.error-box {
    background: #1a0a0a; border: 1px solid #3a1a1a;
    border-left: 3px solid #ff3333; border-radius: 8px;
    padding: 1rem 1.5rem; margin-top: 1rem; color: #ff6666; font-size: 0.85rem;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Encoding maps (match OrdinalEncoder output from notebook) ────────────────
OUTLET_ID_MAP = {
    'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3,
    'OUT019': 4, 'OUT027': 5, 'OUT035': 6, 'OUT045': 7,
    'OUT046': 8, 'OUT049': 9
}
OUTLET_SIZE_MAP = {'High': 0, 'Medium': 1, 'Small': 2}
OUTLET_TYPE_MAP = {
    'Grocery Store': 0, 'Supermarket Type1': 1,
    'Supermarket Type2': 2, 'Supermarket Type3': 3
}

MAE          = 714.42
CURRENT_YEAR = dt.datetime.today().year

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('bigmart_model')

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🛒 Big Mart\nSales Predictor")
st.markdown("---")
st.markdown("### Inputs")

col1, col2 = st.columns(2)

with col1:
    item_mrp = st.number_input(
        "Item MRP (₹)", min_value=10.0, max_value=300.0,
        value=141.62, step=0.01, help="Maximum Retail Price of the item"
    )
    outlet_size = st.selectbox("Outlet Size", options=list(OUTLET_SIZE_MAP.keys()))
    outlet_year = st.number_input(
        "Outlet Establishment Year",
        min_value=1980, max_value=CURRENT_YEAR, value=1999, step=1
    )

with col2:
    outlet_id   = st.selectbox("Outlet Identifier", options=list(OUTLET_ID_MAP.keys()))
    outlet_type = st.selectbox("Outlet Type", options=list(OUTLET_TYPE_MAP.keys()))

st.markdown("---")

if st.button("Predict Sales"):
    try:
        model = load_model()

        p1 = item_mrp
        p2 = OUTLET_ID_MAP[outlet_id]
        p3 = OUTLET_SIZE_MAP[outlet_size]
        p4 = OUTLET_TYPE_MAP[outlet_type]
        p5 = CURRENT_YEAR - outlet_year

        pred  = model.predict(np.array([[p1, p2, p3, p4, p5]]))[0]
        lower = max(0, pred - MAE)
        upper = pred + MAE

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Sales</div>
            <div class="result-value">&#8377;{pred:,.0f}</div>
            <div class="result-range">Estimated range &nbsp;&middot;&nbsp; &#8377;{lower:,.0f} &mdash; &#8377;{upper:,.0f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-card-label">Lower Bound</div>
                <div class="metric-card-value">&#8377;{lower:,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Outlet Age</div>
                <div class="metric-card-value">{p5} yrs</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Upper Bound</div>
                <div class="metric-card-value">&#8377;{upper:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.markdown("""
        <div class="error-box">
            Model file <code>bigmart_model</code> not found.
            Make sure it is in the same directory as this app.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="error-box">Prediction failed: {e}</div>', unsafe_allow_html=True)
