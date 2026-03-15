import numpy as np
import datetime as dt
import joblib
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
.shap-box {
    background: #111111; border: 1px solid #222222;
    border-left: 3px solid #7c5cbf;
    border-radius: 8px; padding: 1.2rem 1.6rem; margin-top: 1rem;
}
.shap-title {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: #7c5cbf; margin-bottom: 0.8rem;
}
.shap-insight {
    font-size: 0.83rem; color: #888; line-height: 1.6; margin-top: 0.8rem;
}
.shap-insight span { color: #ff6b35; font-weight: 600; }
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

# Human-readable labels for SHAP chart (same order as model input)
FEATURE_DISPLAY = {
    'Item_MRP':          'Item MRP',
    'Outlet_Identifier': 'Outlet ID',
    'Outlet_Size':       'Outlet Size',
    'Outlet_Type':       'Outlet Type',
    'Outlet_age':        'Outlet Age',
}
FEATURE_NAMES   = list(FEATURE_DISPLAY.keys())
FEATURE_LABELS  = list(FEATURE_DISPLAY.values())

MAE          = 714.42
CURRENT_YEAR = dt.datetime.today().year

# ── Load model + explainer ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('bigmart_model')

@st.cache_resource
def load_explainer():
    return joblib.load('bigmart_explainer')

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

        input_arr = np.array([[p1, p2, p3, p4, p5]])

        pred  = model.predict(input_arr)[0]
        lower = max(0, pred - MAE)
        upper = pred + MAE

        # ── Prediction result (unchanged from original) ───────────────────
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

        # ── SHAP Explanation ──────────────────────────────────────────────
        try:
            explainer  = load_explainer()
            shap_vals  = explainer.shap_values(input_arr)   # shape (1, 5)
            sv         = shap_vals[0]                        # 1-D array of 5 values

            # Build sorted DataFrame for the bar chart
            order      = np.argsort(np.abs(sv))              # ascending abs
            sorted_labels = [FEATURE_LABELS[i] for i in order]
            sorted_sv     = sv[order]

            # Raw input values mapped back to human-readable strings for annotation
            raw_display = {
                'Item MRP':    f"₹{p1:.2f}",
                'Outlet ID':   outlet_id,
                'Outlet Size': outlet_size,
                'Outlet Type': outlet_type,
                'Outlet Age':  f"{p5} yrs",
            }
            sorted_raw = [raw_display[FEATURE_LABELS[i]] for i in order]

            # ── Chart ─────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor('#111111')
            ax.set_facecolor('#111111')

            colors = ['#ff6b35' if v > 0 else '#4a7fcb' for v in sorted_sv]
            bars   = ax.barh(sorted_labels, sorted_sv, color=colors,
                             height=0.52, zorder=3)

            # Annotate each bar with raw value + SHAP delta
            for bar, sv_val, raw_val in zip(bars, sorted_sv, sorted_raw):
                x_end = bar.get_width()
                ha    = 'left' if x_end >= 0 else 'right'
                x_txt = x_end + (18 if x_end >= 0 else -18)
                ax.text(x_txt,
                        bar.get_y() + bar.get_height() / 2,
                        f"{raw_val}  →  {sv_val:+.0f}",
                        va='center', ha=ha,
                        fontsize=8, color='#aaaaaa',
                        fontfamily='monospace')

            ax.axvline(0, color='#333333', linewidth=1.0, zorder=2)
            ax.set_xlabel('SHAP value  (₹ impact on this prediction)',
                          fontsize=8.5, color='#666666', labelpad=8)

            ax.tick_params(axis='y', colors='#cccccc', labelsize=9)
            ax.tick_params(axis='x', colors='#555555', labelsize=8)
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            ax.spines['left'].set_color('#2a2a2a')
            ax.grid(axis='x', color='#1e1e1e', linestyle='--',
                    linewidth=0.6, zorder=0)

            # Legend
            pos_patch = mpatches.Patch(color='#ff6b35', label='Increases prediction')
            neg_patch = mpatches.Patch(color='#4a7fcb', label='Decreases prediction')
            ax.legend(handles=[pos_patch, neg_patch], loc='lower right',
                      fontsize=7.5, framealpha=0,
                      labelcolor='#888888')

            plt.tight_layout(pad=1.2)

            # ── Render inside styled box ──────────────────────────────────
            st.markdown('<div class="shap-box"><div class="shap-title">🔍 Why this prediction?</div></div>',
                        unsafe_allow_html=True)
            st.pyplot(fig, transparent=False)
            plt.close(fig)

            # ── Plain-English insight ─────────────────────────────────────
            top_idx       = int(np.argmax(np.abs(sv)))
            top_label     = FEATURE_LABELS[top_idx]
            top_val       = list(raw_display.values())[top_idx]
            top_sv        = sv[top_idx]
            direction_str = "pushed the prediction <span>up</span>" if top_sv > 0 \
                            else "pulled the prediction <span>down</span>"

            st.markdown(f"""
            <div class="shap-insight">
                The biggest driver was <span>{top_label}</span> ({top_val}), which
                {direction_str} by <span>₹{abs(top_sv):,.0f}</span>.
                Orange bars increase the forecast; blue bars reduce it.
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.markdown("""
            <div class="shap-box">
                <div class="shap-title">🔍 Why this prediction?</div>
                <div class="shap-insight">
                    SHAP explainer file <code>bigmart_explainer</code> not found.
                    Run the <em>Save SHAP Explainer</em> cell in the notebook first,
                    then redeploy.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as shap_err:
            st.markdown(f"""
            <div class="shap-box">
                <div class="shap-title">🔍 Why this prediction?</div>
                <div class="shap-insight">SHAP explanation unavailable: {shap_err}</div>
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
        st.markdown(f'<div class="error-box">Prediction failed: {e}</div>',
                    unsafe_allow_html=True)
