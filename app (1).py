import numpy as np
import datetime as dt
import joblib
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

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
.shap-header {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: #7c5cbf; margin: 1.6rem 0 0.3rem 0;
}
.shap-subhead {
    font-size: 0.8rem; color: #555; margin-bottom: 0.8rem;
}
.shap-insight {
    background: #111; border: 1px solid #1e1e1e; border-left: 3px solid #7c5cbf;
    border-radius: 8px; padding: 1rem 1.4rem;
    font-size: 0.84rem; color: #888; line-height: 1.7; margin-top: 0.6rem;
}
.shap-insight b { color: #ff6b35; font-weight: 600; }
.shap-error {
    background: #0f0f1a; border: 1px solid #1e1e2e; border-left: 3px solid #7c5cbf;
    border-radius: 8px; padding: 1rem 1.4rem;
    font-size: 0.83rem; color: #666; margin-top: 1rem;
}
.dash-header {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: #ff6b35; margin: 0 0 0.3rem 0;
}
.dash-subhead {
    font-size: 0.8rem; color: #555; margin-bottom: 0.8rem;
}
.dash-insight {
    display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem;
}
.dash-pill {
    background: #1a1a1a; border: 1px solid #252525;
    border-radius: 20px; padding: 0.3rem 0.85rem;
    font-size: 0.75rem; color: #888; white-space: nowrap;
}
.dash-pill b { color: #ff6b35; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Encoding maps ─────────────────────────────────────────────────────────────
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

FEATURE_LABELS = ['Item MRP', 'Outlet ID', 'Outlet Size', 'Outlet Type', 'Outlet Age']

# Global feature importances from XGBRFRegressor (xg_final.feature_importances_)
# These are fixed values from your trained model — update if you retrain.
FEATURE_IMPORTANCES = {
    'Item MRP':    0.6192,
    'Outlet Type': 0.2087,
    'Outlet ID':   0.0831,
    'Outlet Age':  0.0521,
    'Outlet Size': 0.0369,
}

MAE          = 714.42
CURRENT_YEAR = dt.datetime.today().year

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('bigmart_model')

@st.cache_resource
def load_explainer():
    return joblib.load('bigmart_explainer')

# ── SHAP chart ────────────────────────────────────────────────────────────────
def render_shap_chart(sv, raw_display):
    """
    sv          : 1-D numpy array, shape (5,)
    raw_display : dict mapping each FEATURE_LABEL → human-readable string
    """
    order         = np.argsort(np.abs(sv))           # ascending abs → smallest at top
    sorted_labels = [FEATURE_LABELS[i] for i in order]
    sorted_sv     = sv[order]
    sorted_raw    = [raw_display[FEATURE_LABELS[i]] for i in order]

    BG       = '#0d0d0d'
    PANEL_BG = '#131313'
    POS_COL  = '#ff6b35'
    NEG_COL  = '#4a7fcb'

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Bars ──────────────────────────────────────────────────────────────────
    bar_h  = 0.46
    colors = [POS_COL if v > 0 else NEG_COL for v in sorted_sv]
    bars   = ax.barh(
        range(len(sorted_labels)), sorted_sv,
        color=colors, height=bar_h, zorder=3, linewidth=0
    )

    # ── Zero line + grid ──────────────────────────────────────────────────────
    ax.axvline(0, color='#2e2e2e', linewidth=1.4, zorder=2)
    ax.grid(axis='x', color='#191919', linestyle='-', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── Y-axis: feature name labels ───────────────────────────────────────────
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=10, color='#c8c4bc')
    ax.tick_params(axis='y', length=0, pad=10)

    # ── X-axis ticks ──────────────────────────────────────────────────────────
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

    # ── Inline annotations (inside each bar) ──────────────────────────────────
    # Measure axis span AFTER drawing so xlim is auto-set by matplotlib
    fig.canvas.draw()                          # force layout so xlim is final
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    pad    = x_span * 0.012                    # 1.2% of total span as inner pad

    for i, (bar, sv_val, raw_val) in enumerate(
            zip(bars, sorted_sv, sorted_raw)):
        bar_w   = bar.get_width()
        sign    = '+' if sv_val >= 0 else ''
        label   = f"{raw_val}   {sign}₹{sv_val:,.0f}"
        bar_abs = abs(bar_w)

        # Only annotate inside bar if it is wide enough to hold the text
        min_width_fraction = 0.22  # bar must be ≥22% of axis span to annotate inside
        if bar_abs >= x_span * min_width_fraction:
            if sv_val >= 0:
                x_pos, ha = bar.get_x() + bar_w - pad, 'right'
            else:
                x_pos, ha = bar.get_x() + bar_w + pad, 'left'
            txt_color = '#f0ede6'
        else:
            # Annotate outside the bar for narrow bars
            if sv_val >= 0:
                x_pos, ha = bar.get_x() + bar_w + pad, 'left'
            else:
                x_pos, ha = bar.get_x() + bar_w - pad, 'right'
            txt_color = '#888888'

        ax.text(x_pos, i, label, va='center', ha=ha,
                fontsize=8, color=txt_color,
                fontfamily='monospace', zorder=5)

    # ── X label ───────────────────────────────────────────────────────────────
    ax.set_xlabel(
        'SHAP value  —  how much each feature moved the sales forecast',
        fontsize=7.8, color='#3a3a3a', labelpad=10
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    pos_p = mpatches.Patch(facecolor=POS_COL, label='Increases prediction', linewidth=0)
    neg_p = mpatches.Patch(facecolor=NEG_COL, label='Decreases prediction', linewidth=0)
    ax.legend(handles=[pos_p, neg_p], loc='lower right',
              fontsize=7.5, framealpha=0, labelcolor='#555555',
              handlelength=1.0, handleheight=0.8,
              borderpad=0.3, labelspacing=0.3)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.20)
    return fig


# ── Feature Importance Dashboard chart ───────────────────────────────────────
def render_importance_chart():
    """
    Horizontal bar chart of global XGBRFRegressor feature importances.
    Sorted descending (most important at top). Values are fixed from training.
    """
    # Sort descending so most important sits at the bottom of a horizontal chart
    # (matplotlib barh renders bottom-to-top, so we reverse for top-to-bottom visual)
    items  = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1], reverse=False)
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]

    BG       = '#0d0d0d'
    PANEL_BG = '#131313'

    # Colour gradient: most important = brightest orange, least = muted
    max_v   = max(vals)
    palette = ['#ff6b35' if v == max_v
               else '#cc5529' if v >= max_v * 0.25
               else '#7a3218'
               for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bars = ax.barh(labels, vals, color=palette, height=0.44, zorder=3, linewidth=0)

    # Percentage labels inside each bar
    fig.canvas.draw()
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min

    for bar, val in zip(bars, vals):
        bar_w = bar.get_width()
        pct   = f'{val * 100:.1f}%'
        if bar_w >= x_span * 0.12:
            # label inside bar, right-aligned
            ax.text(bar.get_x() + bar_w - x_span * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    pct, va='center', ha='right',
                    fontsize=9, color='#f0ede6',
                    fontfamily='monospace', fontweight='bold', zorder=5)
        else:
            # label outside bar for very short bars
            ax.text(bar.get_x() + bar_w + x_span * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    pct, va='center', ha='left',
                    fontsize=9, color='#666666',
                    fontfamily='monospace', zorder=5)

    # X-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.tick_params(axis='y', colors='#c8c4bc', labelsize=10,  length=0, pad=10)
    ax.grid(axis='x', color='#191919', linestyle='-', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel(
        'Relative contribution to model predictions  (XGBoost feature importance)',
        fontsize=7.8, color='#3a3a3a', labelpad=10
    )

    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.22)
    return fig


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

# ── Feature Importance Dashboard (always visible) ─────────────────────────────
st.markdown('<div class="dash-header">📊 Feature Importance Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="dash-subhead">'
    'How much each input contributes to the model\'s predictions globally, '
    'across all outlets and items.'
    '</div>',
    unsafe_allow_html=True
)

imp_fig = render_importance_chart()
st.pyplot(imp_fig, use_container_width=True)
plt.close(imp_fig)

# Summary pills
pills_html = ""
for feat, imp in sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1], reverse=True):
    pills_html += f'<div class="dash-pill"><b>{feat}</b> &nbsp;{imp*100:.1f}%</div>'
st.markdown(f'<div class="dash-insight">{pills_html}</div>', unsafe_allow_html=True)

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
        pred      = model.predict(input_arr)[0]
        lower     = max(0, pred - MAE)
        upper     = pred + MAE

        # ── Prediction result ─────────────────────────────────────────────────
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

        # ── SHAP section ──────────────────────────────────────────────────────
        try:
            explainer = load_explainer()
            sv        = explainer.shap_values(input_arr)[0]   # shape (5,)

            raw_display = {
                'Item MRP':    f'₹{p1:.2f}',
                'Outlet ID':   outlet_id,
                'Outlet Size': outlet_size,
                'Outlet Type': outlet_type,
                'Outlet Age':  f'{p5} yrs',
            }

            st.markdown('<div class="shap-header">🔍 Why this prediction?</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="shap-subhead">'
                'Each bar shows how much that feature pushed the sales forecast up (orange) or down (blue).'
                '</div>',
                unsafe_allow_html=True
            )

            fig = render_shap_chart(sv, raw_display)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Plain-English ranked insight ──────────────────────────────────
            top_idx   = int(np.argmax(np.abs(sv)))
            top_label = FEATURE_LABELS[top_idx]
            top_raw   = list(raw_display.values())[top_idx]
            top_sv    = sv[top_idx]
            direction = "increased" if top_sv > 0 else "decreased"

            rank_order = np.argsort(np.abs(sv))[::-1]
            medals     = ["🥇", "🥈", "🥉", "④", "⑤"]
            ranked_lines = ""
            for rank, idx in enumerate(rank_order):
                lbl    = FEATURE_LABELS[idx]
                val    = list(raw_display.values())[idx]
                impact = sv[idx]
                arrow  = "▲" if impact > 0 else "▼"
                color  = "#ff6b35" if impact > 0 else "#4a7fcb"
                ranked_lines += (
                    f'<span style="color:#3a3a3a">{medals[rank]}</span> '
                    f'<b>{lbl}</b> '
                    f'<span style="color:#555">({val})</span> '
                    f'<span style="color:{color}">{arrow} ₹{abs(impact):,.0f}</span><br>'
                )

            st.markdown(f"""
            <div class="shap-insight">
                <b>{top_label}</b> ({top_raw}) was the biggest driver —
                it <b>{direction}</b> the prediction by <b>₹{abs(top_sv):,.0f}</b>.<br><br>
                <span style="font-size:0.75rem;color:#3a3a3a;letter-spacing:1.5px;
                             text-transform:uppercase">Feature ranking by impact</span><br>
                <span style="font-size:0.85rem;line-height:2.1">{ranked_lines}</span>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.markdown("""
            <div class="shap-error">
                <b style="color:#7c5cbf">🔍 Why this prediction?</b><br><br>
                SHAP explainer file <code>bigmart_explainer</code> not found.
                Run the <em>Save SHAP Explainer</em> cell in the notebook,
                commit the file to your repo, then redeploy.
            </div>
            """, unsafe_allow_html=True)

        except Exception as shap_err:
            st.markdown(f"""
            <div class="shap-error">
                <b style="color:#7c5cbf">🔍 Why this prediction?</b><br><br>
                SHAP explanation unavailable: {shap_err}
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
