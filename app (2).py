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

# ── Page config ───────────────────────────────────────────────────────────────
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

/* ── Result box ── */
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
.result-range { font-size: 0.85rem; color: #555; margin-top: 0.5rem; }
.result-range b { color: #4a7fcb; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-card {
    flex: 1; background: #141414; border: 1px solid #1f1f1f;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-card-label {
    font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px;
}
.metric-card-value {
    font-family: 'Syne', sans-serif; font-size: 1.3rem;
    font-weight: 700; color: #f0ede6; margin-top: 0.2rem;
}
.metric-card-value.ci { color: #4a7fcb; }
.metric-card-sub {
    font-size: 0.68rem; color: #3a3a3a;
    margin-top: 0.2rem; letter-spacing: 0.5px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase; margin: 1.8rem 0 0.3rem 0;
}
.section-header.orange { color: #ff6b35; }
.section-header.purple { color: #7c5cbf; }
.section-header.teal   { color: #2a9d8f; }
.section-subhead { font-size: 0.8rem; color: #555; margin-bottom: 0.8rem; }

/* ── Insight box ── */
.insight-box {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 1rem 1.4rem;
    font-size: 0.84rem; color: #888; line-height: 1.7; margin-top: 0.6rem;
}
.insight-box.purple { border-left: 3px solid #7c5cbf; }
.insight-box.teal   { border-left: 3px solid #2a9d8f; }
.insight-box b      { color: #ff6b35; font-weight: 600; }

/* ── Dashboard pills ── */
.dash-pills { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; }
.dash-pill {
    background: #1a1a1a; border: 1px solid #252525;
    border-radius: 20px; padding: 0.3rem 0.85rem;
    font-size: 0.75rem; color: #888; white-space: nowrap;
}
.dash-pill b { color: #ff6b35; }

/* ── Error boxes ── */
.error-box {
    background: #1a0a0a; border: 1px solid #3a1a1a;
    border-left: 3px solid #ff3333; border-radius: 8px;
    padding: 1rem 1.5rem; margin-top: 1rem; color: #ff6666; font-size: 0.85rem;
}
.soft-error {
    background: #0f0f1a; border: 1px solid #1e1e2e; border-left: 3px solid #555;
    border-radius: 8px; padding: 1rem 1.4rem;
    font-size: 0.83rem; color: #555; margin-top: 1rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Global feature importances (from xg_final.feature_importances_) ──────────
# Run: for f,v in zip(FEATURE_LABELS, xg_final.feature_importances_): print(f,v)
# and paste your real values here if they differ after retraining.
FEATURE_IMPORTANCES = {
    'Item MRP':    0.6192,
    'Outlet Type': 0.2087,
    'Outlet ID':   0.0831,
    'Outlet Age':  0.0521,
    'Outlet Size': 0.0369,
}

MAE          = 714.42          # fallback if quantile models unavailable
CURRENT_YEAR = dt.datetime.today().year

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('bigmart_model')

@st.cache_resource
def load_explainer():
    return joblib.load('bigmart_explainer')

@st.cache_resource
def load_quantile_models():
    q_low  = joblib.load('bigmart_model_q10')
    q_high = joblib.load('bigmart_model_q90')
    return q_low, q_high

# ── Chart helpers ─────────────────────────────────────────────────────────────
BG       = '#0d0d0d'
PANEL_BG = '#131313'
POS_COL  = '#ff6b35'
NEG_COL  = '#4a7fcb'
GRID_COL = '#191919'


def _base_ax(fig, ax):
    """Apply shared dark theme to any axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color=GRID_COL, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def render_importance_chart():
    items  = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1])   # asc
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]
    max_v  = max(vals)
    palette = [POS_COL if v == max_v else '#cc5529' if v >= max_v * 0.25
               else '#7a3218' for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.0))
    _base_ax(fig, ax)
    bars = ax.barh(labels, vals, color=palette, height=0.44, zorder=3, linewidth=0)

    fig.canvas.draw()
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for bar, val in zip(bars, vals):
        bw  = bar.get_width()
        pct = f'{val*100:.1f}%'
        if bw >= x_span * 0.12:
            ax.text(bar.get_x() + bw - x_span * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    pct, va='center', ha='right', fontsize=9,
                    color='#f0ede6', fontfamily='monospace',
                    fontweight='bold', zorder=5)
        else:
            ax.text(bar.get_x() + bw + x_span * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    pct, va='center', ha='left', fontsize=9,
                    color='#666', fontfamily='monospace', zorder=5)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.tick_params(axis='y', colors='#c8c4bc', labelsize=10,  length=0, pad=10)
    ax.set_xlabel('Relative contribution to model predictions (XGBoost feature importance)',
                  fontsize=7.8, color='#3a3a3a', labelpad=10)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.22)
    return fig


def render_shap_chart(sv, raw_display):
    order         = np.argsort(np.abs(sv))
    sorted_labels = [FEATURE_LABELS[i] for i in order]
    sorted_sv     = sv[order]
    sorted_raw    = [raw_display[FEATURE_LABELS[i]] for i in order]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    _base_ax(fig, ax)
    colors = [POS_COL if v > 0 else NEG_COL for v in sorted_sv]
    bars   = ax.barh(range(len(sorted_labels)), sorted_sv,
                     color=colors, height=0.46, zorder=3, linewidth=0)
    ax.axvline(0, color='#2e2e2e', linewidth=1.4, zorder=2)

    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=10, color='#c8c4bc')
    ax.tick_params(axis='y', length=0, pad=10)
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

    fig.canvas.draw()
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    pad    = x_span * 0.012

    for i, (bar, sv_val, raw_val) in enumerate(zip(bars, sorted_sv, sorted_raw)):
        bw    = bar.get_width()
        sign  = '+' if sv_val >= 0 else ''
        label = f"{raw_val}   {sign}₹{sv_val:,.0f}"
        if abs(bw) >= x_span * 0.22:
            x_pos = (bar.get_x() + bw - pad) if sv_val >= 0 \
                    else (bar.get_x() + bw + pad)
            ha, tc = ('right', '#f0ede6') if sv_val >= 0 else ('left', '#f0ede6')
        else:
            x_pos = (bar.get_x() + bw + pad) if sv_val >= 0 \
                    else (bar.get_x() + bw - pad)
            ha, tc = ('left', '#888') if sv_val >= 0 else ('right', '#888')
        ax.text(x_pos, i, label, va='center', ha=ha,
                fontsize=8, color=tc, fontfamily='monospace', zorder=5)

    ax.set_xlabel('SHAP value  —  how much each feature moved the sales forecast',
                  fontsize=7.8, color='#3a3a3a', labelpad=10)
    pos_p = mpatches.Patch(facecolor=POS_COL, label='Increases prediction', linewidth=0)
    neg_p = mpatches.Patch(facecolor=NEG_COL, label='Decreases prediction', linewidth=0)
    ax.legend(handles=[pos_p, neg_p], loc='lower right', fontsize=7.5,
              framealpha=0, labelcolor='#555', handlelength=1.0,
              handleheight=0.8, borderpad=0.3, labelspacing=0.3)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.20)
    return fig


def render_ci_chart(pred, lower, upper, is_quantile):
    """
    Horizontal gauge showing prediction ± confidence interval.
    Green zone = interval, orange tick = point estimate.
    """
    fig, ax = plt.subplots(figsize=(8, 1.6))
    _base_ax(fig, ax)

    margin  = (upper - lower) * 0.5
    x_lo    = max(0, lower - margin)
    x_hi    = upper + margin

    # Interval band
    ax.barh(0, upper - lower, left=lower, height=0.55,
            color='#4a7fcb', alpha=0.25, zorder=2, linewidth=0)
    ax.barh(0, upper - lower, left=lower, height=0.55,
            color='none', zorder=3, linewidth=1.0,
            edgecolor='#4a7fcb')

    # Bound markers
    for xv, label, align in [(lower, f'₹{lower:,.0f}', 'center'),
                               (upper, f'₹{upper:,.0f}', 'center')]:
        ax.axvline(xv, color='#4a7fcb', linewidth=1.0,
                   linestyle='--', alpha=0.6, zorder=3)
        ax.text(xv, 0.52, label, va='bottom', ha=align,
                fontsize=7.5, color='#4a7fcb', fontfamily='monospace')

    # Point estimate
    ax.axvline(pred, color=POS_COL, linewidth=2.5, zorder=5)
    ax.text(pred, -0.52, f'₹{pred:,.0f}', va='top', ha='center',
            fontsize=9, color=POS_COL, fontfamily='monospace', fontweight='bold')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.8, 0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.set_yticks([])
    ax.grid(axis='x')

    ci_label = '80% quantile interval' if is_quantile else '±MAE range'
    ax.set_title(ci_label, color='#444', fontsize=8,
                 loc='left', pad=6)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.72, bottom=0.28)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛒 Big Mart\nSales Predictor")
st.markdown("---")

# ── Feature Importance Dashboard ─────────────────────────────────────────────
st.markdown('<div class="section-header orange">📊 Feature Importance Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subhead">'
    'Global breakdown — how much each input contributes to the model\'s predictions '
    'across all outlets and items.'
    '</div>',
    unsafe_allow_html=True
)
imp_fig = render_importance_chart()
st.pyplot(imp_fig, use_container_width=True)
plt.close(imp_fig)

pills_html = ''.join(
    f'<div class="dash-pill"><b>{feat}</b>&nbsp;{imp*100:.1f}%</div>'
    for feat, imp in sorted(FEATURE_IMPORTANCES.items(),
                             key=lambda x: x[1], reverse=True)
)
st.markdown(f'<div class="dash-pills">{pills_html}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────────────────────
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
    outlet_type = st.selectbox("Outlet Type",       options=list(OUTLET_TYPE_MAP.keys()))

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Sales"):
    try:
        model = load_model()

        p1 = item_mrp
        p2 = OUTLET_ID_MAP[outlet_id]
        p3 = OUTLET_SIZE_MAP[outlet_size]
        p4 = OUTLET_TYPE_MAP[outlet_type]
        p5 = CURRENT_YEAR - outlet_year

        input_arr  = np.array([[p1, p2, p3, p4, p5]])
        pred       = model.predict(input_arr)[0]

        # ── Confidence interval (quantile or MAE fallback) ────────────────────
        is_quantile = False
        try:
            q_low, q_high = load_quantile_models()
            lower         = max(0.0, float(q_low.predict(input_arr)[0]))
            upper         = float(q_high.predict(input_arr)[0])
            is_quantile   = True
        except Exception:
            lower = max(0.0, pred - MAE)
            upper = pred + MAE

        ci_label  = "80% Confidence Interval" if is_quantile else "±MAE Range"
        ci_note   = "quantile regression" if is_quantile else "fixed ±MAE fallback"

        # ── Result box ────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Sales</div>
            <div class="result-value">&#8377;{pred:,.0f}</div>
            <div class="result-range">
                <b>{ci_label}</b> &nbsp;&middot;&nbsp;
                &#8377;{lower:,.0f} &mdash; &#8377;{upper:,.0f}
                <span style="color:#2a2a2a;font-size:0.75rem">
                &nbsp;({ci_note})
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-card-label">Lower Bound</div>
                <div class="metric-card-value ci">&#8377;{lower:,.0f}</div>
                <div class="metric-card-sub">10th percentile</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Outlet Age</div>
                <div class="metric-card-value">{p5} yrs</div>
                <div class="metric-card-sub">est. {outlet_year}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Upper Bound</div>
                <div class="metric-card-value ci">&#8377;{upper:,.0f}</div>
                <div class="metric-card-sub">90th percentile</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence Interval gauge chart ───────────────────────────────────
        st.markdown(
            '<div class="section-header teal" style="margin-top:1.4rem">'
            '📐 Prediction Interval</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="section-subhead">'
            f'The orange line is the point estimate. '
            f'The blue band is the {ci_label.lower()} — '
            f'the model expects actual sales to fall here ~80% of the time.'
            f'</div>',
            unsafe_allow_html=True
        )
        ci_fig = render_ci_chart(pred, lower, upper, is_quantile)
        st.pyplot(ci_fig, use_container_width=True)
        plt.close(ci_fig)

        if is_quantile:
            width = upper - lower
            st.markdown(f"""
            <div class="insight-box teal">
                The 80% prediction interval spans
                <b>₹{width:,.0f}</b> (₹{lower:,.0f} → ₹{upper:,.0f}).
                This is computed by two dedicated XGBoost quantile models
                (Q10 and Q90) trained on the same features — giving a
                data-driven range rather than a fixed offset.
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        try:
            explainer = load_explainer()
            sv        = explainer.shap_values(input_arr)[0]

            raw_display = {
                'Item MRP':    f'₹{p1:.2f}',
                'Outlet ID':   outlet_id,
                'Outlet Size': outlet_size,
                'Outlet Type': outlet_type,
                'Outlet Age':  f'{p5} yrs',
            }

            st.markdown(
                '<div class="section-header purple">🔍 Why this prediction?</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-subhead">'
                'Each bar shows how much that feature pushed the sales forecast '
                'up (orange) or down (blue) for this specific input.'
                '</div>',
                unsafe_allow_html=True
            )

            shap_fig = render_shap_chart(sv, raw_display)
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)

            # Ranked insight
            top_idx   = int(np.argmax(np.abs(sv)))
            top_label = FEATURE_LABELS[top_idx]
            top_raw   = list(raw_display.values())[top_idx]
            top_sv    = sv[top_idx]
            direction = "increased" if top_sv > 0 else "decreased"

            medals = ["🥇", "🥈", "🥉", "④", "⑤"]
            ranked_lines = ""
            for rank, idx in enumerate(np.argsort(np.abs(sv))[::-1]):
                lbl    = FEATURE_LABELS[idx]
                val    = list(raw_display.values())[idx]
                impact = sv[idx]
                arrow  = "▲" if impact > 0 else "▼"
                color  = POS_COL if impact > 0 else NEG_COL
                ranked_lines += (
                    f'<span style="color:#3a3a3a">{medals[rank]}</span> '
                    f'<b>{lbl}</b> '
                    f'<span style="color:#555">({val})</span> '
                    f'<span style="color:{color}">{arrow} ₹{abs(impact):,.0f}</span><br>'
                )

            st.markdown(f"""
            <div class="insight-box purple">
                <b>{top_label}</b> ({top_raw}) was the biggest driver —
                it <b>{direction}</b> the prediction by
                <b>₹{abs(top_sv):,.0f}</b>.<br><br>
                <span style="font-size:0.75rem;color:#3a3a3a;letter-spacing:1.5px;
                             text-transform:uppercase">Feature ranking by impact</span><br>
                <span style="font-size:0.85rem;line-height:2.1">{ranked_lines}</span>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.markdown("""
            <div class="soft-error">
                <b>🔍 Why this prediction?</b><br><br>
                SHAP explainer file <code>bigmart_explainer</code> not found.
                Run the <em>Save SHAP Explainer</em> cell in the notebook,
                commit the file to your repo, then redeploy.
            </div>
            """, unsafe_allow_html=True)

        except Exception as shap_err:
            st.markdown(f"""
            <div class="soft-error">
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
        st.markdown(
            f'<div class="error-box">Prediction failed: {e}</div>',
            unsafe_allow_html=True
        )
