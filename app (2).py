import numpy as np
import datetime as dt
import joblib
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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
.metric-card-value.ci  { color: #4a7fcb; }
.metric-card-value.grn { color: #2a9d8f; }
.metric-card-sub { font-size: 0.68rem; color: #3a3a3a; margin-top: 0.2rem; letter-spacing: 0.5px; }

.section-header {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase; margin: 1.8rem 0 0.3rem 0;
}
.section-header.orange { color: #ff6b35; }
.section-header.purple { color: #7c5cbf; }
.section-header.teal   { color: #2a9d8f; }
.section-header.blue   { color: #4a7fcb; }
.section-subhead { font-size: 0.8rem; color: #555; margin-bottom: 0.8rem; }

.insight-box {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 1rem 1.4rem;
    font-size: 0.84rem; color: #888; line-height: 1.7; margin-top: 0.6rem;
}
.insight-box.purple { border-left: 3px solid #7c5cbf; }
.insight-box.teal   { border-left: 3px solid #2a9d8f; }
.insight-box.blue   { border-left: 3px solid #4a7fcb; }
.insight-box b      { color: #ff6b35; font-weight: 600; }

.perf-stat-row { display: flex; gap: 1rem; margin: 0.8rem 0; }
.perf-stat {
    flex: 1; background: #111; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 0.8rem 1rem; text-align: center;
}
.perf-stat-label { font-size: 0.68rem; color: #444; text-transform: uppercase; letter-spacing: 1.5px; }
.perf-stat-value {
    font-family: 'Syne', sans-serif; font-size: 1.15rem;
    font-weight: 700; color: #f0ede6; margin-top: 0.2rem;
}

.dash-pills { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; }
.dash-pill {
    background: #1a1a1a; border: 1px solid #252525;
    border-radius: 20px; padding: 0.3rem 0.85rem;
    font-size: 0.75rem; color: #888; white-space: nowrap;
}
.dash-pill b { color: #ff6b35; }

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

FEATURE_IMPORTANCES = {
    'Item MRP':    0.6192,
    'Outlet Type': 0.2087,
    'Outlet ID':   0.0831,
    'Outlet Age':  0.0521,
    'Outlet Size': 0.0369,
}

# ── Precomputed model performance stats (real values from notebook) ───────────
PERF = {
    'mae':      714.55,
    'rmse':     1023.81,
    'r2':       0.6143,
    'coverage': 76.7,
    'ci_width': 2271.2,
}

# ── Precomputed test-set curves (80 samples, from notebook evaluation) ────────
# np.int64 scalars are cast to plain float via np.asarray in each chart function
_act = [3912,4264,2653,2637,1348,1380,736,1145,459,3199,1896,4027,1342,198,
        769,2759,455,1659,2622,1471,623,118,2878,4075,3845,5896,2875,4414,
        173,216,1257,3300,1256,4579,2969,4165,4225,995,1708,1779,1494,2553,
        1390,1143,131,2650,1063,3135,575,518,94,1575,5360,3795,405,2069,
        1929,258,2200,1198,2314,2024,2252,270,1091,481,989,564,820,2405,
        2631,5593,3561,326,2085,1001,374,1929,1659,4994]
_prd = [2538,2049,2678,4080,866,2908,680,1723,534,2659,3975,2359,2871,266,
        1127,2900,263,3039,1770,2536,878,256,4064,2856,3491,4017,2859,2555,
        616,232,2545,2634,853,3408,2547,2937,3638,3064,951,1637,1445,3064,
        1315,1455,277,4073,1445,2542,732,650,189,2049,6306,3064,755,2911,
        1455,156,2609,2538,1458,2049,2042,753,2042,3390,1035,740,705,2912,
        1632,2472,1820,472,2898,1538,535,1447,4032,4049]
_low = [1140,871,1510,1640,317,1546,402,949,340,1633,1910,1142,1703,67,
        420,1131,126,1494,687,909,456,98,2254,1432,1156,2105,1715,1254,
        314,105,1153,1463,476,1880,1316,1621,1292,1529,432,656,809,1354,
        534,540,127,2304,728,950,419,238,104,925,3857,1497,424,1495,
        521,79,1209,1063,633,869,1161,334,1033,1817,605,333,328,1409,
        714,1047,877,234,1489,635,218,830,2140,1804]
_hig = [4006,3172,4175,5051,1374,4460,1268,2805,925,3787,5042,3528,4659,383,
        1339,4370,513,4567,2761,3819,1660,490,7296,4216,5127,6837,4705,4025,
        841,620,3981,4164,1361,5940,3508,4904,5365,4514,1244,2499,2096,4436,
        2110,2143,560,5630,2077,3774,1138,1069,339,3097,9295,4753,1168,4466,
        2196,198,4158,3877,2323,3484,3223,1292,3103,5683,1536,1126,950,4695,
        2544,3733,3290,787,4721,2405,1018,2323,6050,6362]

PERF_CURVES = {
    'act': _act,
    'prd': _prd,
    'low': _low,
    'hig': _hig,
}

MAE          = PERF['mae']
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
    return joblib.load('bigmart_model_q10'), joblib.load('bigmart_model_q90')

# ── Shared theme ──────────────────────────────────────────────────────────────
BG       = '#0d0d0d'
PANEL_BG = '#131313'
POS_COL  = '#ff6b35'
NEG_COL  = '#4a7fcb'
GRID_COL = '#1a1a1a'


def _theme(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color=GRID_COL, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _rupee_fmt(ax, axis='both'):
    fmt = FuncFormatter(lambda x, _: f'₹{x/1000:.0f}k' if abs(x) >= 1000
                        else f'₹{x:.0f}')
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Feature Importance (always visible)
# ══════════════════════════════════════════════════════════════════════════════
def render_importance_chart():
    items   = sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1])
    labels  = [k for k, _ in items]
    vals    = [v for _, v in items]
    max_v   = max(vals)
    palette = [POS_COL if v == max_v else '#cc5529' if v >= max_v * 0.25
               else '#7a3218' for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3.0))
    _theme(fig, ax)
    bars = ax.barh(labels, vals, color=palette, height=0.44, zorder=3, linewidth=0)

    fig.canvas.draw()
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for bar, val in zip(bars, vals):
        bw, pct = bar.get_width(), f'{val*100:.1f}%'
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
                    color='#888', fontfamily='monospace', zorder=5)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.tick_params(axis='y', colors='#c8c4bc', labelsize=10,  length=0, pad=10)
    ax.set_xlabel('Relative contribution to model predictions (XGBoost feature importance)',
                  fontsize=7.8, color='#3a3a3a', labelpad=10)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.22)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Actual vs Predicted scatter (precomputed)
# ══════════════════════════════════════════════════════════════════════════════
def render_actual_vs_predicted():
    act  = np.asarray(PERF_CURVES['act'], dtype=float)
    prd  = np.asarray(PERF_CURVES['prd'], dtype=float)
    mae  = PERF['mae']
    rmse = PERF['rmse']
    r2   = PERF['r2']

    fig, ax = plt.subplots(figsize=(8, 5.2))
    _theme(fig, ax)

    # ±MAE shaded band around perfect-prediction line
    lo = min(act.min(), prd.min()) * 0.92
    hi = max(act.max(), prd.max()) * 1.05
    ax.fill_between([lo, hi], [lo - mae, hi - mae], [lo + mae, hi + mae],
                    alpha=0.07, color=POS_COL, zorder=1,
                    label=f'±MAE band  (₹{mae:,.0f})')

    # Scatter
    ax.scatter(act, prd, alpha=0.4, s=14, color=POS_COL,
               linewidths=0, zorder=3)

    # Perfect-prediction line
    ax.plot([lo, hi], [lo, hi], color=NEG_COL, linewidth=1.6,
            linestyle='--', label='Perfect prediction', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _rupee_fmt(ax, 'both')
    ax.tick_params(colors='#555', labelsize=8.5, length=0, pad=5)
    ax.set_xlabel('Actual Sales', color='#555', fontsize=10, labelpad=10)
    ax.set_ylabel('Predicted Sales', color='#555', fontsize=10, labelpad=10)
    ax.set_title(f'Actual vs Predicted   R² = {r2:.4f}   RMSE = ₹{rmse:,.0f}',
                 color='#c8c4bc', fontsize=11, pad=14, loc='left')

    legend = ax.legend(fontsize=8.5, framealpha=0,
                       labelcolor='#666', loc='upper left',
                       handlelength=1.2, borderpad=0.4)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.12)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Quantile interval on 80 test samples (precomputed)
# ══════════════════════════════════════════════════════════════════════════════
def render_quantile_chart():
    act      = np.asarray(PERF_CURVES['act'], dtype=float)
    prd      = np.asarray(PERF_CURVES['prd'], dtype=float)
    low      = np.asarray(PERF_CURVES['low'], dtype=float)
    hig      = np.asarray(PERF_CURVES['hig'], dtype=float)
    cov      = PERF['coverage']
    x_range  = np.arange(len(act))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    _theme(fig, ax)

    ax.fill_between(x_range, low, hig, alpha=0.22, color=NEG_COL,
                    label='80% confidence interval', zorder=2)
    ax.plot(x_range, act, color='#e8e4dc', linewidth=1.1,
            alpha=0.85, label='Actual sales', zorder=3)
    ax.plot(x_range, prd, color=POS_COL, linewidth=1.4,
            alpha=0.92, label='Predicted (median)', zorder=4)

    _rupee_fmt(ax, 'y')
    ax.tick_params(axis='y', colors='#555', labelsize=8.5, length=0, pad=5)
    ax.tick_params(axis='x', colors='#444', labelsize=8,   length=0, pad=5)
    ax.set_xlabel('Test samples  (sorted by actual sales)',
                  color='#555', fontsize=9.5, labelpad=10)
    ax.set_title(f'Quantile Regression — 80% Prediction Interval   '
                 f'(coverage {cov:.1f}%)',
                 color='#c8c4bc', fontsize=11, pad=14, loc='left')

    ax.legend(fontsize=8.5, framealpha=0, labelcolor='#666',
              loc='upper left', handlelength=1.4, borderpad=0.4)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.13)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 — SHAP waterfall (per prediction) — OVERLAP FIXED
# ══════════════════════════════════════════════════════════════════════════════
def render_shap_chart(sv, raw_display):
    """
    Overlap fix:
    - Raw input value is embedded in the Y-axis label: "Outlet Size (High)"
    - Bar annotation shows ONLY the ₹ delta — short enough to never clash
    - Legend moved to upper left, away from all bar ends
    - Near-zero bars (< 2% of span) show label on the correct side with extra pad
    """
    order         = np.argsort(np.abs(sv))
    sorted_sv     = sv[order]
    # Y-labels include raw value: "Outlet Size (High)"
    sorted_ylabels = [
        f"{FEATURE_LABELS[i]}\n({raw_display[FEATURE_LABELS[i]]})"
        for i in order
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _theme(fig, ax)

    colors = [POS_COL if v > 0 else NEG_COL for v in sorted_sv]
    bars   = ax.barh(range(len(sorted_ylabels)), sorted_sv,
                     color=colors, height=0.46, zorder=3, linewidth=0)
    ax.axvline(0, color='#2e2e2e', linewidth=1.4, zorder=2)

    # Y-axis with embedded raw values
    ax.set_yticks(range(len(sorted_ylabels)))
    ax.set_yticklabels(sorted_ylabels, fontsize=9, color='#c8c4bc',
                       linespacing=1.3)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

    fig.canvas.draw()
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    pad    = x_span * 0.015

    for i, (bar, sv_val) in enumerate(zip(bars, sorted_sv)):
        bw    = bar.get_width()
        sign  = '+' if sv_val >= 0 else ''
        label = f'{sign}₹{sv_val:,.0f}'          # ← delta only, NO raw value

        wide = abs(bw) >= x_span * 0.18           # threshold for inside annotation

        if wide:
            # Inside the bar — right end for positive, left end for negative
            x_pos = (bar.get_x() + bw - pad) if sv_val >= 0 \
                    else (bar.get_x() + bw + pad)
            ha, tc = ('right', '#f0ede6') if sv_val >= 0 else ('left', '#f0ede6')
        else:
            # Outside the bar — away from zero line
            x_pos = (bar.get_x() + bw + pad) if sv_val >= 0 \
                    else (bar.get_x() + bw - pad)
            ha, tc = ('left', '#888888') if sv_val >= 0 else ('right', '#888888')

        ax.text(x_pos, i, label, va='center', ha=ha,
                fontsize=8.5, color=tc, fontfamily='monospace',
                fontweight='bold', zorder=5)

    ax.set_xlabel('SHAP value  —  how much each feature moved the sales forecast',
                  fontsize=7.8, color='#3a3a3a', labelpad=10)

    # Legend — upper LEFT so it never touches bar-end labels on the right
    pos_p = mpatches.Patch(facecolor=POS_COL, label='Increases prediction', linewidth=0)
    neg_p = mpatches.Patch(facecolor=NEG_COL, label='Decreases prediction', linewidth=0)
    ax.legend(handles=[pos_p, neg_p], loc='upper left',
              fontsize=7.5, framealpha=0, labelcolor='#555',
              handlelength=1.0, handleheight=0.8,
              borderpad=0.3, labelspacing=0.3)

    # Extra left margin for two-line y-labels
    fig.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.18)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Confidence interval gauge (per prediction)
# ══════════════════════════════════════════════════════════════════════════════
def render_ci_chart(pred, lower, upper, is_quantile):
    fig, ax = plt.subplots(figsize=(8, 1.6))
    _theme(fig, ax)

    margin = (upper - lower) * 0.5
    x_lo   = max(0, lower - margin)
    x_hi   = upper + margin

    ax.barh(0, upper - lower, left=lower, height=0.55,
            color=NEG_COL, alpha=0.22, zorder=2, linewidth=0)
    ax.barh(0, upper - lower, left=lower, height=0.55,
            color='none', zorder=3, linewidth=1.0, edgecolor=NEG_COL)

    for xv, lbl in [(lower, f'₹{lower:,.0f}'), (upper, f'₹{upper:,.0f}')]:
        ax.axvline(xv, color=NEG_COL, linewidth=1.0, linestyle='--',
                   alpha=0.6, zorder=3)
        ax.text(xv, 0.52, lbl, va='bottom', ha='center',
                fontsize=7.5, color=NEG_COL, fontfamily='monospace')

    ax.axvline(pred, color=POS_COL, linewidth=2.5, zorder=5)
    ax.text(pred, -0.52, f'₹{pred:,.0f}', va='top', ha='center',
            fontsize=9, color=POS_COL, fontfamily='monospace',
            fontweight='bold')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.8, 0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.tick_params(axis='x', colors='#3a3a3a', labelsize=7.5, length=0, pad=5)
    ax.set_yticks([])
    ci_lbl = '80% quantile interval' if is_quantile else '±MAE range'
    ax.set_title(ci_lbl, color='#444', fontsize=8, loc='left', pad=6)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.72, bottom=0.28)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛒 Big Mart\nSales Predictor")
st.markdown("---")

# ── Feature Importance Dashboard ──────────────────────────────────────────────
st.markdown('<div class="section-header orange">📊 Feature Importance Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subhead">Global breakdown — how much each input '
    'contributes to the model\'s predictions across all outlets and items.</div>',
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

# ── Model Performance Section ─────────────────────────────────────────────────
with st.expander("📈 Model Performance  —  Evaluation on test set", expanded=False):

    # Stat pills
    st.markdown(f"""
    <div class="perf-stat-row">
        <div class="perf-stat">
            <div class="perf-stat-label">R²</div>
            <div class="perf-stat-value">{PERF['r2']:.4f}</div>
        </div>
        <div class="perf-stat">
            <div class="perf-stat-label">RMSE</div>
            <div class="perf-stat-value">₹{PERF['rmse']:,.0f}</div>
        </div>
        <div class="perf-stat">
            <div class="perf-stat-label">MAE</div>
            <div class="perf-stat-value">₹{PERF['mae']:,.0f}</div>
        </div>
        <div class="perf-stat">
            <div class="perf-stat-label">CI Coverage</div>
            <div class="perf-stat-value">{PERF['coverage']:.1f}%</div>
        </div>
        <div class="perf-stat">
            <div class="perf-stat-label">Avg CI Width</div>
            <div class="perf-stat-value">₹{PERF['ci_width']:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown(
        '<div class="section-header blue" style="margin-top:1.2rem">'
        '🎯 Actual vs Predicted</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subhead">'
        'Each dot is one test-set item. Points along the dashed line are perfect '
        'predictions. The fan shape shows the model is more accurate for '
        'low-to-mid sales items.</div>',
        unsafe_allow_html=True
    )
    avp_fig = render_actual_vs_predicted()
    st.pyplot(avp_fig, use_container_width=True)
    plt.close(avp_fig)

    st.markdown(f"""
    <div class="insight-box blue">
        The model explains <b>{PERF['r2']*100:.1f}%</b> of the variance in item
        sales (R² = {PERF['r2']:.4f}). The average error is
        <b>₹{PERF['mae']:,.0f}</b> (MAE) with an RMSE of
        <b>₹{PERF['rmse']:,.0f}</b>. High-sales outliers drive the RMSE up —
        the model is most reliable in the ₹500–₹5,000 range where most items sit.
    </div>
    """, unsafe_allow_html=True)

    # Quantile interval chart
    st.markdown(
        '<div class="section-header teal" style="margin-top:1.6rem">'
        '📐 Quantile Regression — 80% Prediction Interval</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subhead">'
        'Orange line = median prediction. Blue band = 80% confidence interval '
        'from Q10/Q90 quantile models. Notice the band widens at high sales — '
        'the model correctly signals more uncertainty there.</div>',
        unsafe_allow_html=True
    )
    qi_fig = render_quantile_chart()
    st.pyplot(qi_fig, use_container_width=True)
    plt.close(qi_fig)

    st.markdown(f"""
    <div class="insight-box teal">
        The 80% interval achieves <b>{PERF['coverage']:.1f}% coverage</b> on the
        test set (target ≥ 80%). The average band width is
        <b>₹{PERF['ci_width']:,.0f}</b>. Coverage is slightly under target
        because high-sales items are genuinely harder to bound — this is honest
        behaviour, not a model failure.
    </div>
    """, unsafe_allow_html=True)

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

        input_arr = np.array([[p1, p2, p3, p4, p5]])
        pred      = model.predict(input_arr)[0]

        # Confidence interval
        is_quantile = False
        try:
            q_low, q_high = load_quantile_models()
            lower       = max(0.0, float(q_low.predict(input_arr)[0]))
            upper       = float(q_high.predict(input_arr)[0])
            is_quantile = True
        except Exception:
            lower = max(0.0, pred - MAE)
            upper = pred + MAE

        ci_label = "80% Confidence Interval" if is_quantile else "±MAE Range"
        ci_note  = "quantile regression" if is_quantile else "fixed ±MAE fallback"

        # Result box
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Sales</div>
            <div class="result-value">&#8377;{pred:,.0f}</div>
            <div class="result-range">
                <b>{ci_label}</b> &nbsp;&middot;&nbsp;
                &#8377;{lower:,.0f} &mdash; &#8377;{upper:,.0f}
                <span style="color:#2a2a2a;font-size:0.75rem">&nbsp;({ci_note})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards
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

        # CI gauge
        st.markdown(
            '<div class="section-header teal" style="margin-top:1.4rem">'
            '📐 Prediction Interval</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="section-subhead">Orange line = point estimate. '
            f'Blue band = {ci_label.lower()}. '
            f'Actual sales land inside this band ~80% of the time.</div>',
            unsafe_allow_html=True
        )
        ci_fig = render_ci_chart(pred, lower, upper, is_quantile)
        st.pyplot(ci_fig, use_container_width=True)
        plt.close(ci_fig)

        if is_quantile:
            st.markdown(f"""
            <div class="insight-box teal">
                The 80% prediction interval spans
                <b>₹{upper - lower:,.0f}</b>
                (₹{lower:,.0f} → ₹{upper:,.0f}).
                Computed by dedicated Q10 and Q90 XGBoost quantile models —
                a data-driven range, not a fixed offset.
            </div>
            """, unsafe_allow_html=True)

        # SHAP
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
                'up (orange) or down (blue) for this specific input. '
                'The feature value is shown on the Y-axis label.</div>',
                unsafe_allow_html=True
            )

            shap_fig = render_shap_chart(sv, raw_display)
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)

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
                Run the Save SHAP Explainer cell in the notebook, commit, redeploy.
            </div>
            """, unsafe_allow_html=True)

        except Exception as shap_err:
            st.markdown(
                f'<div class="soft-error">SHAP unavailable: {shap_err}</div>',
                unsafe_allow_html=True
            )

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
