"""
Upload Data — shared CSV upload page.
Parses the file, stores in st.session_state["df"], accessible on every page.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.styles import inject_css, sidebar_data_status
from utils.feature_engineering import parse_csv

st.set_page_config(page_title="Upload Data", page_icon="📂", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Data")
    st.markdown("---")
    sidebar_data_status()
    st.markdown("---")
    st.markdown(
        """
        **CSV requirements**
        - One column must be a timestamp
        - Remaining columns must be numeric
        - Hourly resolution recommended
        - Any number of traffic metrics
        """,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📂 Upload Network Traffic Data")
st.markdown(
    "Upload your **hourly subnet traffic CSV** once here. "
    "All model pages will read from this shared dataset automatically."
)

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a CSV file (max 200 MB)",
    type=["csv"],
    help="Must contain one datetime column and at least one numeric traffic metric.",
)

if uploaded is None:
    # Show existing data if already loaded
    df = st.session_state.get("df")
    if df is not None:
        st.info(
            f"Using previously uploaded data: "
            f"**{len(df):,} rows** · **{len(df.columns)} columns** · "
            f"from `{df.index[0]}` to `{df.index[-1]}`"
        )
    else:
        st.info("No data loaded yet. Upload a CSV above to get started.")
    st.stop()

# ── Parse ──────────────────────────────────────────────────────────────────────
with st.spinner("Parsing CSV…"):
    df = parse_csv(uploaded)

if df is None:
    st.stop()

# Store in shared session state
st.session_state["df"] = df
st.session_state["filename"] = uploaded.name

# ── Success summary ────────────────────────────────────────────────────────────
st.success(
    f"✅ **{uploaded.name}** loaded — "
    f"**{len(df):,} rows** · **{len(df.columns)} columns** · "
    f"from `{df.index[0]}` to `{df.index[-1]}`"
)

# ── Metrics row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
duration_h = (df.index[-1] - df.index[0]).total_seconds() / 3600
c1.metric("Total rows", f"{len(df):,}")
c2.metric("Numeric columns", len(df.columns))
c3.metric("Time span", f"{duration_h:.0f} h")
c4.metric("Missing values", f"{df.isna().sum().sum():,}")

st.markdown("---")

# ── Preview table ──────────────────────────────────────────────────────────────
st.subheader("Data Preview")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.dataframe(df.head(50), use_container_width=True, height=320)

with col_right:
    st.markdown("**Column statistics**")
    st.dataframe(df.describe().T.round(2), use_container_width=True, height=320)

# ── Traffic overview chart ─────────────────────────────────────────────────────
st.subheader("Traffic Overview")

sel_cols = st.multiselect(
    "Select columns to plot",
    df.columns.tolist(),
    default=df.columns[:min(3, len(df.columns))].tolist(),
)

if sel_cols:
    plot_df = df[sel_cols].copy()
    # Normalise to 0-1 for multi-column comparison
    normalise = st.checkbox("Normalise to 0–1 (for multi-column comparison)", value=len(sel_cols) > 1)
    if normalise:
        plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)

    fig = px.line(
        plot_df,
        labels={"value": "Normalised value" if normalise else "Value", "variable": "Column"},
        template="plotly_dark",
        height=340,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Column info ────────────────────────────────────────────────────────────────
with st.expander("Column details"):
    info = pd.DataFrame({
        "dtype":   df.dtypes,
        "non_null": df.notna().sum(),
        "null_%":  (df.isna().mean() * 100).round(1),
        "min":     df.min().round(2),
        "max":     df.max().round(2),
        "mean":    df.mean().round(2),
        "std":     df.std().round(2),
    })
    st.dataframe(info, use_container_width=True)

st.markdown("---")
st.markdown(
    "✅ Data is now stored in session state. "
    "Navigate to any **model page** in the sidebar to start analysis."
)
