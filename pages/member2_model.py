"""
Member 2 — Time-Series Forecasting Model  (MOST IMPORTANT)
Holt-Winters ETS with trend + seasonality, confidence intervals, STL decomp.
Covers: BO1 (SLA prediction), BO3 (baseline behaviour), BO7 (energy optimisation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from utils.styles import inject_css, sidebar_data_status, no_data_wall, page_header

st.set_page_config(page_title="Traffic Forecasting", page_icon="🔮", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Traffic Forecasting")
    st.markdown("---")
    df_shared = sidebar_data_status()
    st.markdown("---")
    st.markdown("**Forecast settings**")
    horizon     = st.slider("Horizon (hours)",    1, 168, 24)
    seasonal_p  = st.selectbox("Seasonal period", [24, 168], index=0,
                               format_func=lambda x: f"{x}h ({'daily' if x==24 else 'weekly'})")
    trend_type  = st.selectbox("Trend", ["add", "mul", "None"],
                               format_func=lambda x: x.capitalize())
    seas_type   = st.selectbox("Seasonality", ["add", "mul"],
                               format_func=lambda x: x.capitalize())
    st.markdown("---")
    st.markdown(
        "**Algorithm:** Holt-Winters ETS  \n"
        "**Output:** Future n_bytes, n_flows …  \n"
        "**Covers:** BO1, BO3, BO7"
    )

# ── Guard ──────────────────────────────────────────────────────────────────────
if df_shared is None:
    no_data_wall()

page_header(
    "🔮", "Time-Series Traffic Forecasting",
    "Predict future traffic and KPIs using Holt-Winters exponential smoothing.",
    ["BO1", "BO3", "BO7"], "#3b82f6",
)

# ── Column selector ────────────────────────────────────────────────────────────
target = st.selectbox("Select metric to forecast", df_shared.columns.tolist())
series = df_shared[target].dropna()

if len(series) < seasonal_p * 2:
    st.warning(f"Need ≥ {seasonal_p * 2} hours of data for {seasonal_p}h seasonality. "
               "Try a smaller seasonal period or upload more data.")
    st.stop()

# ── Fit model ──────────────────────────────────────────────────────────────────
with st.spinner("Fitting Holt-Winters model…"):
    try:
        t = None if trend_type == "None" else trend_type
        fit = ExponentialSmoothing(
            series,
            trend=t,
            seasonal=seas_type,
            seasonal_periods=seasonal_p,
        ).fit(optimized=True)

        forecast   = fit.forecast(horizon)
        resid_std  = np.std(fit.resid)
        ci_upper   = forecast + 1.96 * resid_std
        ci_lower   = (forecast - 1.96 * resid_std).clip(lower=0)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical hours",  f"{len(series):,}")
c2.metric("Forecast horizon",  f"{horizon} h")
c3.metric("Mean forecast",     f"{forecast.mean():.2f}")
c4.metric("AIC",               f"{fit.aic:.1f}" if hasattr(fit, "aic") else "—")

st.markdown("---")

# ── Main forecast chart ────────────────────────────────────────────────────────
st.subheader("Forecast Chart")

# Limit historical display to last 7× the horizon for readability
show_hist = series.iloc[-min(len(series), horizon * 7):]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=show_hist.index, y=show_hist.values,
    name="Historical", line=dict(color="#60a5fa", width=1.5),
))
fig.add_trace(go.Scatter(
    x=fit.fittedvalues.iloc[-len(show_hist):].index,
    y=fit.fittedvalues.iloc[-len(show_hist):].values,
    name="Fitted", line=dict(color="#93c5fd", width=1, dash="dot"),
    opacity=0.6,
))
# CI shading
fig.add_trace(go.Scatter(
    x=list(ci_upper.index) + list(ci_lower.index[::-1]),
    y=list(ci_upper.values) + list(ci_lower.values[::-1]),
    fill="toself", fillcolor="rgba(251,191,36,0.12)",
    line=dict(color="rgba(0,0,0,0)"), name="95% CI",
))
fig.add_trace(go.Scatter(
    x=forecast.index, y=forecast.values,
    name="Forecast", line=dict(color="#fbbf24", width=2.5),
    mode="lines+markers", marker=dict(size=5),
))

fig.update_layout(
    template="plotly_dark", height=400,
    xaxis_title="Time", yaxis_title=target,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Decomposition ──────────────────────────────────────────────────────────────
st.subheader("Trend & Seasonal Decomposition")

if len(series) >= seasonal_p * 2:
    try:
        decomp = seasonal_decompose(
            series, period=seasonal_p, model="additive",
            extrapolate_trend="freq",
        )
        comp_data = {
            "Observed":  decomp.observed,
            "Trend":     decomp.trend,
            "Seasonal":  decomp.seasonal,
            "Residual":  decomp.resid,
        }
        colors = ["#60a5fa", "#34d399", "#f59e0b", "#f87171"]
        tab1, tab2, tab3, tab4 = st.tabs(list(comp_data.keys()))
        for tab, (name, comp), color in zip([tab1,tab2,tab3,tab4], comp_data.items(), colors):
            with tab:
                fig_c = go.Figure(go.Scatter(
                    x=comp.index, y=comp.values,
                    line=dict(color=color, width=1.5), mode="lines",
                ))
                fig_c.update_layout(
                    template="plotly_dark", height=250,
                    yaxis_title=name, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_c, use_container_width=True)
    except Exception as e:
        st.warning(f"Decomposition unavailable: {e}")

# ── Forecast table + download ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Forecast Table")

fc_df = pd.DataFrame({
    "timestamp":  forecast.index,
    "forecast":   forecast.values.round(2),
    "ci_lower":   ci_lower.values.round(2),
    "ci_upper":   ci_upper.values.round(2),
})
st.dataframe(fc_df, use_container_width=True, height=280)

st.download_button(
    "⬇️ Download forecast CSV",
    fc_df.to_csv(index=False).encode(),
    f"forecast_{target}.csv",
    "text/csv",
)
