"""
Member 1 — SLA Violation / Risk Model
XGBoost binary classifier: predicts hourly SLA breach probability.
Covers: BO1 (core requirement), BO4 (incident prioritisation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

from utils.styles import inject_css, sidebar_data_status, no_data_wall, page_header
from utils.feature_engineering import engineer_features, prepare_for_model

st.set_page_config(page_title="SLA Breach Risk", page_icon="🔴", layout="wide")
inject_css()

# ── Load model artefacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_sla_model():
    model = xgb.XGBClassifier()
    model.load_model("sla_xgboost_model.json")
    with open("model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    return model, cfg["scaler"], cfg["feature_cols"], cfg.get("optimal_threshold", 0.5)

try:
    model, scaler, feature_cols, threshold = load_sla_model()
    model_ok = True
except FileNotFoundError:
    model_ok = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔴 SLA Breach Risk")
    st.markdown("---")
    df_shared = sidebar_data_status()
    st.markdown("---")
    if model_ok:
        st.metric("Decision threshold", f"{threshold:.3f}")
        st.metric("Features", len(feature_cols))
        with st.expander("Feature list"):
            st.write(feature_cols)
    else:
        st.warning("Model files not found.\n\nUpload `sla_xgboost_model.json` and `model_config.pkl` to the project root.")
    st.markdown("---")
    st.markdown(
        "**Algorithm:** XGBoost  \n"
        "**Output:** Breach probability (0–1)  \n"
        "**Covers:** BO1, BO4"
    )

# ── Guards ─────────────────────────────────────────────────────────────────────
if df_shared is None:
    no_data_wall()

if not model_ok:
    st.error("Model files missing. See sidebar for details.")
    st.stop()

# ── Page header ────────────────────────────────────────────────────────────────
page_header(
    "🔴", "SLA Violation Risk Model",
    "Predict hourly probability of SLA breach using a pre-trained XGBoost classifier.",
    ["BO1", "BO4"], "#ef4444",
)

# ── Feature engineering & inference ───────────────────────────────────────────
with st.spinner("Engineering features and running inference…"):
    try:
        df_feat = engineer_features(df_shared)
        X = prepare_for_model(df_feat, feature_cols, scaler)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

proba  = model.predict_proba(X)[:, 1]
labels = (proba >= threshold).astype(int)

results = pd.DataFrame(
    {"breach_probability": proba, "breach_predicted": labels},
    index=df_feat.index,
)

# ── KPI row ────────────────────────────────────────────────────────────────────
n_breach = int(labels.sum())
pct      = 100 * n_breach / len(labels) if len(labels) else 0
avg_risk = float(proba.mean())
max_risk = float(proba.max())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hours analysed",  f"{len(results):,}")
c2.metric("Predicted breaches", n_breach)
c3.metric("Breach rate",    f"{pct:.1f}%")
c4.metric("Avg risk score", f"{avg_risk:.3f}")
c5.metric("Max risk score", f"{max_risk:.3f}")

# ── Risk timeline ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Hourly Breach Risk Timeline")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results.index, y=results["breach_probability"],
    fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
    line=dict(color="#ef4444", width=1.5),
    name="Risk probability", mode="lines",
))
fig.add_hline(
    y=threshold, line_dash="dash", line_color="#fbbf24", line_width=1.5,
    annotation_text=f"Threshold ({threshold:.3f})",
    annotation_position="top left",
    annotation_font_color="#fbbf24",
)
fig.update_layout(
    template="plotly_dark", height=380,
    yaxis=dict(range=[0, 1.05], title="Breach probability"),
    xaxis_title="Time",
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── Breach hours distribution ──────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Risk Distribution")
    fig2 = px.histogram(
        results, x="breach_probability", nbins=40,
        color_discrete_sequence=["#ef4444"],
        template="plotly_dark", height=280,
        labels={"breach_probability": "Breach probability"},
    )
    fig2.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24")
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Breach Hours by Day")
    breach_df = results[results["breach_predicted"] == 1].copy()
    if len(breach_df) > 0:
        breach_df["date"] = breach_df.index.date
        daily = breach_df.groupby("date").size().reset_index(name="breach_hours")
        fig3 = px.bar(
            daily, x="date", y="breach_hours",
            color_discrete_sequence=["#ef4444"],
            template="plotly_dark", height=280,
            labels={"date": "Date", "breach_hours": "Breach hours"},
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No breach hours predicted.")

# ── Predictions table ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Prediction Table")

show_only = st.checkbox("Show breach hours only", value=False)
disp = results[results["breach_predicted"] == 1] if show_only else results
disp = disp.copy()
disp["risk_level"] = pd.cut(
    disp["breach_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
)

st.dataframe(
    disp.style.background_gradient(
        subset=["breach_probability"], cmap="RdYlGn_r", vmin=0, vmax=1
    ),
    use_container_width=True,
    height=320,
)

# ── Download ───────────────────────────────────────────────────────────────────
st.download_button(
    "⬇️ Download predictions CSV",
    results.to_csv().encode(),
    "sla_predictions.csv",
    "text/csv",
)
