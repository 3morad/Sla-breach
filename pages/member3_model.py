"""
Member 3 — Anomaly Detection Model
Isolation Forest trained on-the-fly; outputs anomaly score per timestamp.
Covers: BO1 (latent degradation), BO4 (incident detection), BO7 (green-mode degradation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.styles import inject_css, sidebar_data_status, no_data_wall, page_header

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Anomaly Detection")
    st.markdown("---")
    df_shared = sidebar_data_status()
    st.markdown("---")
    st.markdown("**Model settings**")
    contamination = st.slider(
        "Contamination (expected anomaly fraction)",
        0.01, 0.20, 0.05, 0.01,
        help="Fraction of hours expected to be anomalous.",
    )
    n_estimators = st.slider("Number of trees", 50, 300, 100, 50)
    use_rolling  = st.checkbox("Include rolling-24h features", value=True)
    st.markdown("---")
    st.markdown(
        "**Algorithm:** Isolation Forest  \n"
        "**Output:** Anomaly score per timestamp  \n"
        "**Covers:** BO1, BO4, BO7"
    )

# ── Guard ──────────────────────────────────────────────────────────────────────
if df_shared is None:
    no_data_wall()

page_header(
    "🚨", "Anomaly Detection Model",
    "Detect abnormal network behaviour without labels using Isolation Forest.",
    ["BO1", "BO4", "BO7"], "#f59e0b",
)

# ── Feature preparation ────────────────────────────────────────────────────────
with st.spinner("Building features and training Isolation Forest…"):
    df = df_shared.copy()
    feat_df = df.copy()

    if use_rolling:
        for col in df.columns:
            r = df[col].rolling(24, min_periods=1)
            feat_df[f"{col}_roll_mean"] = r.mean()
            feat_df[f"{col}_roll_std"]  = r.std().fillna(0)

    feat_df["hour"]      = feat_df.index.hour
    feat_df["dayofweek"] = feat_df.index.dayofweek
    feat_df = feat_df.fillna(feat_df.median())

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled)

    scores  = -clf.score_samples(X_scaled)   # higher = more anomalous
    labels  = clf.predict(X_scaled)           # -1 = anomaly, 1 = normal
    is_anom = (labels == -1).astype(int)

results = pd.DataFrame(
    {"anomaly_score": scores, "is_anomaly": is_anom},
    index=feat_df.index,
)

# Normalise scores to 0-1
results["anomaly_score_norm"] = (
    (results["anomaly_score"] - results["anomaly_score"].min()) /
    (results["anomaly_score"].max() - results["anomaly_score"].min() + 1e-9)
)

# ── KPI row ────────────────────────────────────────────────────────────────────
n_anom = int(is_anom.sum())
pct    = 100 * n_anom / len(results)
thresh = results.loc[results["is_anomaly"] == 1, "anomaly_score_norm"].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hours analysed",    f"{len(results):,}")
c2.metric("Anomalous hours",   n_anom)
c3.metric("Anomaly rate",      f"{pct:.1f}%")
c4.metric("Contamination",     f"{contamination:.0%}")

st.markdown("---")

# ── Anomaly score timeline ─────────────────────────────────────────────────────
st.subheader("Anomaly Score Timeline")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results.index, y=results["anomaly_score_norm"],
    fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
    line=dict(color="#f59e0b", width=1.2),
    name="Anomaly score", mode="lines",
))
# Mark anomalies
anom_idx = results[results["is_anomaly"] == 1]
fig.add_trace(go.Scatter(
    x=anom_idx.index, y=anom_idx["anomaly_score_norm"],
    mode="markers", marker=dict(color="#ef4444", size=5, symbol="x"),
    name="Flagged anomaly",
))
fig.update_layout(
    template="plotly_dark", height=360,
    yaxis=dict(range=[0, 1.05], title="Normalised anomaly score"),
    xaxis_title="Time",
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── PCA scatter + distribution ─────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("PCA — Normal vs Anomaly")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "type": ["Anomaly" if a else "Normal" for a in is_anom],
    }, index=feat_df.index)
    fig2 = px.scatter(
        pca_df, x="PC1", y="PC2", color="type",
        color_discrete_map={"Normal": "#60a5fa", "Anomaly": "#ef4444"},
        opacity=0.6, template="plotly_dark", height=300,
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Score Distribution")
    fig3 = go.Figure()
    for label, color, mask in [
        ("Normal",  "#60a5fa", is_anom == 0),
        ("Anomaly", "#ef4444", is_anom == 1),
    ]:
        fig3.add_trace(go.Histogram(
            x=results.loc[mask, "anomaly_score_norm"],
            name=label, marker_color=color,
            opacity=0.7, nbinsx=40,
        ))
    fig3.update_layout(
        barmode="overlay", template="plotly_dark", height=300,
        xaxis_title="Normalised anomaly score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Anomaly events table ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Flagged Events")

show_all = st.checkbox("Show all hours (not just anomalies)", value=False)
disp = results if show_all else results[results["is_anomaly"] == 1]
disp = disp.sort_values("anomaly_score_norm", ascending=False).copy()
disp["anomaly_score_norm"] = disp["anomaly_score_norm"].round(4)
disp["anomaly_score"]      = disp["anomaly_score"].round(4)

st.dataframe(
    disp.style.background_gradient(
        subset=["anomaly_score_norm"], cmap="YlOrRd", vmin=0, vmax=1,
    ),
    use_container_width=True,
    height=300,
)

st.download_button(
    "⬇️ Download anomaly results CSV",
    results.to_csv().encode(),
    "anomaly_results.csv",
    "text/csv",
)
