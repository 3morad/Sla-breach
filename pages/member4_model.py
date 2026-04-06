"""
Member 4 — Clustering Model (Root Cause & Segmentation)
K-Means and DBSCAN on PCA-reduced features; auto-labels clusters.
Covers: BO2 (fault localisation), BO4 (recurring patterns), BO6 (traffic classification)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.styles import inject_css, sidebar_data_status, no_data_wall, page_header

st.set_page_config(page_title="Clustering", page_icon="🔵", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔵 Clustering")
    st.markdown("---")
    df_shared = sidebar_data_status()
    st.markdown("---")
    st.markdown("**Algorithm**")
    algo = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True)
    st.markdown("---")
    if algo == "K-Means":
        n_clusters  = st.slider("Number of clusters (k)", 2, 8, 3)
        n_init      = st.slider("n_init", 5, 20, 10)
    else:
        eps         = st.slider("eps (neighbourhood radius)", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("min_samples", 2, 20, 5)
    st.markdown("---")
    st.markdown(
        "**Algorithms:** K-Means · DBSCAN  \n"
        "**Output:** Cluster label per hour  \n"
        "**Covers:** BO2, BO4, BO6"
    )

# ── Guard ──────────────────────────────────────────────────────────────────────
if df_shared is None:
    no_data_wall()

page_header(
    "🔵", "Clustering & Root Cause Analysis",
    "Segment traffic hours into behavioural clusters and identify recurring patterns.",
    ["BO2", "BO4", "BO6"], "#8b5cf6",
)

# ── Feature prep ───────────────────────────────────────────────────────────────
with st.spinner("Preparing features…"):
    df = df_shared.copy()

    # Add rolling stats + time features
    feat_df = df.copy()
    for col in df.columns:
        r = df[col].rolling(24, min_periods=1)
        feat_df[f"{col}_roll_mean"] = r.mean()
        feat_df[f"{col}_roll_std"]  = r.std().fillna(0)
    feat_df["hour"]      = feat_df.index.hour
    feat_df["dayofweek"] = feat_df.index.dayofweek
    feat_df = feat_df.fillna(feat_df.median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)

    # PCA for visualisation
    pca2 = PCA(n_components=2, random_state=42)
    coords2 = pca2.fit_transform(X_scaled)

# ── Run clustering ─────────────────────────────────────────────────────────────
with st.spinner(f"Running {algo}…"):
    if algo == "K-Means":
        clf    = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        labels = clf.fit_predict(X_scaled)
        cluster_ids = sorted(set(labels))
    else:
        clf    = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clf.fit_predict(X_scaled)
        cluster_ids = sorted(set(labels))
        n_clusters  = len([c for c in cluster_ids if c != -1])

# ── Auto-label clusters ────────────────────────────────────────────────────────
# Heuristic: compare cluster centroids against global stats

def auto_label(cluster_id: int, df_orig: pd.DataFrame, mask: np.ndarray) -> str:
    if cluster_id == -1:
        return "Noise"
    sub = df_orig[mask]
    if len(sub) == 0:
        return f"Cluster {cluster_id}"
    global_mean = df_orig.mean().mean()
    local_mean  = sub.mean().mean()
    local_std   = sub.std().mean()
    global_std  = df_orig.std().mean()
    ratio_mean  = local_mean / (global_mean + 1e-9)
    ratio_std   = local_std  / (global_std  + 1e-9)
    if ratio_mean > 1.4:
        return "Congested"
    if ratio_std > 1.5:
        return "Spiky"
    if ratio_mean < 0.6:
        return "Low-traffic"
    return "Normal"

label_map = {
    cid: auto_label(cid, df_shared, labels == cid)
    for cid in cluster_ids
}
label_names = [label_map[l] for l in labels]

results = pd.DataFrame(
    {"cluster_id": labels, "cluster_label": label_names},
    index=feat_df.index,
)

# ── KPI row ────────────────────────────────────────────────────────────────────
noise_n = int((labels == -1).sum()) if algo == "DBSCAN" else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hours analysed", f"{len(results):,}")
c2.metric("Clusters found",  n_clusters)
c3.metric("Noise points (DBSCAN)", noise_n if algo == "DBSCAN" else "—")
c4.metric("Algorithm", algo)

st.markdown("---")

# ── PCA scatter ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("Cluster Visualisation (PCA 2-D)")
    pca_df = pd.DataFrame({
        "PC1": coords2[:, 0],
        "PC2": coords2[:, 1],
        "Cluster": label_names,
    })
    COLOR_SEQ = px.colors.qualitative.Safe
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Cluster",
        color_discrete_sequence=COLOR_SEQ,
        opacity=0.7, template="plotly_dark", height=380,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Cluster Distribution")
    dist = results["cluster_label"].value_counts().reset_index()
    dist.columns = ["label", "count"]
    fig2 = px.pie(
        dist, names="label", values="count",
        color_discrete_sequence=COLOR_SEQ,
        template="plotly_dark", height=380, hole=0.35,
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Timeline coloured by cluster ───────────────────────────────────────────────
st.subheader("Cluster Assignment Timeline")

fig3 = go.Figure()
for cid in cluster_ids:
    mask = results["cluster_id"] == cid
    name = label_map[cid]
    fig3.add_trace(go.Scatter(
        x=results.index[mask],
        y=[cid] * mask.sum(),
        mode="markers",
        marker=dict(size=6, symbol="square"),
        name=f"{cid}: {name}",
    ))
fig3.update_layout(
    template="plotly_dark", height=220,
    yaxis=dict(title="Cluster ID", dtick=1),
    xaxis_title="Time",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig3, use_container_width=True)

# ── Cluster profiles ───────────────────────────────────────────────────────────
st.subheader("Cluster Profiles (Mean Feature Values)")

profile_data = []
raw_cols = df_shared.columns.tolist()
for cid in [c for c in cluster_ids if c != -1]:
    mask = results["cluster_id"] == cid
    row  = {"Cluster": f"{cid}: {label_map[cid]}", "Count": mask.sum()}
    row.update(df_shared[mask].mean().round(2).to_dict())
    profile_data.append(row)

if profile_data:
    profile_df = pd.DataFrame(profile_data).set_index("Cluster")
    st.dataframe(
        profile_df.style.background_gradient(cmap="Blues", subset=raw_cols),
        use_container_width=True,
    )

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.download_button(
    "⬇️ Download cluster results CSV",
    results.to_csv().encode(),
    "cluster_results.csv",
    "text/csv",
)
