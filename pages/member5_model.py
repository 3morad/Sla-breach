"""
Member 5 — Correlation / Dependency Model
Pearson & Spearman correlation, rolling correlation, dependency network graph.
Covers: BO2 (root cause analysis)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from utils.styles import inject_css, sidebar_data_status, no_data_wall, page_header

st.set_page_config(page_title="Correlation & Dependency", page_icon="📊", layout="wide")
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Correlation")
    st.markdown("---")
    df_shared = sidebar_data_status()
    st.markdown("---")
    corr_method = st.selectbox("Correlation method", ["pearson", "spearman"])
    roll_window = st.slider("Rolling window (hours)", 6, 168, 24, 6)
    graph_thresh = st.slider(
        "Network edge threshold |r| ≥",
        0.3, 0.95, 0.6, 0.05,
        help="Only draw edges between pairs whose |correlation| exceeds this value.",
    )
    st.markdown("---")
    st.markdown(
        "**Methods:** Pearson · Spearman · Rolling · Graph  \n"
        "**Output:** 'When X rises, Y also rises'  \n"
        "**Covers:** BO2"
    )

# ── Guard ──────────────────────────────────────────────────────────────────────
if df_shared is None:
    no_data_wall()

page_header(
    "📊", "Correlation & Dependency Analysis",
    "Understand relationships between KPIs using static correlation, rolling correlation, and a dependency network.",
    ["BO2"], "#10b981",
)

df = df_shared.copy()

if df.shape[1] < 2:
    st.error("Need at least 2 numeric columns to compute correlation.")
    st.stop()

# ── Static correlation matrix ──────────────────────────────────────────────────
st.subheader(f"Correlation Matrix ({corr_method.capitalize()})")

corr = df.corr(method=corr_method)

fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu",
    zmin=-1, zmax=1,
    text=corr.round(2).values,
    texttemplate="%{text}",
    colorbar=dict(title="r"),
))
fig.update_layout(
    template="plotly_dark", height=max(300, 60 * len(corr.columns)),
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Top correlated pairs ───────────────────────────────────────────────────────
st.subheader("Top Correlated Pairs")

pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
pairs.columns = ["KPI A", "KPI B", "correlation"]
pairs["abs_corr"] = pairs["correlation"].abs()
pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
pairs["correlation"] = pairs["correlation"].round(4)
pairs["relationship"] = pairs["correlation"].apply(
    lambda r: "Strong positive" if r > 0.7
    else ("Moderate positive" if r > 0.4
    else ("Negative" if r < -0.4
    else "Weak"))
)

col_a, col_b = st.columns([2, 1])
with col_a:
    st.dataframe(pairs.head(20), use_container_width=True, height=280)
with col_b:
    fig_bar = px.bar(
        pairs.head(10),
        x="correlation", y=pairs.head(10).apply(lambda r: f"{r['KPI A']} ↔ {r['KPI B']}", axis=1),
        orientation="h",
        color="correlation",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        template="plotly_dark", height=280,
        labels={"y": "", "correlation": "r"},
    )
    fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Rolling correlation ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Rolling {roll_window}h Correlation")

if df.shape[1] >= 2:
    col_x = st.selectbox("KPI A", df.columns.tolist(), index=0, key="rx")
    col_y = st.selectbox("KPI B", df.columns.tolist(),
                         index=min(1, len(df.columns) - 1), key="ry")

    if col_x != col_y:
        roll_corr = (
            df[col_x]
            .rolling(roll_window, min_periods=max(2, roll_window // 2))
            .corr(df[col_y])
        )
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr.values,
            fill="tozeroy", fillcolor="rgba(16,185,129,0.10)",
            line=dict(color="#10b981", width=1.5),
            mode="lines", name=f"r({col_x}, {col_y})",
        ))
        fig_rc.add_hline(y=0,    line_dash="dash", line_color="#6b7280", line_width=1)
        fig_rc.add_hline(y=0.7,  line_dash="dot",  line_color="#fbbf24", line_width=1,
                         annotation_text="r=0.7",  annotation_position="right")
        fig_rc.add_hline(y=-0.7, line_dash="dot",  line_color="#f87171", line_width=1,
                         annotation_text="r=-0.7", annotation_position="right")
        fig_rc.update_layout(
            template="plotly_dark", height=320,
            yaxis=dict(range=[-1.05, 1.05], title="Pearson r"),
            xaxis_title="Time",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("Select two different KPIs.")

# ── Dependency network graph ───────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Dependency Network  (edges where |r| ≥ {graph_thresh:.2f})")

G = nx.Graph()
G.add_nodes_from(df.columns.tolist())

for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if i >= j:
            continue
        r = corr.loc[c1, c2]
        if abs(r) >= graph_thresh:
            G.add_edge(c1, c2, weight=abs(r), corr=r)

if G.number_of_edges() == 0:
    st.info(f"No pairs with |r| ≥ {graph_thresh:.2f}. Lower the threshold in the sidebar.")
else:
    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "#10b981" if data["corr"] > 0 else "#ef4444"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, data["weight"] * 4), color=color),
            hoverinfo="none", showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    degrees = [G.degree(n) for n in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=11, color="#f1f5f9"),
        marker=dict(
            size=[8 + d * 5 for d in degrees],
            color="#8b5cf6",
            line=dict(color="#c4b5fd", width=1),
        ),
        hovertext=[
            f"{n}<br>Degree: {G.degree(n)}" for n in G.nodes()
        ],
        hoverinfo="text",
        showlegend=False,
    )

    fig_net = go.Figure(data=edge_traces + [node_trace])
    fig_net.update_layout(
        template="plotly_dark", height=480,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption(
        "Green edges = positive correlation · Red edges = negative correlation · "
        "Node size ∝ number of connections"
    )

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.download_button(
    "⬇️ Download correlation matrix CSV",
    corr.to_csv().encode(),
    "correlation_matrix.csv",
    "text/csv",
)
