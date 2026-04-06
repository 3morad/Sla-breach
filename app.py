"""
NetPulse Analytics — Telecom Network Intelligence Platform
Entry point / Home page
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils.styles import inject_css, sidebar_data_status

st.set_page_config(
    page_title="NetPulse Analytics",
    page_icon="🛜",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛜 NetPulse Analytics")
    st.markdown("---")
    sidebar_data_status()
    st.markdown("---")
    st.markdown(
        """
        **Quick start**
        1. Go to **Upload Data**
        2. Upload your hourly CSV
        3. Pick any model page
        """,
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.caption("CESNET-TimeSeries24 · XGBoost · Isolation Forest · K-Means · ETS")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="dashboard-hero">
        <div class="hero-title">🛜 NetPulse Analytics</div>
        <div class="hero-subtitle">
            AI-powered network intelligence platform for Telecom ISPs —
            forecast traffic, detect anomalies, predict SLA breaches, segment subnets,
            and map KPI dependencies from a single dashboard.
        </div>
        <div style="display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;margin-top:1rem">
            <span style="background:#1e3a5f;color:#60a5fa;padding:0.3rem 0.8rem;
                         border-radius:20px;font-size:0.75rem;font-weight:700">
                CESNET-TimeSeries24
            </span>
            <span style="background:#052e16;color:#4ade80;padding:0.3rem 0.8rem;
                         border-radius:20px;font-size:0.75rem;font-weight:700">
                5 AI Models
            </span>
            <span style="background:#1e1b4b;color:#c4b5fd;padding:0.3rem 0.8rem;
                         border-radius:20px;font-size:0.75rem;font-weight:700">
                Hourly Resolution
            </span>
            <span style="background:#3b0f0f;color:#fca5a5;padding:0.3rem 0.8rem;
                         border-radius:20px;font-size:0.75rem;font-weight:700">
                Real-time Ready
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Model cards ────────────────────────────────────────────────────────────────
MODELS = [
    {
        "key": "forecasting", "icon": "🔮", "num": "01",
        "title": "Time-Series Forecasting",
        "purpose": "Predict future traffic & KPIs",
        "desc": "Holt-Winters ETS model with trend + seasonality. Outputs hourly forecasts with 95% confidence intervals and STL decomposition.",
        "bos": ["BO1", "BO3", "BO7"],
        "models": "ETS · Holt-Winters · SARIMA",
        "output": "Future n_bytes, n_flows, etc.",
        "color": "#3b82f6",
        "page": "pages/member2_model.py",
    },
    {
        "key": "anomaly", "icon": "🚨", "num": "02",
        "title": "Anomaly Detection",
        "purpose": "Detect abnormal behaviour without labels",
        "desc": "Isolation Forest trained on-the-fly on your data. Flags hours with anomalous traffic patterns by anomaly score.",
        "bos": ["BO1", "BO4", "BO7"],
        "models": "Isolation Forest · One-Class SVM",
        "output": "Anomaly score per timestamp",
        "color": "#f59e0b",
        "page": "pages/member3_model.py",
    },
    {
        "key": "sla", "icon": "🔴", "num": "03",
        "title": "SLA Violation Risk",
        "purpose": "Predict probability of SLA breach",
        "desc": "Pre-trained XGBoost classifier with optimal threshold. Uses z-scores, rolling stats, lag features, and time features.",
        "bos": ["BO1", "BO4"],
        "models": "XGBoost · Random Forest · Logistic Regression",
        "output": "Breach probability (0–1)",
        "color": "#ef4444",
        "page": "pages/member1_model.py",
    },
    {
        "key": "clustering", "icon": "🔵", "num": "04",
        "title": "Clustering & Root Cause",
        "purpose": "Group subnets and detect affected areas",
        "desc": "K-Means and DBSCAN on PCA-reduced features. Auto-labels clusters as normal / congested / spiky based on centroid stats.",
        "bos": ["BO2", "BO4", "BO6"],
        "models": "K-Means · DBSCAN · Hierarchical",
        "output": "Cluster label per hour",
        "color": "#8b5cf6",
        "page": "pages/member4_model.py",
    },
    {
        "key": "correlation", "icon": "📊", "num": "05",
        "title": "Correlation & Dependency",
        "purpose": "Understand relationships between KPIs",
        "desc": "Pearson / Spearman correlation matrix, rolling correlation over time, and an interactive dependency network graph.",
        "bos": ["BO2"],
        "models": "Pearson · Spearman · Rolling · Graph",
        "output": '"When X increases, Y also increases"',
        "color": "#10b981",
        "page": "pages/member5_model.py",
    },
]

st.markdown('<div class="section-label">AI Model Suite</div>', unsafe_allow_html=True)

# 3-column top row, 2-column bottom row
row1 = st.columns(3, gap="medium")
row2_wrap = st.columns([1, 1, 1], gap="medium")   # centre-align the last 2

for i, (col, model) in enumerate(zip(list(row1) + list(row2_wrap[:2]), MODELS)):
    with col:
        bo_html = "".join(
            f'<span class="badge {b.lower()}">{b}</span>' for b in model["bos"]
        )
        st.markdown(
            f"""
            <div class="model-card card-{model['key']}">
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">
                    <span style="font-size:1.6rem;line-height:1">{model['icon']}</span>
                    <div>
                        <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;
                                    color:{model['color']};text-transform:uppercase">
                            MODEL {model['num']}
                        </div>
                        <div class="card-title">{model['title']}</div>
                    </div>
                </div>
                <div class="card-purpose">{model['purpose']}</div>
                <div class="card-desc">{model['desc']}</div>
                <div style="font-size:0.72rem;color:#4b5563;margin-bottom:0.5rem">
                    <strong style="color:#6b7280">Algorithms:</strong> {model['models']}
                </div>
                <div style="font-size:0.72rem;color:#4b5563;margin-bottom:0.6rem">
                    <strong style="color:#6b7280">Output:</strong> {model['output']}
                </div>
                <div>{bo_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Getting started ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">How to Use</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
steps = [
    ("1️⃣", "Upload Data", "Go to **Upload Data** in the sidebar. Upload your hourly subnet CSV once — it's shared across all pages."),
    ("2️⃣", "Pick a Model", "Navigate to any model page. Each page reads from the shared data automatically."),
    ("3️⃣", "Configure", "Adjust model parameters in the sidebar (horizon, contamination, n_clusters, etc.)."),
    ("4️⃣", "Download", "Every model page provides a **Download CSV** button for the results."),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
    col.markdown(
        f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;
                    padding:1rem;text-align:center;height:100%">
            <div style="font-size:1.8rem;margin-bottom:0.5rem">{icon}</div>
            <div style="font-weight:700;color:#f1f5f9;margin-bottom:0.4rem;font-size:0.9rem">{title}</div>
            <div style="font-size:0.78rem;color:#94a3b8">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "NetPulse Analytics · Built on CESNET-TimeSeries24 · "
    "XGBoost · Isolation Forest · ETS · K-Means · DBSCAN · Pearson/Spearman"
)
