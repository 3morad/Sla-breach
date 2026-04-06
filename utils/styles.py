import streamlit as st

# ── Per-model accent colours ──────────────────────────────────────────────────
MODEL_COLORS = {
    "forecasting": "#3b82f6",
    "anomaly":     "#f59e0b",
    "sla":         "#ef4444",
    "clustering":  "#8b5cf6",
    "correlation": "#10b981",
}

CUSTOM_CSS = """
<style>
/* ── Layout ──────────────────────────────────────── */
.block-container { padding-top: 1.5rem !important; }
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* ── Hero banner ─────────────────────────────────── */
.dashboard-hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto 1rem;
}

/* ── Model cards ─────────────────────────────────── */
.model-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.25rem 1.25rem 1rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.model-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.card-forecasting::before { background: #3b82f6; }
.card-anomaly::before     { background: #f59e0b; }
.card-sla::before         { background: #ef4444; }
.card-clustering::before  { background: #8b5cf6; }
.card-correlation::before { background: #10b981; }

.card-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}
.card-purpose {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.75rem;
}
.card-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

/* ── BO badges ───────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.45rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin: 0.1rem 0.1rem 0 0;
}
.bo1 { background:#1e3a5f; color:#60a5fa; }
.bo2 { background:#312e81; color:#a78bfa; }
.bo3 { background:#064e3b; color:#6ee7b7; }
.bo4 { background:#78350f; color:#fcd34d; }
.bo6 { background:#1e1b4b; color:#c4b5fd; }
.bo7 { background:#0c4a6e; color:#7dd3fc; }

/* ── Sidebar data status ─────────────────────────── */
.ds-ok {
    background: #052e16;
    border: 1px solid #16a34a;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    color: #4ade80;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}
.ds-missing {
    background: #1c1917;
    border: 1px dashed #44403c;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    color: #78716c;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}

/* ── Section label ───────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.5rem;
}

/* ── Alert rows ──────────────────────────────────── */
.alert-high {
    background: #3b0f0f;
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    margin: 0.3rem 0;
}
.alert-ok {
    background: #052e16;
    border-left: 3px solid #22c55e;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    margin: 0.3rem 0;
}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_data_status():
    """Show shared data status in sidebar. Returns the df if loaded."""
    df = st.session_state.get("df")
    if df is not None:
        st.sidebar.markdown(
            f'<div class="ds-ok">✅ Data loaded — '
            f'<strong>{len(df):,}</strong> rows × '
            f'<strong>{len(df.columns)}</strong> cols</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="ds-missing">⚠️ No data — go to <b>Upload Data</b></div>',
            unsafe_allow_html=True,
        )
    return df


def no_data_wall():
    """Full-page block when session data is missing."""
    st.warning("No data loaded yet.", icon="📂")
    st.info("Go to **Upload Data** in the sidebar to upload your CSV first.")
    st.stop()


def page_header(icon: str, title: str, purpose: str, bos: list[str], color: str):
    """Consistent top-of-page header for every model page."""
    bo_html = "".join(f'<span class="badge {b.lower()}">{b}</span>' for b in bos)
    st.markdown(
        f"""
        <div style="border-left:4px solid {color};padding:0.5rem 1rem;margin-bottom:1.5rem;
                    background:{'#0d1b2a' if color == '#3b82f6' else '#111827'};border-radius:0 8px 8px 0">
            <div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">{icon} {title}</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem">{purpose}</div>
            <div style="margin-top:0.5rem">{bo_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
