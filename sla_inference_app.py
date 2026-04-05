import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import holidays
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SLA Breach Risk Monitor",
    page_icon="🔴",
    layout="wide",
)

# ── Load model artefacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("sla_xgboost_model.json")
    with open("model_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    return model, cfg

try:
    model, cfg = load_model()
    scaler        = cfg["scaler"]
    feature_cols  = cfg["feature_cols"]
    threshold     = cfg.get("optimal_threshold", 0.5)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.info("Upload `sla_xgboost_model.json` and `model_config.pkl` to the app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates training-time feature engineering.

    Expects a DataFrame with a datetime index (hourly) and at minimum
    a numeric column for each traffic metric present in the original CSV.
    All raw numeric columns are treated as traffic metrics.
    """
    df = df.copy()

    # --- ensure datetime index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Z-scores (global) ──────────────────────────────────────────────────
    for col in numeric_cols:
        m, s = df[col].mean(), df[col].std()
        df[f"{col}_zscore"] = (df[col] - m) / s if s > 0 else 0.0

    # ── Rolling 24-hour statistics ─────────────────────────────────────────
    for col in numeric_cols:
        r = df[col].rolling(window=24, min_periods=1)
        df[f"{col}_roll24_mean"] = r.mean()
        df[f"{col}_roll24_std"]  = r.std().fillna(0)
        df[f"{col}_roll24_min"]  = r.min()
        df[f"{col}_roll24_max"]  = r.max()

    # ── Lag features ───────────────────────────────────────────────────────
    for col in numeric_cols:
        for lag in [1, 6, 24]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ── Time features ──────────────────────────────────────────────────────
    df["hour"]      = df.index.hour
    df["dayofweek"] = df.index.dayofweek          # 0=Monday … 6=Sunday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Czech public holidays (extend country list if needed)
    try:
        cz_holidays = holidays.CZ(years=df.index.year.unique().tolist())
        df["is_holiday"] = df.index.normalize().isin(cz_holidays).astype(int)
    except Exception:
        df["is_holiday"] = 0

    # ── Drop rows that are all-NaN after lag shifting ──────────────────────
    df = df.dropna(subset=[c for c in df.columns if "_lag" in c], how="all")

    return df


def prepare_for_model(df_feat: pd.DataFrame) -> np.ndarray:
    """Select feature_cols, scale, and return numpy array ready for inference."""
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(
            f"The following required features are missing after engineering:\n{missing}\n\n"
            "Check that your CSV contains the same traffic columns used during training."
        )
    X = df_feat[feature_cols].copy()
    X = X.fillna(X.median())
    X_scaled = scaler.transform(X)
    return X_scaled


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🔴 SLA Breach Risk Monitor")
st.markdown(
    "Upload an **hourly network traffic CSV** for a subnet. "
    "The model will predict the probability of an SLA breach for each hour."
)

with st.sidebar:
    st.header("Model Info")
    st.metric("Decision Threshold", f"{threshold:.3f}")
    st.metric("Features used", len(feature_cols))
    with st.expander("Feature list"):
        st.write(feature_cols)
    st.markdown("---")
    st.markdown(
        "**Model**: XGBoost binary classifier  \n"
        "**Data**: CESNET-TimeSeries24  \n"
        "**Granularity**: 1 hour  \n"
    )

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload CSV (must have a datetime column + numeric traffic columns)",
    type=["csv"],
)

if uploaded is None:
    st.info("Waiting for CSV upload.")
    st.stop()

# ── Parse CSV ──────────────────────────────────────────────────────────────────
try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Auto-detect datetime column
dt_col = None
for col in raw.columns:
    try:
        parsed = pd.to_datetime(raw[col], infer_datetime_format=True)
        dt_col = col
        raw[col] = parsed
        break
    except Exception:
        continue

if dt_col is None:
    st.error(
        "No datetime column detected. "
        "Ensure one column contains timestamps (e.g. 'timestamp', 'datetime', 'time')."
    )
    st.stop()

raw = raw.set_index(dt_col).sort_index()
raw = raw.select_dtypes(include=[np.number])   # keep only numeric columns

if raw.empty:
    st.error("No numeric columns found after parsing. Check your CSV format.")
    st.stop()

st.success(f"Loaded **{len(raw)} rows** from `{uploaded.name}`  |  columns: {list(raw.columns)}")

# ── Feature engineering ────────────────────────────────────────────────────────
with st.spinner("Engineering features…"):
    try:
        df_feat = engineer_features(raw)
        X = prepare_for_model(df_feat)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()

# ── Inference ──────────────────────────────────────────────────────────────────
proba  = model.predict_proba(X)[:, 1]
labels = (proba >= threshold).astype(int)

results = pd.DataFrame(
    {"breach_probability": proba, "breach_predicted": labels},
    index=df_feat.index,
)

# ── Summary metrics ────────────────────────────────────────────────────────────
n_breach = labels.sum()
pct      = 100 * n_breach / len(labels) if len(labels) else 0
avg_risk = proba.mean()
max_risk = proba.max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Hours analysed",  len(results))
col2.metric("Breach hours",    n_breach)
col3.metric("Breach rate",     f"{pct:.1f}%")
col4.metric("Max risk score",  f"{max_risk:.3f}")

st.markdown("---")

# ── Risk timeline chart ────────────────────────────────────────────────────────
st.subheader("Hourly Breach Risk Timeline")

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(results.index, results["breach_probability"],
                alpha=0.3, color="steelblue", label="Risk probability")
ax.plot(results.index, results["breach_probability"],
        color="steelblue", linewidth=0.8)
ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
           label=f"Threshold ({threshold:.3f})")

# Shade breach periods
breach_mask = results["breach_predicted"] == 1
ax.fill_between(results.index, 0, 1,
                where=breach_mask, alpha=0.15, color="red",
                transform=ax.get_xaxis_transform(), label="Predicted breach")

ax.set_ylim(0, 1)
ax.set_ylabel("Breach probability")
ax.set_xlabel("Time")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate(rotation=30)
ax.legend(loc="upper right")
ax.set_title("SLA Breach Risk — Hourly Forecast")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Results table ──────────────────────────────────────────────────────────────
st.subheader("Prediction Table")

show_breach_only = st.checkbox("Show breach hours only", value=False)
display_df = results[results["breach_predicted"] == 1] if show_breach_only else results
display_df = display_df.copy()
display_df["breach_probability"] = display_df["breach_probability"].round(4)

st.dataframe(
    display_df.style.background_gradient(
        subset=["breach_probability"], cmap="RdYlGn_r", vmin=0, vmax=1
    ),
    use_container_width=True,
)

# ── Download ───────────────────────────────────────────────────────────────────
csv_bytes = results.to_csv().encode()
st.download_button(
    label="Download predictions CSV",
    data=csv_bytes,
    file_name="sla_predictions.csv",
    mime="text/csv",
)
