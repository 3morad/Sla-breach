import pandas as pd
import numpy as np
import streamlit as st
import holidays


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_csv(uploaded_file) -> "pd.DataFrame | None":
    """
    Read an uploaded CSV, detect the datetime column, set it as the index,
    keep only numeric columns, and return the result.
    Shows st.error on failure and returns None.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

    dt_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            if parsed.notna().sum() > len(df) * 0.8:
                dt_col = col
                df[col] = parsed
                break
        except Exception:
            continue

    if dt_col is None:
        st.error(
            "No datetime column detected. "
            "Ensure one column contains timestamps (e.g. 'timestamp', 'time')."
        )
        return None

    df = df.set_index(dt_col).sort_index()
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        st.error("No numeric columns found. Check your CSV format.")
        return None

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline matching SLA model training:
      - Global z-scores per column
      - Rolling 24-hour mean / std / min / max
      - Lag features at 1h, 6h, 24h
      - Time features: hour, dayofweek, is_weekend, is_holiday (CZ)
    Drops rows that are fully NaN in all lag columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Z-scores
    for col in numeric_cols:
        m, s = df[col].mean(), df[col].std()
        df[f"{col}_zscore"] = (df[col] - m) / s if s > 0 else 0.0

    # Rolling 24h stats
    for col in numeric_cols:
        r = df[col].rolling(window=24, min_periods=1)
        df[f"{col}_roll24_mean"] = r.mean()
        df[f"{col}_roll24_std"]  = r.std().fillna(0)
        df[f"{col}_roll24_min"]  = r.min()
        df[f"{col}_roll24_max"]  = r.max()

    # Lag features
    for col in numeric_cols:
        for lag in [1, 6, 24]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Time features
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    try:
        cz = holidays.CZ(years=df.index.year.unique().tolist())
        df["is_holiday"] = df.index.normalize().isin(cz).astype(int)
    except Exception:
        df["is_holiday"] = 0

    lag_cols = [c for c in df.columns if "_lag" in c]
    df = df.dropna(subset=lag_cols, how="all")
    return df


def prepare_for_model(
    df_feat: pd.DataFrame,
    feature_cols: list,
    scaler,
) -> np.ndarray:
    """Select feature_cols, median-impute any NaNs, then scale."""
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(
            f"Missing features after engineering:\n{missing}\n\n"
            "Check that your CSV contains the same traffic columns used during training."
        )
    X = df_feat[feature_cols].copy()
    X = X.fillna(X.median())
    return scaler.transform(X)


# ── Session-state helpers ─────────────────────────────────────────────────────

def get_session_df() -> "pd.DataFrame | None":
    """Return the shared DataFrame stored in st.session_state, or None."""
    return st.session_state.get("df", None)
