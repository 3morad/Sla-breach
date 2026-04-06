"""
Lightweight FastAPI client shared across all model pages.

Usage
-----
from utils.api_client import call_endpoint, df_to_payload

result = call_endpoint("http://localhost:8000/predict/sla", df_to_payload(df))
if result is None:
    # server not running – fall back to local inference
    ...
else:
    predictions = result["predictions"]
"""

from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def call_endpoint(
    url: str,
    payload: dict,
    timeout: int = 30,
) -> dict | None:
    """
    POST *payload* (JSON-serialisable dict) to *url*.

    Returns:
        dict  – parsed JSON response on success
        None  – if the server is unreachable (ConnectionError / timeout)

    Raises:
        RuntimeError – for HTTP 4xx / 5xx or other unexpected errors
    """
    if not _HAS_REQUESTS:
        raise RuntimeError(
            "`requests` is not installed. Add it to requirements.txt."
        )

    try:
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except _requests.exceptions.ConnectionError:
        return None          # server not running — caller falls back locally
    except _requests.exceptions.Timeout:
        return None
    except _requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"API returned {resp.status_code}: {resp.text}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"API call failed: {exc}") from exc


def df_to_payload(df: pd.DataFrame) -> dict:
    """
    Serialise a DataFrame to a JSON-safe dict suitable for POST requests.

    Schema
    ------
    {
        "index":   ["2024-01-01 00:00", ...],
        "columns": ["bytes_in", "bytes_out", ...],
        "data":    [[...], [...], ...]
    }
    """
    return {
        "index":   [str(i) for i in df.index],
        "columns": df.columns.tolist(),
        "data":    df.where(pd.notnull(df), None).values.tolist(),
    }


def payload_to_df(payload: dict) -> pd.DataFrame:
    """Inverse of df_to_payload — reconstruct a DataFrame from the dict."""
    df = pd.DataFrame(
        payload["data"],
        columns=payload["columns"],
        index=pd.to_datetime(payload["index"]),
    )
    df.index.name = "timestamp"
    return df
