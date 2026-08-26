"""Two research factors: skip-5 momentum and earnings yield."""

from __future__ import annotations

import pandas as pd


def add_factors(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = panel.copy()
    fcfg = cfg.get("factors", {})
    window = int(fcfg.get("momentum_window", 20))
    skip = int(fcfg.get("momentum_skip", 5))

    px = df.groupby("ts_code")["close_used"]
    df["mom"] = px.shift(skip) / px.shift(skip + window) - 1

    pe = pd.to_numeric(df["pe_ttm"], errors="coerce")
    df["ep"] = pe.where(pe > 0).rdiv(1)
    return df
