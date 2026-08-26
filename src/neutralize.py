"""Cross-sectional winsorize, z-score, industry and size neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = panel.copy()
    for col in cols:
        df[f"{col}_raw"] = df[col]
        df[f"{col}_cs"] = df.groupby("trade_date")[col].transform(_cs_winsor_z)
        df[f"{col}_n"] = np.nan

    parts = []
    for _, g in df.groupby("trade_date", sort=False):
        g = g.copy()
        for col in cols:
            g[f"{col}_n"] = _industry_size_residual(g, f"{col}_cs")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _cs_winsor_z(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        z = x - x.mean()
        sd = z.std()
        return z / sd if sd and sd > 0 else z
    lo, hi = med - 3 * 1.4826 * mad, med + 3 * 1.4826 * mad
    x = x.clip(lo, hi)
    sd = x.std()
    return (x - x.mean()) / sd if sd and sd > 0 else x * 0


def _industry_size_residual(g: pd.DataFrame, col: str) -> pd.Series:
    y = pd.to_numeric(g[col], errors="coerce")
    out = pd.Series(np.nan, index=g.index, dtype=float)
    valid = y.notna()
    if valid.sum() < 8:
        out.loc[valid] = y.loc[valid] - y.loc[valid].mean()
        return out
    ind = pd.get_dummies(g.loc[valid, "industry"], dummy_na=False, dtype=float)
    size = pd.to_numeric(g["total_mv"], errors="coerce") if "total_mv" in g.columns else pd.Series(np.nan, index=g.index)
    if size.notna().mean() < 0.3 and "amount" in g.columns:
        size = pd.to_numeric(g["amount"], errors="coerce")
    size = np.log(size.clip(lower=1)).loc[valid]
    x = pd.concat([ind, size.rename("log_mv")], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    yy = y.reindex(x.index)
    if len(yy) < 8:
        out.loc[valid] = y.loc[valid] - y.loc[valid].mean()
        return out
    beta, *_ = np.linalg.lstsq(x.to_numpy(), yy.to_numpy(), rcond=None)
    out.loc[x.index] = yy.to_numpy() - x.to_numpy() @ beta
    return out
