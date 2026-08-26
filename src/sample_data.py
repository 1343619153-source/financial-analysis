"""Generate a tiny A-share-like panel so the pipeline runs without Tushare."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Paths


INDUSTRIES = ["银行", "新能源", "消费", "制造"]


def build_sample_panel(start: str, end: str, n_stocks: int = 40, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    codes = [f"{600000 + i:06d}.SH" if i % 2 == 0 else f"{1 + i:06d}.SZ" for i in range(n_stocks)]
    industries = [INDUSTRIES[i % len(INDUSTRIES)] for i in range(n_stocks)]
    list_dates = pd.to_datetime(["2016-01-04"] * n_stocks)
    names = [f"样例{i:02d}" for i in range(n_stocks)]
    is_st = [False] * n_stocks
    is_st[-1] = True
    names[-1] = "ST样例"

    stocks = pd.DataFrame(
        {
            "ts_code": codes,
            "name": names,
            "industry": industries,
            "list_date": list_dates,
            "is_st": is_st,
        }
    )

    rows = []
    for i, code in enumerate(codes):
        n = len(dates)
        log_ret = rng.normal(0.0003, 0.018, size=n)
        close = 10 * np.exp(np.cumsum(log_ret))
        if industries[i] == "新能源":
            close *= np.linspace(1.0, 1.25, n)
        adj_factor = np.ones(n)
        split = n // 2
        close[split:] *= 0.5
        adj_factor[split:] = 2.0
        high = close * (1 + rng.uniform(0.0, 0.02, n))
        low = close * (1 - rng.uniform(0.0, 0.02, n))
        open_ = close * (1 + rng.normal(0, 0.005, n))
        volume = rng.integers(1_000_000, 8_000_000, n)
        suspend = np.zeros(n, dtype=bool)
        if i == 0:
            suspend[20:23] = True
        pct_chg = np.concatenate([[0.0], np.diff(close) / np.maximum(close[:-1], 1e-8)])
        limit = np.abs(pct_chg) > 0.095
        extra_limit = rng.choice(n, size=max(1, n // 25), replace=False)
        limit[extra_limit] = True
        pe_ttm = np.clip(rng.normal(18, 8, n), 3, 80)
        total_mv = rng.uniform(80, 800, n) * 1e4  # 万元, tushare-like
        for j, d in enumerate(dates):
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "open": open_[j],
                    "high": high[j],
                    "low": low[j],
                    "close": close[j],
                    "pct_chg": pct_chg[j] * 100,
                    "vol": float(volume[j]),
                    "adj_factor": adj_factor[j],
                    "pe_ttm": pe_ttm[j],
                    "total_mv": total_mv[j],
                    "suspend": suspend[j],
                    "limit": bool(limit[j]),
                }
            )

    daily = pd.DataFrame(rows)
    daily = daily.merge(stocks[["ts_code", "industry", "name", "list_date", "is_st"]], on="ts_code")
    calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
    return {"stocks": stocks, "calendar": calendar, "daily": daily}


def save_sample(paths: Paths, start: str, end: str) -> Path:
    panel = build_sample_panel(start, end)
    out = paths.sample / "panel.parquet"
    panel["daily"].to_parquet(out, index=False)
    panel["stocks"].to_parquet(paths.sample / "stocks.parquet", index=False)
    panel["calendar"].to_parquet(paths.sample / "calendar.parquet", index=False)
    return out
