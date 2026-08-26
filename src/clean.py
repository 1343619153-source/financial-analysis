"""Align calendar, build adjusted prices, apply sample filters."""

from __future__ import annotations

import pandas as pd


def prepare_panel(daily: pd.DataFrame, cfg: dict, use_adj: bool = True, exclude_limit: bool | None = None) -> pd.DataFrame:
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df = df.sort_values(["ts_code", "trade_date"])

    g = df.groupby("ts_code", group_keys=False)
    last_adj = g["adj_factor"].transform("last")
    if use_adj:
        df["close_used"] = df["close"] * df["adj_factor"] / last_adj
    else:
        df["close_used"] = df["close"]

    df["fwd_ret_1"] = g["close_used"].shift(-1) / df["close_used"] - 1
    df["fwd_ret_5"] = g["close_used"].shift(-5) / df["close_used"] - 1
    df["fwd_ret_20"] = g["close_used"].shift(-20) / df["close_used"] - 1

    df["list_days"] = (df["trade_date"] - df["list_date"]).dt.days
    min_days = int(cfg.get("min_list_days", 180))
    filters = cfg.get("filters", {})
    drop_st = bool(filters.get("exclude_st", True))
    drop_suspend = bool(filters.get("exclude_suspend", True))
    drop_limit = bool(filters.get("exclude_limit", True) if exclude_limit is None else exclude_limit)

    mask = df["list_days"] >= min_days
    if drop_st:
        mask &= ~df["is_st"].fillna(False)
    if drop_suspend:
        mask &= ~df["suspend"].fillna(False)
    if drop_limit:
        mask &= ~df["limit"].fillna(False)

    df["in_sample"] = mask
    return df
