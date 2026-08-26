"""Load sample panel or download Tushare HS300 daily bars."""

from __future__ import annotations

import os
import time

import pandas as pd
from dotenv import load_dotenv

from src.config import Paths, ROOT
from src.sample_data import save_sample


def _token() -> str:
    load_dotenv(ROOT / ".env")
    return os.getenv("TUSHARE_TOKEN", "").strip()


def load_or_download(cfg: dict, paths: Paths) -> pd.DataFrame:
    source = str(cfg.get("source", "sample")).lower()
    start, end = cfg["start_date"], cfg["end_date"]
    if source == "sample":
        save_sample(paths, start, end)
        daily = pd.read_parquet(paths.sample / "panel.parquet")
        daily.to_parquet(paths.raw / "daily.parquet", index=False)
        return daily
    if source == "tushare":
        return download_tushare(cfg, paths)
    raise ValueError(f"unknown source: {source}")


def download_tushare(cfg: dict, paths: Paths) -> pd.DataFrame:
    token = _token()
    if not token:
        raise RuntimeError("未找到 TUSHARE_TOKEN。请复制 .env.example 为 .env 并填写，或改用 source: sample")

    import tushare as ts

    pro = ts.pro_api(token)
    start = cfg["start_date"].replace("-", "")
    end = cfg["end_date"].replace("-", "")
    max_n = int(cfg.get("tushare_max_stocks", 40))

    # 用区间末成分近似股票池；存在幸存者偏差，见 docs/口径说明.md
    weights = pro.index_weight(index_code="000300.SH", start_date=end, end_date=end)
    if weights is None or weights.empty:
        weights = pro.index_weight(index_code="000300.SH")
    codes = sorted(weights["con_code"].dropna().unique().tolist())[:max_n]

    stocks = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,list_date")
    stocks["is_st"] = stocks["name"].str.contains("ST", na=False)
    stocks = stocks[stocks["ts_code"].isin(codes)].copy()

    daily_parts, adj_parts, basic_parts = [], [], []
    for i, code in enumerate(codes):
        d = pro.daily(ts_code=code, start_date=start, end_date=end)
        a = pro.adj_factor(ts_code=code, start_date=start, end_date=end)
        b = pro.daily_basic(ts_code=code, start_date=start, end_date=end, fields="ts_code,trade_date,pe_ttm,total_mv")
        if d is not None and not d.empty:
            daily_parts.append(d)
        if a is not None and not a.empty:
            adj_parts.append(a)
        if b is not None and not b.empty:
            basic_parts.append(b)
        if (i + 1) % 5 == 0:
            time.sleep(0.4)

    if not daily_parts:
        raise RuntimeError("Tushare 未返回日线，请检查积分权限与日期区间")

    daily = pd.concat(daily_parts, ignore_index=True)
    adj = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()
    basic = pd.concat(basic_parts, ignore_index=True) if basic_parts else pd.DataFrame()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    if not adj.empty:
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        daily = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    else:
        daily["adj_factor"] = 1.0
    if not basic.empty:
        basic["trade_date"] = pd.to_datetime(basic["trade_date"])
        daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    daily = daily.merge(stocks, on="ts_code", how="left")
    daily["suspend"] = daily["vol"].fillna(0) <= 0
    daily["limit"] = daily["pct_chg"].abs() >= 9.5
    daily["adj_factor"] = daily["adj_factor"].fillna(1.0)
    daily.to_parquet(paths.raw / "daily.parquet", index=False)
    stocks.to_parquet(paths.raw / "stocks.parquet", index=False)
    return daily
