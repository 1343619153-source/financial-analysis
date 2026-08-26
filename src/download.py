"""Load cached panel, sample data, BaoStock A-share bars, or Tushare."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.config import Paths, ROOT
from src.sample_data import save_sample


def _token() -> str:
    load_dotenv(ROOT / ".env")
    return os.getenv("TUSHARE_TOKEN", "").strip()


def _meta_path(paths: Paths) -> Any:
    return paths.raw / "meta.json"


def _cache_ok(cfg: dict, paths: Paths) -> bool:
    if not bool(cfg.get("use_cache", True)):
        return False
    daily_path = paths.raw / "daily.parquet"
    meta_path = _meta_path(paths)
    if not daily_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys = ("source", "start_date", "end_date", "max_stocks")
    return all(str(meta.get(k)) == str(cfg.get(k)) for k in keys)


def _write_meta(cfg: dict, paths: Paths, extra: dict) -> None:
    payload = {
        "source": cfg.get("source"),
        "start_date": cfg.get("start_date"),
        "end_date": cfg.get("end_date"),
        "max_stocks": cfg.get("max_stocks") or cfg.get("tushare_max_stocks") or cfg.get("akshare_max_stocks"),
        **extra,
    }
    _meta_path(paths).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_download(cfg: dict, paths: Paths) -> pd.DataFrame:
    source = str(cfg.get("source", "sample")).lower()
    if source != "sample" and _cache_ok(cfg, paths):
        print(f"使用缓存：{paths.raw / 'daily.parquet'}")
        return pd.read_parquet(paths.raw / "daily.parquet")
    if source == "sample":
        start, end = cfg["start_date"], cfg["end_date"]
        save_sample(paths, start, end)
        daily = pd.read_parquet(paths.sample / "panel.parquet")
        daily.to_parquet(paths.raw / "daily.parquet", index=False)
        _write_meta(cfg, paths, {"n_stocks": int(daily["ts_code"].nunique())})
        return daily
    if source == "baostock":
        return download_baostock(cfg, paths)
    if source in {"akshare", "tushare"}:
        return download_tushare(cfg, paths)
    raise ValueError(f"unknown source: {source}")


def _bs_df(rs) -> pd.DataFrame:
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    cols = list(rs.fields) if getattr(rs, "fields", None) else None
    return pd.DataFrame(rows, columns=cols)


def _to_ts_code(code: str) -> str:
    raw = str(code).replace("sh.", "").replace("sz.", "")
    raw = raw.zfill(6)
    if str(code).startswith("sh.") or raw.startswith(("6", "9")):
        return f"{raw}.SH"
    return f"{raw}.SZ"


def _limit_threshold(ts_code: str) -> float:
    num = ts_code.split(".")[0]
    if num.startswith(("300", "301", "688", "689")):
        return 19.5
    return 9.5


def download_baostock(cfg: dict, paths: Paths) -> pd.DataFrame:
    import baostock as bs

    start, end = cfg["start_date"], cfg["end_date"]
    max_n = int(cfg.get("max_stocks") or cfg.get("tushare_max_stocks") or 50)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败：{lg.error_msg}")
    try:
        cons = _bs_df(bs.query_hs300_stocks())
        if cons.empty:
            raise RuntimeError("BaoStock 未返回沪深300成分")
        codes = cons["code"].dropna().astype(str).tolist()[:max_n]
        industry_map: dict[str, str] = {}
        list_map: dict[str, str] = {}
        name_map: dict[str, str] = dict(zip(cons["code"], cons["code_name"]))
        parts = []
        for i, code in enumerate(codes, start=1):
            print(f"拉取 {i}/{len(codes)} {code} {name_map.get(code, '')}")
            unadj = _bs_df(
                bs.query_history_k_data_plus(
                    code,
                    "date,code,open,high,low,close,volume,amount,tradestatus,pctChg,peTTM,isST",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="3",
                )
            )
            hfq = _bs_df(
                bs.query_history_k_data_plus(
                    code,
                    "date,close",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="2",
                )
            )
            if unadj.empty:
                continue
            if not hfq.empty:
                hfq = hfq.rename(columns={"close": "close_hfq"})
                unadj = unadj.merge(hfq[["date", "close_hfq"]], on="date", how="left")
            else:
                unadj["close_hfq"] = pd.NA
            ind = _bs_df(bs.query_stock_industry(code=code))
            if not ind.empty:
                industry_map[code] = str(ind.iloc[-1].get("industry") or "未知")
            basic = _bs_df(bs.query_stock_basic(code=code))
            if not basic.empty:
                list_map[code] = str(basic.iloc[0].get("ipoDate") or "")
            parts.append(unadj)
            time.sleep(0.05)
        if not parts:
            raise RuntimeError("BaoStock 未返回日线")
    finally:
        bs.logout()

    daily = pd.concat(parts, ignore_index=True)
    num_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg", "peTTM", "close_hfq"]
    for c in num_cols:
        if c in daily.columns:
            daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily["trade_date"] = pd.to_datetime(daily["date"])
    daily["ts_code"] = daily["code"].map(_to_ts_code)
    daily["adj_factor"] = daily["close_hfq"] / daily["close"]
    daily.loc[~daily["close"].gt(0), "adj_factor"] = 1.0
    daily["adj_factor"] = daily["adj_factor"].fillna(1.0)
    daily["pe_ttm"] = daily["peTTM"]
    daily["vol"] = daily["volume"]
    daily["pct_chg"] = daily["pctChg"]
    daily["suspend"] = daily["tradestatus"].astype(str) != "1"
    daily["is_st"] = daily["isST"].astype(str) == "1"
    daily["name"] = daily["code"].map(name_map)
    daily["industry"] = daily["code"].map(industry_map).fillna("未知")
    daily["list_date"] = pd.to_datetime(daily["code"].map(list_map), errors="coerce")
    daily["limit"] = daily.apply(
        lambda r: abs(float(r["pct_chg"])) >= _limit_threshold(str(r["ts_code"]))
        if pd.notna(r["pct_chg"])
        else False,
        axis=1,
    )
    daily["total_mv"] = pd.NA
    keep = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "list_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "pct_chg",
        "adj_factor",
        "pe_ttm",
        "total_mv",
        "suspend",
        "is_st",
        "limit",
    ]
    out = daily[keep].sort_values(["ts_code", "trade_date"])
    out.to_parquet(paths.raw / "daily.parquet", index=False)
    _write_meta(
        cfg,
        paths,
        {
            "n_stocks": int(out["ts_code"].nunique()),
            "n_rows": int(len(out)),
            "universe": "沪深300最新成分（存在幸存者偏差）",
        },
    )
    print(f"已保存 {out['ts_code'].nunique()} 只股票、{len(out)} 行 -> {paths.raw / 'daily.parquet'}")
    return out


def download_tushare(cfg: dict, paths: Paths) -> pd.DataFrame:
    token = _token()
    if not token:
        raise RuntimeError("未找到 TUSHARE_TOKEN。请复制 .env.example 为 .env 并填写，或改用 source: baostock")

    import tushare as ts

    pro = ts.pro_api(token)
    start = cfg["start_date"].replace("-", "")
    end = cfg["end_date"].replace("-", "")
    max_n = int(cfg.get("max_stocks") or cfg.get("tushare_max_stocks", 40))

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
    if "amount" not in daily.columns:
        daily["amount"] = pd.NA
    daily.to_parquet(paths.raw / "daily.parquet", index=False)
    _write_meta(cfg, paths, {"n_stocks": int(daily["ts_code"].nunique())})
    return daily
