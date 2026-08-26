"""IC, quantile spreads, and a simple turnover diagnostic for research delivery."""

from __future__ import annotations

import pandas as pd


FWD = {1: "fwd_ret_1", 5: "fwd_ret_5", 20: "fwd_ret_20"}


def evaluate_factor(panel: pd.DataFrame, factor_col: str, cfg: dict) -> dict[str, pd.DataFrame]:
    qn = int(cfg.get("evaluate", {}).get("quantiles", 5))
    horizons = list(cfg.get("evaluate", {}).get("forward_days", [1, 5, 20]))
    df = panel.loc[panel["in_sample"]].copy()
    df = df.dropna(subset=[factor_col])

    ic_rows = []
    q_rows = []
    to_rows = []
    prev_q = None
    for dt, g in df.groupby("trade_date"):
        if g[factor_col].nunique() < qn:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g[factor_col], qn, labels=False, duplicates="drop")
        if g["q"].nunique() < qn:
            continue
        for h in horizons:
            ret_col = FWD[int(h)]
            sub = g.dropna(subset=[ret_col])
            if len(sub) < 20:
                continue
            ic = sub[factor_col].rank().corr(sub[ret_col].rank())
            ic_rows.append({"trade_date": dt, "horizon": h, "rank_ic": ic, "n": len(sub)})
            stats = sub.groupby("q")[ret_col].mean()
            for q, val in stats.items():
                q_rows.append({"trade_date": dt, "horizon": h, "quantile": int(q) + 1, "mean_ret": val})
        if prev_q is not None:
            both = g[["ts_code", "q"]].merge(prev_q, on="ts_code", suffixes=("", "_lag"))
            if len(both):
                to_rows.append(
                    {
                        "trade_date": dt,
                        "turnover": float((both["q"] != both["q_lag"]).mean()),
                    }
                )
        prev_q = g[["ts_code", "q"]]

    ic = pd.DataFrame(ic_rows)
    quintile = pd.DataFrame(q_rows)
    turnover = pd.DataFrame(to_rows)
    summary = _summarize(ic, quintile, turnover, qn)
    return {"ic": ic, "quintile": quintile, "turnover": turnover, "summary": summary}


def _summarize(ic: pd.DataFrame, quintile: pd.DataFrame, turnover: pd.DataFrame, qn: int) -> pd.DataFrame:
    rows = []
    if ic.empty:
        return pd.DataFrame()
    for h, part in ic.groupby("horizon"):
        mean_ic = part["rank_ic"].mean()
        ir = mean_ic / part["rank_ic"].std() if part["rank_ic"].std() else float("nan")
        q = quintile[quintile["horizon"] == h]
        spread = float("nan")
        if not q.empty:
            hi = q.loc[q["quantile"] == qn, "mean_ret"].mean()
            lo = q.loc[q["quantile"] == 1, "mean_ret"].mean()
            spread = hi - lo
        rows.append(
            {
                "horizon": h,
                "rank_ic_mean": mean_ic,
                "rank_ic_ir": ir,
                "q5_minus_q1": spread,
                "turnover_mean": turnover["turnover"].mean() if not turnover.empty else float("nan"),
                "n_days": part["trade_date"].nunique(),
            }
        )
    return pd.DataFrame(rows)
