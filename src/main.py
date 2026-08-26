"""Run the research-data pipeline: ingest -> clean -> factors -> evaluate -> report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.clean import prepare_panel
from src.config import Paths, load_config
from src.download import load_or_download
from src.evaluate import evaluate_factor
from src.factors import add_factors
from src.neutralize import neutralize
from src.report import write_report

FACTORS = ["mom", "ep"]


def run(config_path: str | None = None) -> None:
    cfg = load_config(Path(config_path) if config_path else None)
    paths = Paths.from_root()
    daily = load_or_download(cfg, paths)
    notes = [
        f"数据源：{cfg.get('source')}",
        f"区间：{cfg['start_date']} ~ {cfg['end_date']}",
        "动量：过去 20 日收益，去掉最近 5 日",
        "EP：1 / pe_ttm（pe_ttm<=0 视为缺失）",
        "中性化：截面 MAD 去极值 + Z-Score，再对行业哑变量与 log(市值) 回归取残差",
        "评价对象：中性化后的 mom_n / ep_n，仅样本内交易日",
    ]

    ablations = cfg.get("ablation") or [{"name": "baseline", "use_adj": True, "exclude_limit": True}]
    summaries: dict = {}
    for spec in ablations:
        name = spec["name"]
        panel = prepare_panel(
            daily,
            cfg,
            use_adj=bool(spec.get("use_adj", True)),
            exclude_limit=bool(spec.get("exclude_limit", True)),
        )
        panel = add_factors(panel, cfg)
        panel = neutralize(panel, FACTORS)
        panel.to_parquet(paths.processed / f"panel_{name}.parquet", index=False)

        tables = []
        first_ev = None
        for fac in FACTORS:
            ev = evaluate_factor(panel, f"{fac}_n", cfg)
            if ev["summary"] is not None and not ev["summary"].empty:
                ev["summary"].insert(0, "factor", fac)
            tables.append(ev["summary"])
            if fac == "mom":
                first_ev = ev
        if first_ev is None:
            first_ev = {"ic": pd.DataFrame(), "quintile": pd.DataFrame(), "summary": pd.DataFrame()}
        first_ev["summary"] = pd.concat([t for t in tables if t is not None and not t.empty], ignore_index=True)
        summaries[name] = first_ev
        notes.append(
            f"`{name}`：use_adj={spec.get('use_adj')}，exclude_limit={spec.get('exclude_limit')}，"
            f"样本行数={int(panel['in_sample'].sum())}"
        )

    report = write_report(paths.reports, summaries, notes)
    print(f"完成。报告：{report}")
    print(f"加工数据：{paths.processed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A 股投研底表与单因子评价")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
