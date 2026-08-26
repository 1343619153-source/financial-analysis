"""Write Markdown + charts for the research desk delivery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in df.itertuples(index=False):
        body.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    reports_dir: Path,
    summaries: dict[str, dict[str, pd.DataFrame]],
    notes: list[str],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A 股投研底表与单因子评价报告",
        "",
        "本报告面向研究组，用于判断因子是否进入备选，不构成交易策略或收益承诺。",
        "",
        "## 口径摘要",
        "",
    ]
    lines.extend(f"- {n}" for n in notes)
    lines += ["", "## 评价结果", ""]

    for variant, payload in summaries.items():
        summary = payload["summary"]
        lines += [f"### 方案 `{variant}`", ""]
        if summary is None or summary.empty:
            lines += ["无有效评价日（样本过少或因子缺失）。", ""]
            continue
        lines.append(_md_table(summary.round(4)))
        lines.append("")
        _plot_ic(payload["ic"], reports_dir / f"ic_{variant}.png")
        _plot_quintile(payload["quintile"], reports_dir / f"quintile_{variant}.png")
        lines += [
            f"![Rank IC](ic_{variant}.png)",
            "",
            f"![五分组收益](quintile_{variant}.png)",
            "",
        ]

    lines += [
        "## 如何阅读",
        "",
        "- Rank IC：当日因子值与未来收益的 Spearman 相关，均值接近 0 表示区分度弱。",
        "- Q5-Q1：最高组减最低组的平均未来收益，用于看分层是否单调。",
        "- 对照 `baseline` / `no_adj` / `no_limit_filter`：同一因子在不同口径下的评价差异，是数据岗要讲清的点。",
        "",
    ]
    out = reports_dir / "factor_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _plot_ic(ic: pd.DataFrame, path: Path) -> None:
    if ic is None or ic.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for h, part in ic.groupby("horizon"):
        part = part.sort_values("trade_date")
        ax.plot(part["trade_date"], part["rank_ic"].cumsum(), label=f"{h}d")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("累计 Rank IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_quintile(q: pd.DataFrame, path: Path) -> None:
    if q is None or q.empty:
        return
    sub = q[q["horizon"] == q["horizon"].min()]
    means = sub.groupby("quantile")["mean_ret"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index.astype(str), means.values)
    ax.set_xlabel("分位（1=低因子值）")
    ax.set_ylabel("平均未来收益")
    ax.set_title("五分组平均收益（最短持有期）")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
