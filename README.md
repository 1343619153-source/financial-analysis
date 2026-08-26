# A 股投研底表与单因子评价

给量化 / 投研数据实习准备的本地项目：把行情做成可追溯的日频底表，计算 **动量** 和 **EP** 两个因子，输出 IC 与五分组报告。面向研究组判断因子是否备选，**不是交易策略**。

当前默认用内置样例数据，不需要 Tushare token，克隆后即可跑通。GitHub 上传以后再说。

## 能做什么

1. **数据对接**：样例面板，或 Tushare 沪深300 日线 / 复权因子 / 估值
2. **口径**：复权、ST、停牌、涨跌停、上市不满 180 日
3. **特征**：20 日动量（去掉最近 5 日）、EP = 1 / pe_ttm；截面去极值、标准化、行业 + 市值中性化
4. **交付**：`reports/factor_report.md`，并对照「不复权 / 不过滤涨跌停」评价是否变化

## 环境

需要 Python 3.10+。在本目录执行：

```powershell
cd "D:\Financial analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.main
```

跑完后看：

- `reports/factor_report.md`
- `data/processed/panel_baseline.parquet`

## 换成真实 A 股数据

1. 复制 `.env.example` 为 `.env`，填入 `TUSHARE_TOKEN`
2. 把 `config.yaml` 里 `source: sample` 改成 `source: tushare`
3. 再运行 `python -m src.main`

Tushare 按股票循环拉取，默认只取沪深300 前 40 只（`tushare_max_stocks`），避免第一次请求过多。成分用区间末日截面，有幸存者偏差，见 `docs/口径说明.md`。

## 目录

```
src/download.py      拉数 / 样例
src/clean.py         复权与样本过滤
src/factors.py       动量、EP
src/neutralize.py    去极值 / 标准化 / 中性化
src/evaluate.py      Rank IC、五分组、换手
src/report.py        Markdown 报告
src/main.py          入口
docs/口径说明.md
```

## 简历表述（示意）

构建 A 股日频投研底表（复权、停牌、涨跌停、上市天数）；计算跳过近 5 日的 20 日动量与 EP，并做行业市值中性化；输出 Rank IC、五分组与换手，对照复权 / 涨跌停过滤前后的评价差异，供研究侧判断因子是否备选。
