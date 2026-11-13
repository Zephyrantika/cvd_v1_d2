# 方向二交接包

给另一台机器上的 agent 用的完整工作包。解压后先读 `HANDOFF.md`，再读 `AGENTS.md`。

## 里面有什么

| 路径 | 内容 | 能不能改 |
|---|---|---|
| `HANDOFF.md` | 交接文档：项目背景、口径、里程碑、坑、交付物 | 可补充 |
| `AGENTS.md` | 给 agent 的规则：铁律、口径、第一个任务 | 可补充 |
| `CORRECTIONS.md` | 与旧文档不一致的两处口径（T1=9 不是 6，568 不是 688） | 只读 |
| `manifest.json` | 每个文件的 sha256 与大小 | 重新生成 |
| `direction2/` | 新项目：M0 审计脚本、配置、产物、测试 | **随便改** |
| `v0/` | 基线：源码、清洗后的数据、图、RWR 排名、模块、指标 | **只读** |
| `docs/` | 项目档案 01–08、方向二说明、技术文档、工作流程 | 可补充 |
| `literature/` | 文献包：摘要、papers.csv、refs.bib、pdfs | 只读 |

包里没有 `v0/data/raw/`（1.5 GB 原始数据库）。方向二不需要它，图、折、分数都已经算好。将来做数据扩增或重构图时再按 `docs/02_v0_baseline.md` 第 2 节的版本清单下载。

## 三步开始

```bash
# 1. 环境
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ./v0
pip install -e ./direction2

# 2. 确认 M0 可复现
cd direction2
python scripts/00a_audit_data.py --v0-root ../v0 --config configs/tiers.yaml --out outputs/audit
pytest -q tests

# 3. 按 HANDOFF 第 8 节做第一个工作周
```

Windows 下把 `source .venv/bin/activate` 换成 `.venv\Scripts\activate`，其余命令相同。

## 校验完整性

`manifest.json` 里每个文件都有 sha256。抽查任意一个：

```bash
sha256sum v0/outputs/modules/modules.parquet
```

对不上说明传输损坏，重点检查 `v0/outputs/rankings/`（577 MB）和 `v0/data/graphs/`。

## 规模

解压后约 691 MB，主要是 `v0/outputs/rankings/`（577 MB）和 `v0/data/processed/`（41 MB）、`v0/data/graphs/`（37 MB）。
