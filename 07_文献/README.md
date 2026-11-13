# 方向二文献包（V1 为导向）

> 2026-10-28 口径修订：关联数改用唯一 (disease, gene) 对 **568**（原始记录 688 行）；分层 T1=9、T2=17、T3=426、T4=18。复算见交接包根目录的 `CORRECTIONS.md`。

论文以方向二为中心：表型驱动网络上的**可靠机制模块发现与评价**。V1 是导向案例、数据与评价基础设施，不是论文的算法贡献。只收同行评议期刊论文，没有 arXiv、bioRxiv 和会议论文。

## 论文定位

中心问题：现有「传播 + 社区发现」模块里，哪些是疾病特异的，哪些是度数/通用模块；怎样用零模型、稳定性和外部证据判断模块可靠。

V1 的角色：

- 提供 452 个疾病、16,992 个候选基因、5 折划分、RWR/校正分数、Leiden 模块和泄漏审计。
- 提供 GNN 分支负结果（五折 MRR 0.0217 对 RWR 0.0550）和数据审计（568 对关联、原始 688 行、94.2% 疾病只有 1–2 个正样本），说明为什么主指标不能只靠基因恢复。
- 不作为论文主模型，不写「HGCN 提升模块」。

论文方案见 `09_论文方案_方向二为中心.md`；方法、零模型、输出 schema 和验收门见 `../方向二_技术文档.md`；V1-A 技术细节见 `CardioMechanism_Model/docs/v1_technical_spec.md`。

## 收录范围

28 篇，全部期刊论文：

**方向二核心 23 篇**

- 模块方法与零模型：Menche、Goh、DIAMOnD、Heinz、HotNet2、ROBUST、ROBUST-Web、DOMINO、Leiden、共识聚类、NeDRex、GNN4DM、CGMega。
- 评价与偏差：DREAM、CoLiPE、Lazareva、Maslov、Fortunato & Hric、Fortunato & Barthélemy。
- 心血管外部证据：Lee 2021、Koplev 2022、Zhu 2024、Hecker 2026。

**V1 导向 5 篇**

- RWR、Phenolyzer、Köhler 2009、PU 综述、数据泄漏。

**GNN 后续阅读（不计入 28 篇）**

- Speos、XGDAG、BioPathNet、两篇图表示学习综述、HGTDR。只在讨论「以后要不要上图学习」时引，见 `01_V1核心方法.md`。

R-GCN（ESWC 2018）、HGT（WWW 2020）、GCN（ICLR 2017）是会议论文，不在计数里。

Scientific Reports 的两篇（Leiden、共识聚类）是算法原文，Nature Portfolio；如果导师只认 CNS 正刊和子刊，把这两篇降为引用，不占必读。其余论文都在 Nature/Science/Cell 系、PNAS、Nature Reviews、AJHG、NAR、Nature Methods/Genetics/Communications、Nature Biomedical Engineering、Nature Cardiovascular Research、Cardiovascular Research、Briefings in Bioinformatics、Bioinformatics、PLOS Computational Biology 这一档。

## 目录

```text
literature_direction2/
├── README.md
├── 00_速览.md              28 篇一页清单 + 方向二组件映射
├── 01_V1核心方法.md        GNN 后续阅读（不计入 28 篇）
├── 02_表型与基线.md        RWR、Phenolyzer、Exomiser、语义相似度
├── 03_评价与泄漏.md        PU、DREAM、泄漏、度数偏差
├── 04_方向二模块.md        共享/特异、DIAMOnD、ROBUST、零模型
├── 05_心血管证据.md        CVD 网络医学、STARNET、血管 PPI、CAD 基因集
├── 06_结合方案.md          方向二为中心、V1-A 为输入的实验和决策门
├── 07_数据工具与复现.md    V1-A 产物、环境、命令、现状
├── 08_方向二最新进展与融合.md  2022–2026 最新方法、能否融合、优先级
├── 09_论文方案_方向二为中心.md  论文题目、RQ、贡献、结构、图表、写作顺序
├── papers.csv              机器可读清单
├── refs.bib                BibTeX，卷期页码按 Crossref 核对
└── pdfs/                   开放获取全文，其余给 DOI
```

## 怎么读

**先读方向二。** `04`（模块与零模型）→ `03`（评价/泄漏/偏差）→ `05`（心血管外部证据）→ `08`（最新方法）→ `09`（论文方案）。这五份是论文主线。

**V1 只用来看数据与案例。** `02`（传播与表型）、`07`（V1-A 产物与负结果）、`06`（方向二实验）。不要把 V1 当成论文模型。

**写论文时。** 按 `09` 的章节和图表组织；`01` 只在 Discussion 的「未来图学习」里引。

**找最新方法。** 读 `08`。ROBUST、ROBUST-Web、DOMINO 已经并入核心 23 篇，不再单列。

## 口径

- 论文中心是方向二；V1 是导向案例，不是主模型。
- V1-A 的模型是 RWR + 校正；GNN 分支是负结果，不写成「GNN 有效」。
- 基因恢复指标只在 T1/T2 报告；T3 只报不依赖正样本的 z_spec、稳定性、GO。
- Reactome 只作图内证据；GO/GWAS/Zhu/Hecker/Koplev 作外部证据。
- 不写因果。
- `01` 和 GNN 论文不计入 28 篇。

最后核对日期：2026-10-22。
