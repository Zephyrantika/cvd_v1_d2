# 方向二递交包 v2（2026-11-19）

数据获取方式见 `docs/DATA_SOURCES.md`（各数据库地址、体量、处理脚本）。

## 这一版比 v1 改了什么

v1 的核心结论是"基线在疾病特异性上最强，九种方法无一超过；我们的贡献落在可靠性评价框架与复现率上"。v2 推翻了其中一半，原因是查出一个测量口径错误，并在此基础上把方法做实。

**问题一：A 级比例一直打不过基线。** v1 报基线 59.3%、我们的方法 46–52%。查下来是两把尺子：基线的 A 级用的是它自己的共现稳定性（通过率约 64%），而我们的候选用 20 次独立扰动复现率（通过率约 95%）。统一成「100 次随机表型零模型算 p + 20 次独立扰动算复现率」后，基线只有 22.6%。

**问题二：候选模块只做过加权和优化，z 与稳定性此消彼长。** v2 改用 Pareto 前沿：在 λ×μ×规模共 26 个候选里，按「复现率≥0.7 且 hub≤0.5」取 z 最大者。选点规则在看结果之前固定，对全部疾病一致。

**问题三：数据链太窄。** 标签从 585 对扩到 1,761 对（新增 DisGeNET 1,319、Open Targets 1,125、ClinGen 100、CTD direct 209），T1 疾病从 9 个增到 77 个；并补了 KEGG、WikiPathways、HPA、gene→phenotype 四层。

## 结果

| 指标 | 基线 | v2 方法 | 提升 |
|---|---|---|---|
| A 级比例（五折 452 疾病，同口径） | 22.6% | **50.9%** | 2.25 倍，p=2.3×10⁻²⁹ |
| A 级（公平基线：基线也可选尺寸） | 26.8% | 50.9% | 1.90 倍，p=2.2×10⁻²⁰ |
| p_spec≤0.01 通过率 | 31.6% | 56.0% | 1.77 倍，p=2.6×10⁻²⁰ |
| 样本外 z（交叉拟合） | 0.722 | 0.858 | +18.8%，p=1.1×10⁻¹⁸ |
| 复现率（共识模块） | 0.730 | 0.853 | +16.8%，五折全胜 |
| 模块密度 | 0.253 | 0.360 | +42.3%，p=4.4×10⁻²⁹ |
| hub 集中度 | 0.182 | 0.156 | −14.3%，p=2.4×10⁻²⁷ |
| 攻击容忍度（剩余连通） | 0.841 | 0.892 | +6.1%，p=5.2×10⁻⁶ |
| GWAS 支持（度数匹配） | 1.74–2.43 | 1.33–1.69 | 无显著差异（三折 R1 数值更高） |

## 稳健性

换扰动种子（fold 0）：复现率 0.853 对 0.683，A 级 70.1% 对 43.7%。
不截断基线（fold 0）：48.4%，R1 仍高 25 点。
阈值网格九组（p 0.01/0.02/0.05 × 复现率 0.6/0.7/0.8）：R1 均高 19–28 点。
hub 阈值 0.3/0.5/0.7：结论不变。

## 限制

1. z_spec 与模块构建同源，交叉拟合缓解但未消除。
2. A 级的稳定性一半受选点影响（选点用复现率），p_spec 那一半完全独立。
3. GWAS 支持在度数匹配零模型下无显著差异；原来的"下降"主要是度数偏差造成的。
4. 跨网络在社区富集口径下无显著差异；原来的"不可迁移"是 60% 重合阈值造成的。
5. 模块召回已知基因仍不足 10%，瓶颈在排序与标签密度。

## 文件

| 路径 | 内容 |
|---|---|
| `tables/t1_A_grade_five_fold.csv` | 五折 A 级对比（基线/公平基线/R1） |
| `tables/t2_main_metrics.csv` | 全部主指标与检验 |
| `tables/t3_robustness.csv` | 四项稳健性检查 |
| `tables/t4_external.csv` | 外部证据（含不利项） |
| `tables/t5_label_sources.csv` | 标签来源构成 |
| `docs/FIVE_FOLD_SUMMARY.md` | 五折总结（主文档） |
| `docs/MAINLINE_RESULTS.md` | 主线结果与更正记录 |
| `docs/EVAL_PROTOCOL.md` | 与 DREAM/PhEval 的评测口径对齐 |
| `docs/DATA_CHAIN_V1.md` | 数据链表结构与 QC |
| `docs/EXPERIMENT_LOG.md` | E1–E85 实验日志 |
| `docs/SOLUTION_BEAT_BASELINE.md` | 精修方法的设计与取舍 |
| `docs/LABEL_EXPANSION.md` | 标签扩充前后对比 |
| `docs/METHODS_COMPARISON.md` | 九种方法对照（含精确实现） |
| `docs/VERSIONING.md` | 版本管理约定 |
| `code/` | 65–96 号脚本（交叉拟合、选点、分级、攻击容忍、外部验证、数据链） |


## 数据来源与获取

所有外部数据均可公开获取，下面是本项目实际使用的地址（截至 2026-11-19 有效）。

| 数据 | 用途 | 地址 |
|---|---|---|
| HPO | 表型本体、疾病—表型、基因—表型 | https://purl.obolibrary.org/obo/hp/hpoa/ （phenotype.hpoa、genes_to_disease.txt、genes_to_phenotype.txt） |
| MONDO | 疾病本体与家族 | https://purl.obolibrary.org/obo/mondo.obo |
| STRING | 蛋白互作（400/700/900 三档） | https://string-db.org/cgi/download |
| BioGRID | 互作补充 | https://downloads.thebiogrid.org/BioGRID |
| Reactome | 通路 | https://reactome.org/download-data |
| GO | 功能富集 | http://geneontology.org/docs/download-ontology/ |
| GTEx | 心血管 5 组织表达 | https://gtexportal.org/home/downloads/adult-gtex |
| GWAS Catalog | 常见变异支持 | https://www.ebi.ac.uk/gwas/docs/file-downloads |
| ClinVar | 疾病—基因 | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ |
| ClinGen | 基因—疾病有效性分级 | https://search.clinicalgenome.org/kb/gene-validity |
| DisGeNET | curated 疾病—基因（需 API key） | https://www.disgenet.org/downloads ；接口 https://api.disgenet.com/api/v1/gda/summary |
| Open Targets | 疾病—靶点关联 | https://platform.opentargets.org/downloads ；接口 https://api.platform.opentargets.org/api/v4/graphql |
| CTD | 基因—疾病（direct/inferred） | https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz （3.1 GB） |
| HuRI | 实验验证互作组 | https://www.interactome-atlas.org/download |
| HPA | 组织表达 | https://www.proteinatlas.org/download/proteinatlas.tsv.zip |
| KEGG | 通路 | https://rest.kegg.jp/link/hsa/pathway 、https://rest.kegg.jp/list/pathway/hsa |
| WikiPathways | 通路 | https://data.wikipathways.org/current/gmt/ |
| HGNC | 基因 ID 映射 | https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt |
| HumanBase | 组织特异网络（心/主动脉/动脉） | https://s3-us-west-2.amazonaws.com/humanbase/networks/heart_top.gz 等 |

### CTD 与 HumanBase 的获取说明

CTD 原始文件 3.1 GB（`CTD_genes_diseases.tsv.gz`），超过 GitHub 单个附件 2 GB 上限，**未随包上传**，请从下面地址直接下载：

https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz

下载后用 `scripts/71_ctd_align.py` 解析：只保留 `DirectEvidence == "marker/mechanism"` 的行，疾病经 MeSH→MONDO 映射、基因经 HGNC→Entrez 映射，产出 `ctd_gene_disease.parquet`。该解析结果（209 条 direct、98,625 条 inferred）已在 Release 的 `direction2_v2_external_tables_2026-11-19.zip` 中，可直接使用，不必重下 3.1 GB。

### HumanBase 网络

三个组织网络体积较大（heart 250 MB、artery 385 MB、aorta 1.0 GB），未入包，按上表地址直接下载即可，文件名格式为 `{tissue}_top.gz`，三列：基因 A、基因 B、概率。


## v2.1 更新（2026-11-20）

补齐了三项外部效度检验，结论改写：

| 证据 | 结果 |
|---|---|
| 表型一致性（模块基因 HPO vs 疾病表型） | Jaccard 四折更好，三折显著（fold0 p=6.4e-5、fold1 p=7.2e-5、fold2 p=1.5e-4、fold3 p=6.6e-4） |
| GWAS（度数匹配零模型） | 五折无显著差异（fold0 1.30 对 1.11，p=0.80；fold2/3/4 R1 数值更高） |
| 跨网络（心脏网络社区富集） | res=1 0.055 对 0.068（p=0.71）、res=3 0.0074 对 0.0054（p=0.48），均无差异 |

外部效度由「一赢两输」变为「四赢两平」，没有一项显著更差。此前两项“输”均为测量口径问题：GWAS 用同大小随机集做对照会给 hub 密集的基线送分；跨网络用 60% 成员重合阈值在两个不同图上不可满足。


## v2.2 更新（2026-11-20 晚）

两项加固完成，均支持主结论。

| 检查 | 结果 |
|---|---|
| 换扰动种子铺五折（种子 999，未参与选点） | 复现率五折全胜：0.853/0.760/0.929/0.874/0.874 对 0.683/0.575/0.854/0.691/0.817，p 从 1.1×10⁻⁵ 到 0.042 |
| 换种子口径的 A 级 | 基线 21.0%、R1 48.4%（419 疾病），增 137 减 22，McNemar p=1.7×10⁻²¹ |
| ClinGen 高置信子集（Definitive/Strong，55 个疾病） | 基线 A 34.5%、R1 70.9%，增 20 减 0，p=1.9×10⁻⁶ |

含义：A 级的稳定性一半不是选点造成的（换一批扰动依然五折全胜）；在标签证据最强的疾病上优势更大且无一例变差（不是靠弱标签撑起来的）。

至此六个常见质疑均有对应检验：口径不一致、基线截断、选点自由、稳定性选点、弱标签、阈值挑选。
