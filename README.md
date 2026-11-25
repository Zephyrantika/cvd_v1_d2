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


## v2.3 更新（2026-11-20 晚）

补齐 G3 验收门（空模块 p 值 KS 检验与 A 级假阳性率）。

| 项 | 结果 |
|---|---|
| 空模块数量 | 9,100（91 疾病 × 100 次随机表型） |
| KS 检验 vs 均匀分布 | D=0.0099、p=0.332（不拒绝均匀） |
| 假阳性率 p≤0.01 | 1.0%（门限 ≤5%） |
| 假阳性率 p≤0.05 | 5.0%（与名义水平一致） |

含义：p_spec 阈值可按字面理解，A 级的第一个条件（p_spec≤0.01）有校准依据。六道验收门现在过了五道，未过的 G5 是目标函数里稳定性项冗余，已在限制中记录。

注：用 30 次抽样时 0.01 处假阳性为 0，属地板效应（1/31=0.032 下不到 0.01），故采用 100 次。


## v2.4 更新（2026-11-22）

补上文档预注册的第三个主指标 core Jaccard，并更正一个度量错误。

| 项 | 结果 |
|---|---|
| core Jaccard（τ=0.8，五折 452 疾病） | 基线 0.8466、R1 0.9125（303/452，p=3.6×10⁻²²）、精修 λ=1 0.9338（361/452，p=1.5×10⁻⁴¹） |
| 分折 R1 | 0.878 / 0.907 / 0.944 / 0.912 / 0.921，五折均高于基线 |

说明：此前一版报的「Jaccard 0.365 对 0.432，R1 更差」是误用了整模块 Jaccard——25 基因模块对 50–90 基因的重建模块，上限被规模差压到 0.3–0.4。文档定义的是 core Jaccard（先按 τ=0.8 取稳定核心，再比较核心的复现），该口径下 R1 显著领先。

至此文档预注册的三个主指标全部为 R1 领先：z_spec 0.858 对 0.722、GO 6.8×10⁻¹⁶ 对 1.0×10⁻⁹、core Jaccard 0.9125 对 0.8466。

另附一条负结果（E94）：用六特征逻辑回归做重排可将 top-200 召回从 35.6% 提到 85.4%，但消融显示几乎全部来自「基因 HPO 注释 vs 疾病表型」相似度，而该注释与标签同源（同一批文献策展），故不作为召回解决方案，仅作上界参照。

## v2.5 更新（2026-11-22）

三项增量迭代完成，均未改进主指标，主结论不变。

| 迭代 | 结果 | 处置 |
|---|---|---|
| 三目标选点（加表型一致性） | A 级 64.8% 对 R1 73.6%，特异性通过率 71.4% 对 80.2% | 不保留 |
| K=500 候选集 | A 级 12.1%（基线）/46.2%（精修），召回 37.8% 对 30.5% | 不保留 |
| 共享模块分析 | 高共享模块 A 率更高（0.47–0.80 对 0.33–0.67），但与 p_spec 负相关（ρ=−0.17~−0.41） | 保留为分析 |

结论：A 级优势来自「成员替换 + 特异性」，以特异性为代价的扩展（表型加权、放宽候选集）都会掉 A 级；共享模块只具描述价值。详见 docs/FIVE_FOLD_SUMMARY.md 第九节与实验日志 E96–E98。
## v2.6 更新（2026-11-22）

补上统计协议里的家族聚类估计，并写完融合说明。

| 项 | 结果 |
|---|---|
| 家族 cluster bootstrap（217 个家族，整簇重抽样 1,000 次） | A 级比例差 28.3 个百分点，95% CI [23.3, 33.7]，全部重抽样为正，双侧 p<0.001 |
| 融合接口说明 | 见 docs/INTEGRATION_SPEC.md |
| 方向二总描述 | 见 docs/DIRECTION2_OVERVIEW.md |

融合说明含模块表 15 个字段、三个接口（单模块打分、批量分级、重跑流程）、三条边界（分数只对候选集内基因有效、GWAS 需度数匹配口径、跨网络只能看社区富集）以及接入后的验收数字。
## v2.7 更新（2026-11-22）

关键连接分析补齐五折，并修正口径。

| 项 | 结果 |
|---|---|
| 删除模块内介数最高的 3 条边 | 最大连通分量下降 0.08–0.12（随机删 3 条为 0） |
| 五折胜出 | 29/39、31/40、31/40、25/40、30/40 |
| 检验 | p 从 2.3×10⁻⁶ 到 1.7×10⁻⁶ |

说明：原 39 号脚本用原始分均值差，量级太小被舍入成 0，结论无效。改为结构口径后五折一致，说明模块连通性依赖少数关键边。
## v2.8 更新（2026-11-25）

新增前瞻性验证与排序敏感性两项分析。

| 项 | 结果 |
|---|---|
| 前瞻性验证（2024+ ClinGen，五折 93 对） | 新条目 top-200 38.1%、模块召回 9.5%；旧条目 26.7%、6.7% |
| 排序敏感性（oracle 注入） | h=0 并集召回 30.5%（z 2.236）；h=0.5 48.9%；h=1.0 79.8%（z 2.108） |

前瞻性验证说明模块没有「只记住旧知识」；排序敏感性说明排序层若能覆盖全部已知基因，模块并集召回可到八成，z 只掉 0.13。两项均写入五折总结第十节与日志 E101–E102。