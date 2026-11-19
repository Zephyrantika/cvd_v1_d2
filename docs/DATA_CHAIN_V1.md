# 数据链 v1（2026-11-16）

把现有全部来源清洗、对齐、拼成一套带 QC 与校验和的事实表。目录 `phenotype-modules-v0/data/chain/`。

## 一、表清单

| 表 | 行数 | 说明 |
|---|---|---|
| dim_disease | 452 | 疾病、家族、已知基因数、表型数 |
| dim_gene | 16,580 | Entrez、symbol、Ensembl、是否在候选集 |
| dim_phenotype | 19,894 | HPO 术语与 IC |
| dim_pathway | 3,543 | Reactome / KEGG / WikiPathways 合并 |
| fact_disease_phenotype | 11,532 | 疾病—表型 |
| fact_disease_gene | 4,238 行 / 1,761 对 | 原始 568、HPO 455、ClinVar 462、ClinGen 100、DisGeNET 1,319（curated）、Open Targets 1,125（score≥0.1）、CTD 209（仅 marker/mechanism），带来源与分数 |
| fact_gene_phenotype | 264,510 | 基因—HPO |
| fact_gene_pathway | 182,054 | Reactome 104,078 / WikiPathways 39,373 / KEGG 38,603 |
| fact_gene_tissue | 76,153 | GTEx 5 个心血管组织的中位 TPM |
| fact_gene_gene | — | STRING 232,869 / BioGRID 865,627 / HuRI 38,712 |

每张表带 sha256，见 `chain_manifest.json`；QC 汇总在 `chain_qc.json`。

## 二、清洗规则

1. 基因统一 Entrez 字符串；符号与 Ensembl 经 HGNC 映射，映射失败的不入表（候选集 16,992 中有 412 个无 HGNC 记录）。
2. 疾病统一 MONDO；OMIM 经 mondo.obo 的 xref 转 MONDO。
3. 表型统一 HPO；IC 取 fold 0 训练折的计算值。
4. 疾病—基因按 (disease, gene, source) 去重，保留来源标记，不做跨源合并取并集。
5. 基因—基因无向边按 (min, max, source) 去重，去掉自环，两端都必须在候选集内。
6. 基因—通路按来源加前缀（reactome:/kegg:/wikipathways:），不做跨库合并。
7. 只保留 452 个疾病与 16,992 个候选基因范围内的事实；范围外的记录计数后丢弃。

## 三、QC 结论

- 疾病—基因：1,755 个唯一对（原 585 的 3 倍），其中 DisGeNET 439 个新对、Open Targets 724 个、ClinGen 15 个；只有 1–2 个基因的疾病从 426 降到 316，T1（≥5 个）从 9 增到 77。CTD 的 98,625 条推断关联（中位排名 6,867）未入链，只保留 209 条 direct。
- 表型侧正常：452 个疾病全部有表型，平均 25.5 个术语。
- 通路侧充足：3,543 条通路、18.2 万条基因—通路关系，覆盖 8,400 多个候选基因。
- 基因—表型层是新补的：26.5 万对，覆盖 5,060 个候选基因与 404/416 个已知基因。
- 网络侧三套：STRING 23.3 万条、BioGRID 86.6 万条、HuRI 3.9 万条，后两者只用于敏感性分析。
- 仍有 12 个已知基因不在候选集内（原审计为 12 个，已如实保留）。

## 四、已知缺陷

1. BioGRID 的 86.6 万条边未按证据类型过滤（原始文件含全部实验类型），密度偏高，不适合直接替换 STRING。
2. 疾病—基因的 585 对几乎全部来自 OMIM/Orphanet 的同一批整理，跨源增量为 17 对，标签层实质上没有扩大。
3. GTEx 只有 5 个组织的中位表达，没有样本级矩阵，做不了共表达。
4. 通路三库之间没有实体对齐（同一通路在 Reactome 与 KEGG 各算各的）。
5. 组织网络（HumanBase 心/主动脉/动脉）体积大，暂未入链，单独存在 `data/external/networks/humanbase/`。

## 五、怎么用

```python
import pandas as pd
chain = "ARBEIT_G3S1/SENIOR_MODEL_STRUCTURE/phenotype-modules-v0/data/chain"
disease = pd.read_parquet(f"{chain}/dim_disease.parquet")
labels = pd.read_parquet(f"{chain}/fact_disease_gene.parquet")
```

重跑：`python scripts/68_build_data_chain.py`，约 40 秒，输出覆盖写。
