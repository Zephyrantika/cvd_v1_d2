# 数据来源与获取方式

本项目全部输入来自公开数据库。下面按层列出地址、体量与处理脚本，按需自行下载。

## 一、疾病与表型

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| HPO 本体 | https://purl.obolibrary.org/obo/hp/hp.obo | 10 MB | 表型术语与层级 |
| 疾病—表型 | https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa | 34 MB | 452 个疾病的表型集合 |
| 基因—表型 | https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt | 20 MB | 26.5 万对基因—HPO，用于表型一致性 |
| 基因—疾病（HPO） | https://purl.obolibrary.org/obo/hp/hpoa/genes_to_disease.txt | 1.4 MB | 疾病—基因标签来源之一 |
| MONDO 本体 | https://purl.obolibrary.org/obo/mondo.obo | 95 MB | 疾病归一与家族划分 |

## 二、网络

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| STRING | https://string-db.org/cgi/download | 1.0 GB | 蛋白互作，取 400/700/900 三档 |
| BioGRID | https://downloads.thebiogrid.org/BioGRID | 172 MB | 互作补充 |
| HuRI | https://www.interactome-atlas.org/download | 1.6 MB | 实验验证互作组（可选敏感性分析） |
| HumanBase 心脏 | https://s3-us-west-2.amazonaws.com/humanbase/networks/heart_top.gz | 250 MB | 组织网络对照（跨网络检验） |
| HumanBase 动脉 | https://s3-us-west-2.amazonaws.com/humanbase/networks/artery_top.gz | 385 MB | 同上 |
| HumanBase 主动脉 | https://s3-us-west-2.amazonaws.com/humanbase/networks/aorta_top.gz | 1.0 GB | 同上 |

HumanBase 文件为三列 TSV：基因 A、基因 B、概率（Entrez ID）。

## 三、通路与功能

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| Reactome | https://reactome.org/download-data | 96 MB | 基因—通路 |
| GO | http://geneontology.org/docs/download-ontology/ | 41 MB | 富集检验（排除 IEA） |
| KEGG 通路链接 | https://rest.kegg.jp/link/hsa/pathway | 0.9 MB | 基因—通路 |
| KEGG 通路名称 | https://rest.kegg.jp/list/pathway/hsa | 22 KB | 同上 |
| WikiPathways | https://data.wikipathways.org/current/gmt/ | 0.3 MB | 基因—通路（GMT 内为 Entrez） |

## 四、疾病—基因标签

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| ClinVar | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ | 1.3 MB | 疾病—基因（gene_condition_source_id） |
| ClinGen 有效性 | https://search.clinicalgenome.org/kb/gene-validity | 1.1 MB | 分层标签（Definitive/Strong 等） |
| DisGeNET | https://www.disgenet.org/downloads | API | curated 关联，需注册获取 key |
| Open Targets | https://platform.opentargets.org/downloads | API | 疾病—靶点，接口 https://api.platform.opentargets.org/api/v4/graphql |
| CTD | https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz | 3.1 GB | 基因—疾病（direct 与 inferred） |

## 五、表达与变异

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| GTEx（心血管 5 组织） | https://gtexportal.org/home/downloads/adult-gtex | 8 MB | 表达一致性 |
| HPA | https://www.proteinatlas.org/download/proteinatlas.tsv.zip | 7 MB | 组织表达补充 |
| GWAS Catalog | https://www.ebi.ac.uk/gwas/docs/file-downloads | 66 MB | 常见变异支持 |

## 六、映射

| 数据 | 地址 | 体量 | 用途 |
|---|---|---|---|
| HGNC 全表 | https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt | 17 MB | Ensembl / Entrez / 符号互转 |

## 七、大文件的处理方式

有两个数据源超过 GitHub 单文件 2 GB 限制，**未随包上传**：

CTD 原始文件 3.1 GB，直接从上表地址下载，用 `scripts/71_ctd_align.py` 解析（只保留 marker/mechanism 行，MeSH→MONDO、HGNC→Entrez）。解析结果已在 `direction2_v2_external_tables_2026-11-19.zip` 内（209 条 direct、98,625 条 inferred），日常复现不必重下。

HumanBase 三个组织网络合计 1.6 GB，按上表地址逐个下载即可，仅在跨网络检验时用到。

## 八、重建顺序

1. `scripts/61_data_chain.py`、`scripts/68_build_data_chain.py`：把 HPO、MONDO、STRING、Reactome、GO、GTEx、GWAS、ClinVar、ClinGen、DisGeNET、Open Targets、CTD、KEGG、WikiPathways、HPA 对齐成 `data/chain/` 的 10 张表。
2. `scripts/70_disgenet.py`、`scripts/69_opentargets.py`、`scripts/71_ctd_align.py`：三个需要接口或大文件的来源单独解析，产出 parquet 后再并入。
3. `scripts/63_external_layers.py`：HPA 与 WikiPathways 的解析。
4. 其余脚本读 `data/chain/` 与 `data/processed/`，不直接读原始库。

各表行数与校验和见 `data/chain/chain_manifest.json` 与 `chain_qc.json`。
