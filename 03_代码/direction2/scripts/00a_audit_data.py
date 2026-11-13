"""M0 数据审计与分层。

只读 V0 产物，统计每病正样本数、HPO 术语数、基因复用度、关联来源，
按 T1-T4 打层，并与 DOCUMENTATION/08_v1_outcome_and_data_audit.md 第 3 节
逐项核对。所有不一致都显式列出，不静默改数。

口径（重要）：
  - association_rows : 原始记录行数，一条记录一个来源，同一 (disease,gene)
                       若同时来自 NCBI 与 Orphanet 会占两行。
  - unique_pairs     : 唯一的 (disease, gene) 对，是"某病有几个已知基因"的
                       正确基数，分层与所有统计都用这个。
本脚本以 unique_pairs 为准，同时报出 rows，供与文档对账。

用法：
    python scripts/00a_audit_data.py --v0-root <phenotype-network-v0> \
        --config configs/tiers.yaml --out outputs/audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


DEFAULTS = {
    "tiers": {"t1_min_positives": 5, "t2_min_positives": 3, "t4_max_hpo": 3},
    "reporting": {"primary_pool": ["T2", "T3"], "t1_separate": True},
    "recovery": {"allowed_tiers": ["T1", "T2"]},
}

# 08 文档第 3 节声称的数字。status 由脚本判定，不在此处预判。
DOC_CLAIMS = [
    ("疾病数", 452),
    ("关联记录行数", 688),
    ("唯一基因数", 416),
    ("来源 NCBI 条数", 462),
    ("来源 Orphanet 条数", 226),
    ("每病正样本中位数", 1),
    ("每病正样本均值", 1.26),
    ("每病正样本最大值", 15),
    ("1-2 个正样本的疾病数", 426),
    ("3-5 个正样本的疾病数", 20),
    ("6-10 个正样本的疾病数", 3),
    ("11-20 个正样本的疾病数", 3),
    ("≥21 个正样本的疾病数", 0),
    ("只关联一个疾病的基因数", 319),
    ("跨疾病复用基因数", 97),
    ("最大基因复用疾病数", 7),
    ("每病 HPO 中位数", 16),
    ("每病 HPO 最小值", 3),
    ("每病 HPO 最大值", 156),
    ("HPO<=3 的疾病数", 18),
    ("HPO<=5 的疾病数", 44),
    ("正样本数与 HPO 数的 Spearman", 0.205),
    ("T1（≥5 个正样本）疾病数", 9),
    ("T2（3-4 个正样本）疾病数", 17),
]

# 控制台输出用 ASCII 标签，避免 Windows GBK 控制台把中文打乱；中文只在文件里。
ASCII_LABELS = {
    "疾病数": "n_diseases",
    "关联记录行数": "n_association_rows",
    "唯一基因数": "n_unique_genes",
    "来源 NCBI 条数": "n_source_ncbi",
    "来源 Orphanet 条数": "n_source_orphanet",
    "每病正样本中位数": "median_positives",
    "每病正样本均值": "mean_positives",
    "每病正样本最大值": "max_positives",
    "1-2 个正样本的疾病数": "n_diseases_1_2",
    "3-5 个正样本的疾病数": "n_diseases_3_5",
    "6-10 个正样本的疾病数": "n_diseases_6_10",
    "11-20 个正样本的疾病数": "n_diseases_11_20",
    "≥21 个正样本的疾病数": "n_diseases_21_plus",
    "只关联一个疾病的基因数": "n_genes_single_disease",
    "跨疾病复用基因数": "n_genes_reused",
    "最大基因复用疾病数": "max_gene_reuse",
    "每病 HPO 中位数": "median_hpo",
    "每病 HPO 最小值": "min_hpo",
    "每病 HPO 最大值": "max_hpo",
    "HPO<=3 的疾病数": "n_diseases_hpo_le3",
    "HPO<=5 的疾病数": "n_diseases_hpo_le5",
    "正样本数与 HPO 数的 Spearman": "spearman_pos_vs_hpo",
    "T1（≥5 个正样本）疾病数": "n_tier_t1",
    "T2（3-4 个正样本）疾病数": "n_tier_t2",
}


def load_config(path: Path | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for section, values in loaded.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    return cfg


def source_bucket(source: str) -> str:
    s = str(source).lower()
    if "ncbi" in s:
        return "ncbi"
    if "orphadata" in s or "orphanet" in s:
        return "orphanet"
    return "other"


def bucket_positives(n: int) -> str:
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    return ">=21"


def assign_tier(n_pos: int, n_hpo: int, cfg: dict) -> tuple[str, bool]:
    t1 = int(cfg["tiers"]["t1_min_positives"])
    t2 = int(cfg["tiers"]["t2_min_positives"])
    t4 = int(cfg["tiers"]["t4_max_hpo"])
    if n_pos >= t1:
        tier = "T1"
    elif n_pos >= t2:
        tier = "T2"
    else:
        tier = "T3"
    return tier, bool(n_hpo <= t4)


def available_metrics(tier: str, recovery_tiers: list[str]) -> str:
    base = "z_spec;stability;go"
    if tier in recovery_tiers:
        return base + ";recovery(descriptive-only)"
    return base


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0-root", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/audit")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    recovery_tiers = list(cfg["recovery"].get("allowed_tiers", ["T1", "T2"]))
    v0_root = Path(args.v0_root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assoc = pd.read_parquet(v0_root / "data/processed/cardiovascular_disease_genes.parquet")
    pheno = pd.read_parquet(v0_root / "data/processed/cardiovascular_disease_phenotypes.parquet")
    folds = pd.read_parquet(v0_root / "data/folds/disease_fold_assignments.parquet")
    universe = pd.read_parquet(v0_root / "data/folds/candidate_gene_universe.parquet")

    assoc = assoc.copy()
    assoc["source_bucket"] = assoc["source"].map(source_bucket)

    # ---- 正样本：按唯一 (disease, gene) 对计 ----
    pairs = assoc[["disease_id", "gene_id"]].drop_duplicates()
    n_pairs = int(len(pairs))
    n_rows = int(len(assoc))
    per_pair_sources = assoc.groupby(["disease_id", "gene_id"])["source_bucket"].nunique()
    n_pairs_multi_source = int((per_pair_sources > 1).sum())

    pos = pairs.groupby("disease_id")["gene_id"].nunique().rename("n_positives")
    hpo = pheno.groupby("disease_id")["hpo_id"].nunique().rename("n_hpo")

    names = (
        assoc.drop_duplicates("disease_id")[["disease_id", "disease_name"]]
        .set_index("disease_id")["disease_name"]
    )

    table = folds[["disease_id", "fold", "disease_family_id"]].copy()
    table["disease_name"] = table["disease_id"].map(names)
    table = table.merge(pos, left_on="disease_id", right_index=True, how="left")
    table = table.merge(hpo, left_on="disease_id", right_index=True, how="left")
    table["n_positives"] = table["n_positives"].fillna(0).astype(int)
    table["n_hpo"] = table["n_hpo"].fillna(0).astype(int)
    table[["tier", "t4_phenotype_poor"]] = table.apply(
        lambda r: pd.Series(assign_tier(r["n_positives"], r["n_hpo"], cfg)), axis=1
    )
    table["available_metrics"] = table["tier"].map(lambda t: available_metrics(t, recovery_tiers))

    assert len(table) == 452, f"disease table has {len(table)} rows"
    assert table["tier"].isin(["T1", "T2", "T3"]).all()

    tier_counts = table["tier"].value_counts().reindex(["T1", "T2", "T3"], fill_value=0).astype(int)
    t4_count = int(table["t4_phenotype_poor"].sum())

    # ---- 基因复用与来源 ----
    gene_reuse = pairs.groupby("gene_id")["disease_id"].nunique()
    source_counts = assoc["source_bucket"].value_counts().to_dict()
    type_counts = assoc["association_type"].value_counts().to_dict()

    # ---- 分布 ----
    dist = table["n_positives"].map(bucket_positives).value_counts()
    dist = dist.reindex(["1-2", "3-5", "6-10", "11-20", ">=21"], fill_value=0).astype(int)
    spearman = float(table[["n_positives", "n_hpo"]].corr(method="spearman").iloc[0, 1])

    # ---- 候选集覆盖 ----
    universe_ids = set(universe["gene_id"].astype(str))
    positive_genes = set(pairs["gene_id"].astype(str))
    n_pos_in_universe = len(positive_genes & universe_ids)

    # ---- 数据完整性 ----
    fold_diseases = set(folds["disease_id"])
    integrity = {
        "n_diseases_in_folds": int(len(fold_diseases)),
        "assoc_diseases_outside_folds": int(len(set(assoc["disease_id"]) - fold_diseases)),
        "pheno_diseases_outside_folds": int(len(set(pheno["disease_id"]) - fold_diseases)),
        "diseases_without_phenotype": int(len(fold_diseases - set(pheno["disease_id"]))),
        "duplicate_disease_rows_in_folds": int(folds["disease_id"].duplicated().sum()),
    }
    graph_checks = []
    for fold in sorted(int(f) for f in folds["fold"].unique()):
        node_map = pd.read_parquet(v0_root / f"data/graphs/fold_{fold}/node_map.parquet")
        genes = set(node_map.loc[node_map["node_type"] == "gene", "node_id"].astype(str))
        graph_checks.append(
            {
                "fold": fold,
                "n_nodes": int(len(node_map)),
                "n_genes": int(len(genes)),
                "n_phenotypes": int((node_map["node_type"] == "phenotype").sum()),
                "n_pathways": int((node_map["node_type"] == "pathway").sum()),
                "genes_equal_candidate_universe": bool(genes == universe_ids),
            }
        )
    opp = pairs[~pairs["gene_id"].astype(str).isin(universe_ids)].copy()
    opp["gene_id"] = opp["gene_id"].astype(str)
    opp = opp.merge(names.rename("disease_name"), left_on="disease_id", right_index=True, how="left")
    opp = opp[["disease_id", "disease_name", "gene_id"]].sort_values(["disease_id", "gene_id"])

    # ---- 与文档对账 ----
    observed = {
        "疾病数": int(len(table)),
        "关联记录行数": n_rows,
        "唯一基因数": int(gene_reuse.size),
        "来源 NCBI 条数": int(source_counts.get("ncbi", 0)),
        "来源 Orphanet 条数": int(source_counts.get("orphanet", 0)),
        "每病正样本中位数": int(table["n_positives"].median()),
        "每病正样本均值": round(float(table["n_positives"].mean()), 2),
        "每病正样本最大值": int(table["n_positives"].max()),
        "1-2 个正样本的疾病数": int(dist["1-2"]),
        "3-5 个正样本的疾病数": int(dist["3-5"]),
        "6-10 个正样本的疾病数": int(dist["6-10"]),
        "11-20 个正样本的疾病数": int(dist["11-20"]),
        "≥21 个正样本的疾病数": int(dist[">=21"]),
        "只关联一个疾病的基因数": int((gene_reuse == 1).sum()),
        "跨疾病复用基因数": int((gene_reuse > 1).sum()),
        "最大基因复用疾病数": int(gene_reuse.max()),
        "每病 HPO 中位数": int(table["n_hpo"].median()),
        "每病 HPO 最小值": int(table["n_hpo"].min()),
        "每病 HPO 最大值": int(table["n_hpo"].max()),
        "HPO<=3 的疾病数": int((table["n_hpo"] <= 3).sum()),
        "HPO<=5 的疾病数": int((table["n_hpo"] <= 5).sum()),
        "正样本数与 HPO 数的 Spearman": round(spearman, 3),
        "T1（≥5 个正样本）疾病数": int(tier_counts["T1"]),
        "T2（3-4 个正样本）疾病数": int(tier_counts["T2"]),
    }

    def match(a, b) -> bool:
        if isinstance(a, float) or isinstance(b, float):
            return abs(float(a) - float(b)) <= 5e-3
        return int(a) == int(b)

    reconciliation = []
    for item, doc_value in DOC_CLAIMS:
        got = observed[item]
        status = "match" if match(got, doc_value) else "differs"
        row = {"item": item, "doc_value": doc_value, "recomputed": got, "status": status}
        if item == "关联记录行数":
            row["note"] = f"行数 {n_rows}，去重后唯一 (disease,gene) 对 {n_pairs}"
        if item == "T1（≥5 个正样本）疾病数":
            row["note"] = "文档按 ≥6 统计，阈值写的是 ≥5"
        if item == "T2（3-4 个正样本）疾病数":
            row["note"] = "文档的 20 是 3-5 桶，含 3 个恰好 5 个正样本的疾病"
        reconciliation.append(row)

    # ---- 折 × 层 ----
    crosstab = pd.crosstab(table["fold"], table["tier"]).reindex(
        columns=["T1", "T2", "T3"], fill_value=0
    )
    crosstab["合计"] = crosstab.sum(axis=1)

    # ---- 分层概括 ----
    tier_summary = (
        table.groupby("tier")
        .agg(
            疾病数=("disease_id", "nunique"),
            正样本中位数=("n_positives", "median"),
            正样本均值=("n_positives", "mean"),
            正样本最大=("n_positives", "max"),
            HPO中位数=("n_hpo", "median"),
            表型不足数=("t4_phenotype_poor", "sum"),
        )
        .reindex(["T1", "T2", "T3"])
    )
    tier_summary["正样本均值"] = tier_summary["正样本均值"].round(2)

    t1_table = table[table["tier"] == "T1"][
        ["disease_id", "disease_name", "fold", "n_positives", "n_hpo"]
    ].sort_values(["n_positives", "disease_id"], ascending=[False, True])

    # ---- 落盘 ----
    out_cols = [
        "disease_id", "disease_name", "fold", "disease_family_id",
        "n_positives", "n_hpo", "tier", "t4_phenotype_poor", "available_metrics",
    ]
    table[out_cols].to_parquet(out_dir / "disease_tiers.parquet", index=False)
    crosstab.to_csv(out_dir / "tier_by_fold.csv", encoding="utf-8-sig")
    t1_table.to_csv(out_dir / "t1_diseases.csv", index=False, encoding="utf-8-sig")
    opp.to_csv(out_dir / "positives_outside_universe.csv", index=False, encoding="utf-8-sig")

    report = {
        "config": cfg,
        "v0_root": str(v0_root),
        "counting_base": {
            "association_rows": n_rows,
            "unique_disease_gene_pairs": n_pairs,
            "pairs_with_multiple_sources": n_pairs_multi_source,
            "note": "分层与所有统计以 unique_pairs 为基数；rows 仅用于与文档对账",
        },
        "totals": {
            "n_diseases": int(len(table)),
            "n_genes": int(gene_reuse.size),
            "n_positive_genes_in_candidate_universe": n_pos_in_universe,
            "n_positive_genes_outside_universe": int(len(positive_genes) - n_pos_in_universe),
        },
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "integrity": integrity,
        "graph_checks": graph_checks,
        "n_positive_genes_outside_universe": int(len(positive_genes) - n_pos_in_universe),
        "t4_cross_count": t4_count,
        "t4_cross_by_tier": {
            f"{tier}": int(v) for tier, v in table.groupby("tier")["t4_phenotype_poor"].sum().items()
        },
        "positives_distribution": {k: int(v) for k, v in dist.items()},
        "association_types": {str(k): int(v) for k, v in type_counts.items()},
        "spearman_positives_vs_hpo": round(spearman, 4),
        "reconciliation_with_doc_08": reconciliation,
        "n_differs": int(sum(1 for r in reconciliation if r["status"] != "match")),
    }
    (out_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---- 人读的审计说明 ----
    diff_rows = pd.DataFrame(
        [
            {"项目": r["item"], "08 文档": r["doc_value"], "本次复算": r["recomputed"], "说明": r.get("note", "")}
            for r in reconciliation
            if r["status"] != "match"
        ]
    )
    note = f"""# 数据审计与分层（M0）

生成脚本：`scripts/00a_audit_data.py`，输入只读 V0 产物，配置见 `configs/tiers.yaml`。
审计日期：2026-10-28。

## 结论

当前基准集是 **{len(table)} 个心血管疾病、{n_pairs} 个唯一的疾病—基因对（原始记录 {n_rows} 行）、{int(gene_reuse.size)} 个基因、{int((table['n_hpo']>0).sum())} 个有表型注释的疾病**。按每病正样本数分层：T1（≥{cfg['tiers']['t1_min_positives']}）**{tier_counts['T1']} 个**，T2（{cfg['tiers']['t2_min_positives']}-{cfg['tiers']['t1_min_positives']-1}）**{tier_counts['T2']} 个**，T3（1-2）**{tier_counts['T3']} 个**；另有 {t4_count} 个疾病 HPO ≤{cfg['tiers']['t4_max_hpo']}，与 T1-T3 交叉，单独标记。

每病正样本中位数 {int(table['n_positives'].median())}，均值 {table['n_positives'].mean():.2f}，最大 {int(table['n_positives'].max())}；1-2 个正样本的疾病占 {dist['1-2']/len(table)*100:.1f}%。这个规模下，任何"恢复了多少已知疾病基因"的指标都不能作为主指标。

## 口径

`association_rows` 与 `unique_pairs` 是两个基数，不能混用。同一 (disease, gene) 若同时被 NCBI 与 Orphanet 收录会占两行：本次 {n_rows} 行记录去重后是 {n_pairs} 对，其中 {n_pairs_multi_source} 对来自两个来源。**分层、均值、所有后续统计一律用唯一对**；行数只用于与旧文档对账。

## 数据完整性

- {integrity['n_diseases_in_folds']} 个疾病全部在折划分里，重复分配 {integrity['duplicate_disease_rows_in_folds']} 条。
- 关联表里落在基准集之外的疾病 {integrity['assoc_diseases_outside_folds']} 个，表型表里 {integrity['pheno_diseases_outside_folds']} 个；没有表型注释的疾病 {integrity['diseases_without_phenotype']} 个。
- 每病 HPO 术语最少 {int(table['n_hpo'].min())} 个，中位数 {int(table['n_hpo'].median())} 个。
- 五折图的基因节点与 16,992 候选集一致的有 {sum(1 for g in graph_checks if g['genes_equal_candidate_universe'])}/{len(graph_checks)} 折，每折节点 {graph_checks[0]['n_nodes']:,} 个（表型 {graph_checks[0]['n_phenotypes']:,}、基因 {graph_checks[0]['n_genes']:,}、通路 {graph_checks[0]['n_pathways']:,}）。
- 416 个正样本基因里 {n_pos_in_universe} 个在候选集内，{len(positive_genes) - n_pos_in_universe} 个不在，清单见 `positives_outside_universe.csv`。

## 与 08 文档的对账

{md_table(diff_rows) if len(diff_rows) else '全部一致。'}

共 {len(reconciliation)} 项对账，{int(sum(1 for r in reconciliation if r['status'] != 'match'))} 项不一致。

## 分层结果

{md_table(tier_summary.reset_index().rename(columns={'tier': '层'}))}

T4（HPO ≤3）分布：{', '.join(f"{k} {v} 个" for k, v in report['t4_cross_by_tier'].items())}，与上表交叉，不可相加。

## 折 × 层

{md_table(crosstab.reset_index().rename(columns={'fold': '折'}))}

## T1 疾病清单

{md_table(t1_table)}

## 使用约定

- 后续所有结果表必须带 `tier` 列；T1 单独成表，不与 T3 混合求平均。
- 基因恢复类指标只在 {'/'.join(recovery_tiers)} 上计算，且需标注疾病数与正样本数。
- 改 `configs/tiers.yaml` 等同换口径，必须重跑本脚本及所有下游。
- 08 等文档的 T1/T2 数字已于 2026-10-28 修订，改动清单见交接包根目录的 CORRECTIONS.md。
"""
    (out_dir / "AUDIT_NOTE.md").write_text(note, encoding="utf-8")

    # ---- 控制台（ASCII，避免 GBK 控制台乱码）----
    print(f"v0_root: {v0_root}")
    print(f"diseases={len(table)} rows={n_rows} unique_pairs={n_pairs} multi_source_pairs={n_pairs_multi_source}")
    print(f"genes={int(gene_reuse.size)} positives_in_universe={n_pos_in_universe}")
    print(f"tiers: T1={tier_counts['T1']} T2={tier_counts['T2']} T3={tier_counts['T3']} t4_cross={t4_count}")
    print(
        "integrity: diseases_in_folds="
        f"{integrity['n_diseases_in_folds']} assoc_outside={integrity['assoc_diseases_outside_folds']} "
        f"pheno_outside={integrity['pheno_diseases_outside_folds']} no_pheno={integrity['diseases_without_phenotype']}"
    )
    print(
        "graph: gene_set_matches_universe="
        f"{sum(1 for g in graph_checks if g['genes_equal_candidate_universe'])}/{len(graph_checks)}"
    )
    print(f"positives: median={table['n_positives'].median():.1f} mean={table['n_positives'].mean():.2f} max={table['n_positives'].max()}")
    print(f"hpo: median={table['n_hpo'].median():.1f} min={table['n_hpo'].min()} max={table['n_hpo'].max()}")
    print(f"spearman(pos,hpo)={spearman:.4f}")
    print("")
    print("reconciliation vs DOC 08:")
    for i, r in enumerate(reconciliation):
        flag = "OK  " if r["status"] == "match" else "DIFF"
        label = ASCII_LABELS.get(r["item"], f"item_{i:02d}")
        print(f"  [{flag}] {label}: doc={r['doc_value']} recomputed={r['recomputed']}")
    print(f"diffs={report['n_differs']}/{len(reconciliation)}")
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
