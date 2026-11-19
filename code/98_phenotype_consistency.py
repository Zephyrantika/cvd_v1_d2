"""外部效度（一）：表型一致性。

对每个模块，检验「模块基因的 HPO 注释」是否富集了该疾病的表型术语：
  对疾病表型集合里的每个术语 t，做超几何检验（模块内带 t 注释的基因数 vs 全基因组带 t 的基因数）
  取 BH 校正后的最小 FDR 作为该模块的表型一致性
另算 Jaccard：模块基因的 HPO 术语并集 与 疾病表型集合 的重叠比例。

该指标不使用 RWR 分数、不使用网络拓扑，只依赖基因—表型注释。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, wilcoxon
from statsmodels.stats.multitest import multipletests

DIR = Path(__file__).resolve().parents[1]
UNIVERSE = 16992


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    chain = DIR / "data/chain"
    gene_hpo = pd.read_parquet(chain / "fact_gene_phenotype.parquet")
    gene_hpo["gene_id"] = gene_hpo["gene_id"].astype(str)
    gene_hpo["hpo_id"] = gene_hpo["hpo_id"].astype(str)
    terms_by_gene = gene_hpo.groupby("gene_id")["hpo_id"].apply(set).to_dict()
    genes_by_term = gene_hpo.groupby("hpo_id")["gene_id"].apply(set).to_dict()

    disease_hpo = pd.read_parquet(chain / "fact_disease_phenotype.parquet")
    disease_hpo["disease_id"] = disease_hpo["disease_id"].astype(str)
    disease_hpo["hpo_id"] = disease_hpo["hpo_id"].astype(str)
    terms_by_disease = disease_hpo.groupby("disease_id")["hpo_id"].apply(set).to_dict()

    def consistency(genes: set[str], disease: str) -> tuple[float, float]:
        terms = terms_by_disease.get(disease, set())
        targets = {g for g in genes if g in terms_by_gene}
        if not terms or len(targets) < 3:
            return np.nan, np.nan
        union_terms = set()
        for g in targets:
            union_terms |= terms_by_gene[g]
        jaccard = len(union_terms & terms) / len(union_terms | terms) if (union_terms | terms) else np.nan
        pvals = []
        for term in terms:
            carriers = genes_by_term.get(term, set())
            if len(carriers) < 5:
                continue
            x = len(targets & carriers)
            if x == 0:
                continue
            pvals.append(hypergeom.sf(x - 1, UNIVERSE, len(carriers), len(targets)))
        if not pvals:
            return np.nan, jaccard
        q = multipletests(pvals, method="fdr_bh")[1].min()
        return float(q), jaccard

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    r1 = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    r1["gene_id"] = r1["gene_id"].astype(str)
    r1_sets = r1.groupby("disease_id")["gene_id"].apply(set).to_dict()

    rows = []
    for disease, module_id in largest.items():
        if disease not in r1_sets:
            continue
        base_genes = set(list(base_sets.get(module_id, set()))[:25])
        qb, jb = consistency(base_genes, disease)
        qr, jr = consistency(r1_sets[disease], disease)
        rows.append({"disease_id": disease, "q_base": qb, "q_r1": qr, "j_base": jb, "j_r1": jr})

    table = pd.DataFrame(rows)
    out = DIR / f"outputs/phenotype_consistency/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "consistency.parquet", index=False)

    def summarize(col_a: str, col_b: str, lower_better: bool) -> dict:
        paired = table[[col_a, col_b]].dropna()
        if not len(paired):
            return {}
        better = (paired[col_b] < paired[col_a]) if lower_better else (paired[col_b] > paired[col_a])
        return {"n": int(len(paired)),
                "base_median": float(paired[col_a].median()),
                "r1_median": float(paired[col_b].median()),
                "r1_better": int(better.sum()),
                "p": float(wilcoxon(paired[col_b], paired[col_a]).pvalue)}

    payload = {"fold": args.fold,
               "fdr": summarize("q_base", "q_r1", lower_better=True),
               "jaccard": summarize("j_base", "j_r1", lower_better=False)}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
