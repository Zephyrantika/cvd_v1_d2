"""迭代三：跨疾病共享模块的识别与可靠性评价。

做法：
  1. 统计每个基因出现在多少个疾病的模块里（共享度）；
  2. 每个模块算 shared_frac（共享度 ≥ 阈值的成员比例）；
  3. 与 A 级四项判据做相关，回答「共享模块更可靠还是更泛化」；
  4. 输出高频共享基因清单及其通路注释。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--shared-threshold", type=int, default=10)
    args = parser.parse_args()

    members = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    members["gene_id"] = members["gene_id"].astype(str)
    members["disease_id"] = members["disease_id"].astype(str)

    gene_disease_count = members.groupby("gene_id")["disease_id"].nunique()
    shared_genes = set(gene_disease_count[gene_disease_count >= args.shared_threshold].index)

    module_stats = []
    for disease, group in members.groupby("disease_id"):
        genes = set(group["gene_id"])
        module_stats.append({"disease_id": disease, "size": len(genes),
                             "shared_frac": len(genes & shared_genes) / len(genes),
                             "mean_sharedness": float(np.mean([gene_disease_count.get(g, 0) for g in genes]))})
    stats = pd.DataFrame(module_stats)

    graded_path = DIR / f"outputs/refined_grading/fold_{args.fold}_r1/refined_graded.parquet"
    out = DIR / f"outputs/shared_modules/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"fold": args.fold, "shared_threshold": args.shared_threshold,
               "shared_genes": int(len(shared_genes)),
               "modules": int(len(stats)),
               "shared_frac_mean": round(float(stats.shared_frac.mean()), 4),
               "shared_frac_median": round(float(stats.shared_frac.median()), 4)}

    if graded_path.exists():
        graded = pd.read_parquet(graded_path)[["disease_id", "p_spec", "replication", "hub", "go_fdr", "grade"]]
        merged = stats.merge(graded, on="disease_id")
        merged.to_parquet(out / "shared_vs_grade.parquet", index=False)
        correlations = {}
        for column in ("p_spec", "replication", "hub"):
            rho, p = spearmanr(merged.shared_frac, merged[column])
            correlations[column] = {"rho": round(float(rho), 3), "p": float(p)}
        merged["A"] = merged.grade == "A"
        high = merged[merged.shared_frac >= merged.shared_frac.median()]
        low = merged[merged.shared_frac < merged.shared_frac.median()]
        payload["correlations"] = correlations
        payload["A_rate_high_shared"] = round(float(high.A.mean()), 4)
        payload["A_rate_low_shared"] = round(float(low.A.mean()), 4)

    top_shared = gene_disease_count.sort_values(ascending=False).head(20)
    pathways = pd.read_parquet(DIR / "data/chain/fact_gene_pathway.parquet")
    pathways["gene_id"] = pathways["gene_id"].astype(str)
    dim_pathway = pd.read_parquet(DIR / "data/chain/dim_pathway.parquet")
    names = {str(pid): str(pid) for pid in dim_pathway["pathway_id"]}
    listing = []
    for gene, count in top_shared.items():
        pts = pathways[pathways.gene_id == gene]["pathway_id"].astype(str).head(3).tolist()
        listing.append({"gene_id": gene, "n_diseases": int(count),
                        "pathways": [names.get(p, p) for p in pts]})
    payload["top_shared_genes"] = listing

    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "top_shared_genes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
