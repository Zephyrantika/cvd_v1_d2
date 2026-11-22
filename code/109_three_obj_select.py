"""迭代一：三目标选点（z、复现率、表型一致性）。

候选来源与 R1 相同（baseline / lam0.5 / lam1.0 / size15-40），新增第三目标：
模块基因 HPO 注释与疾病表型的 Jaccard。选择规则：
  约束 rep≥0.7、hub≤0.5；在满足约束的候选里取 z + κ·std(pheno) 最大。
输出 members 与 selected，供 73 号脚本分级。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]
KAPPA = 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    args = parser.parse_args()

    out = DIR / "outputs/three_obj"
    out.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(DIR / f"outputs/pareto/byp_fold{args.fold}/candidates_byp.parquet")
    candidates["disease_id"] = candidates["disease_id"].astype(str)

    source_files = {
        "baseline": DIR / f"outputs/pareto/base_members_fold{args.fold}.parquet",
        "lam0.5": DIR / f"outputs/lam_refine/fold_{args.fold}_lam0.5_mu1.0/lam_refine_members.parquet",
        "lam1.0": DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0/lam_refine_members.parquet",
        "size15": DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0_size15/lam_refine_members.parquet",
        "size20": DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0_size20/lam_refine_members.parquet",
        "size30": DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0_size30/lam_refine_members.parquet",
        "size40": DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0_size40/lam_refine_members.parquet",
    }
    members: dict[str, dict[str, set[str]]] = {}
    for name, path in source_files.items():
        if not path.exists():
            print(f"[warn] missing {name}: {path}")
            continue
        frame = pd.read_parquet(path)
        frame["gene_id"] = frame["gene_id"].astype(str)
        frame["disease_id"] = frame["disease_id"].astype(str)
        members[name] = frame.groupby("disease_id")["gene_id"].apply(set).to_dict()

    chain = DIR / "data/chain"
    gene_hpo = pd.read_parquet(chain / "fact_gene_phenotype.parquet")
    gene_hpo["gene_id"] = gene_hpo["gene_id"].astype(str)
    gene_hpo["hpo_id"] = gene_hpo["hpo_id"].astype(str)
    terms_by_gene = gene_hpo.groupby("gene_id")["hpo_id"].apply(set).to_dict()
    disease_hpo = pd.read_parquet(chain / "fact_disease_phenotype.parquet")
    disease_hpo["disease_id"] = disease_hpo["disease_id"].astype(str)
    disease_hpo["hpo_id"] = disease_hpo["hpo_id"].astype(str)
    terms_by_disease = disease_hpo.groupby("disease_id")["hpo_id"].apply(set).to_dict()

    ppi = pd.read_parquet(DIR.parent / "phenotype-network-v0/data/processed/string_gene_edges_700.parquet",
                          columns=["gene_a", "gene_b", "weight"])
    ppi["gene_a"] = ppi["gene_a"].astype(str)
    ppi["gene_b"] = ppi["gene_b"].astype(str)

    def pheno_jaccard(genes: set[str], disease: str) -> float:
        terms = terms_by_disease.get(disease, set())
        union = set()
        for g in genes:
            union |= terms_by_gene.get(g, set())
        if not union or not terms:
            return 0.0
        return len(union & terms) / len(union | terms)

    def hub_of(genes: set[str]) -> float:
        sub = ppi[ppi["gene_a"].isin(genes) & ppi["gene_b"].isin(genes)]
        if sub.empty:
            return 1.0
        deg: dict[str, float] = {}
        for a, b, w in zip(sub["gene_a"], sub["gene_b"], sub["weight"].astype(float)):
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
        total = sum(deg.values())
        return float(max(deg.values()) / total) if total > 0 else 1.0

    rows = []
    for row in candidates.itertuples(index=False):
        source = row.source
        if source not in members:
            continue
        module = members[source].get(row.disease_id)
        if not module or len(module) < 5:
            continue
        rows.append({"disease_id": row.disease_id, "source": source, "z": float(row.z),
                     "p_spec": float(row.p_spec), "replication": float(row.replication),
                     "pheno": pheno_jaccard(module, row.disease_id), "hub": hub_of(module)})
    table = pd.DataFrame(rows)
    if table.empty:
        print("[error] no candidates")
        return 1
    table["pheno_z"] = (table["pheno"] - table["pheno"].mean()) / (table["pheno"].std(ddof=0) + 1e-9)
    table["score"] = table["z"] + args.kappa * table["pheno_z"]
    table.to_parquet(out / f"candidates_three_fold{args.fold}.parquet", index=False)

    picked = []
    for disease, group in table.groupby("disease_id"):
        eligible = group[(group.replication >= 0.7) & (group.hub <= 0.5)]
        pool = eligible if len(eligible) else group
        best = pool.sort_values("score", ascending=False).iloc[0]
        picked.append({"disease_id": disease, "source": best.source, "z": float(best.z),
                       "replication": float(best.replication), "pheno": float(best.pheno),
                       "hub": float(best.hub)})
    selected = pd.DataFrame(picked)
    selected.to_parquet(out / f"selected_three_fold{args.fold}.parquet", index=False)

    member_rows = []
    for row in selected.itertuples(index=False):
        module = members[row.source].get(row.disease_id, set())
        for gene in module:
            member_rows.append({"module_id": f"{row.disease_id}:three", "disease_id": row.disease_id,
                                "gene_id": gene})
    pd.DataFrame(member_rows).to_parquet(out / f"members_three_fold{args.fold}.parquet", index=False)

    summary = {"fold": args.fold, "kappa": args.kappa, "diseases": int(len(selected)),
               "source_mix": selected.source.value_counts().to_dict(),
               "z_mean": round(float(selected.z.mean()), 3),
               "pheno_mean": round(float(selected.pheno.mean()), 4),
               "rep_mean": round(float(selected.replication.mean()), 3)}
    (out / f"summary_fold{args.fold}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
