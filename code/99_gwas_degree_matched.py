"""外部效度（二）：GWAS 富集的度数匹配零模型。

原来的零模型是从全基因组随机抽同数量基因，但基线模块天生 hub 密集、GWAS 基因也 hub 密集，
等于基线占便宜。这里改成：对模块里的每个基因，从**同一度数分位**的基因里抽一个替代，
保持度数构成不变，再看 GWAS 命中比例。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DIR = Path(__file__).resolve().parents[1]
V0 = DIR.parent / "phenotype-network-v0"
DRAWS = 100
BINS = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    universe = pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str).tolist()
    universe_set = set(universe)

    gwas = pd.read_parquet(V0 / "data/processed/gwas_gene_support.parquet", columns=["gene_id", "p_value"])
    gwas["gene_id"] = gwas["gene_id"].astype(str)
    gwas_any = set(gwas["gene_id"]) & universe_set

    ppi = pd.read_parquet(V0 / "data/processed/string_gene_edges_700.parquet",
                          columns=["gene_a", "gene_b", "weight"])
    ppi["gene_a"] = ppi["gene_a"].astype(str)
    ppi["gene_b"] = ppi["gene_b"].astype(str)
    degree: dict[str, float] = {}
    for a, b, w in zip(ppi["gene_a"], ppi["gene_b"], ppi["weight"].astype(float)):
        degree[a] = degree.get(a, 0.0) + float(w)
        degree[b] = degree.get(b, 0.0) + float(w)

    deg_series = pd.Series({g: degree.get(g, 0.0) for g in universe})
    bins = pd.qcut(deg_series.rank(method="first"), BINS, labels=False)
    pool_by_bin: dict[int, list[str]] = {}
    for gene, b in bins.items():
        pool_by_bin.setdefault(int(b), []).append(gene)
    bin_of = {g: int(b) for g, b in bins.items()}

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    r1 = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    r1["gene_id"] = r1["gene_id"].astype(str)
    r1_sets = r1.groupby("disease_id")["gene_id"].apply(set).to_dict()

    def matched_z(genes: set[str]) -> float:
        target = [g for g in genes if g in bin_of]
        if len(target) < 5:
            return np.nan
        observed = len([g for g in target if g in gwas_any]) / len(target)
        nulls = []
        for _ in range(args.draws):
            sample = [rng.choice(pool_by_bin[bin_of[g]]) for g in target]
            nulls.append(len([g for g in sample if g in gwas_any]) / len(sample))
        nulls = np.array(nulls)
        return float((observed - nulls.mean()) / (nulls.std(ddof=1) + 1e-9))

    rows = []
    for disease, module_id in largest.items():
        if disease not in r1_sets:
            continue
        rows.append({"disease_id": disease,
                     "z_base": matched_z(set(list(base_sets.get(module_id, set()))[:25])),
                     "z_r1": matched_z(r1_sets[disease])})
    table = pd.DataFrame(rows)
    out = DIR / f"outputs/gwas_matched/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "gwas_matched.parquet", index=False)

    paired = table.dropna()
    payload = {"fold": args.fold, "draws": args.draws, "n": int(len(paired)),
               "base_median": round(float(paired.z_base.median()), 3),
               "r1_median": round(float(paired.z_r1.median()), 3),
               "r1_better": int((paired.z_r1 > paired.z_base).sum()),
               "p": float(wilcoxon(paired.z_r1, paired.z_base).pvalue)}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
