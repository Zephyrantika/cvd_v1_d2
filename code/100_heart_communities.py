"""外部效度（三）：跨网络社区富集，替代「成员重合 60%」。

做法：在 HumanBase 心脏网络（p≥0.2，81.6 万条边、9,438 个基因）上跑一次 Leiden 得到社区，
再对每个模块检验「模块基因是否显著富集于某个社区」（超几何 + BH），
比较基线模块与 R1 模块的最佳富集显著性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, wilcoxon
from statsmodels.stats.multitest import multipletests

DIR = Path(__file__).resolve().parents[1]
V0 = DIR.parent / "phenotype-network-v0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    universe = pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str).tolist()
    universe_set = set(universe)
    heart = pd.read_csv(DIR / "data/external/networks/humanbase/heart_top.gz", sep="\t", header=None,
                        names=["a", "b", "p"], dtype={"a": str, "b": str})
    heart = heart[(heart["p"] >= args.threshold) & heart["a"].isin(universe_set) & heart["b"].isin(universe_set)]
    genes = sorted(set(heart["a"]) | set(heart["b"]))
    index = {g: i for i, g in enumerate(genes)}
    graph = ig.Graph()
    graph.add_vertices(len(genes))
    graph.add_edges([(index[a], index[b]) for a, b in zip(heart["a"], heart["b"])])
    graph.es["weight"] = heart["p"].to_numpy(dtype=float)
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                         weights="weight", resolution_parameter=args.resolution, seed=42)
    communities = [set(genes[i] for i in block) for block in partition if len(block) >= 20]
    print(f"[info] graph {len(genes)} genes, {graph.ecount()} edges, {len(partition)} communities, kept {len(communities)}")

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    r1 = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    r1["gene_id"] = r1["gene_id"].astype(str)
    r1_sets = r1.groupby("disease_id")["gene_id"].apply(set).to_dict()

    N = len(genes)

    def best_enrichment(module: set[str]) -> float:
        target = {g for g in module if g in index}
        if len(target) < 5:
            return np.nan
        pvals = []
        for community in communities:
            k = len(community)
            x = len(target & community)
            if x == 0:
                continue
            pvals.append(hypergeom.sf(x - 1, N, k, len(target)))
        if not pvals:
            return np.nan
        return float(multipletests(pvals, method="fdr_bh")[1].min())

    rows = []
    for disease, module_id in largest.items():
        if disease not in r1_sets:
            continue
        rows.append({"disease_id": disease,
                     "q_base": best_enrichment(set(list(base_sets.get(module_id, set()))[:25])),
                     "q_r1": best_enrichment(r1_sets[disease])})
    table = pd.DataFrame(rows)
    out = DIR / f"outputs/heart_communities/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "enrichment.parquet", index=False)

    paired = table.dropna()
    payload = {"fold": args.fold, "threshold": args.threshold, "n": int(len(paired)),
               "communities": len(communities), "graph_genes": len(genes),
               "base_median": float(paired.q_base.median()) if len(paired) else None,
               "r1_median": float(paired.q_r1.median()) if len(paired) else None,
               "r1_better": int((paired.q_r1 < paired.q_base).sum()) if len(paired) else 0,
               "p": float(wilcoxon(paired.q_r1, paired.q_base).pvalue) if len(paired) > 5 else None}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
