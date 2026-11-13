"""E7 共享/特异模块（文档 §4.6）。

同折内疾病两两算模块基因集的最大 Jaccard，建疾病—疾病相似图，Leiden 社区发现；
社区内基因按出现在多少疾病里分成共享核心（≥3 疾病）与特异核心（1–2 疾病）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

SIM_THRESHOLD = 0.15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="v0", choices=["v0","consensus"])
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "shared_specific.log", "38_shared_specific")
    started = time.time()
    out_dir = paths.out("shared", f"fold_{args.fold}", args.source)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "consensus":
        members = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"),
                                  columns=["module_id", "gene_id", "disease_id"])
        members["gene_id"] = members["gene_id"].astype(str)
        members["disease_id"] = members["disease_id"].astype(str)
    else:
        modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"))
        members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"),
                                  columns=["module_id", "gene_id", "disease_id"])
        members["gene_id"] = members["gene_id"].astype(str)
        members["disease_id"] = members["disease_id"].astype(str)
        members = members[members["disease_id"].isin(set(modules.loc[modules.fold == args.fold, "disease_id"].astype(str)))]

    module_genes = members.groupby("module_id")["gene_id"].apply(set).to_dict()
    disease_modules = members.groupby("disease_id")["module_id"].apply(list).to_dict()
    diseases = sorted(disease_modules)
    log.info("diseases", n=len(diseases))

    def max_jaccard(a: str, b: str) -> float:
        best = 0.0
        for ma in disease_modules[a]:
            ga = module_genes[ma]
            for mb in disease_modules[b]:
                gb = module_genes[mb]
                union = ga | gb
                if union:
                    best = max(best, len(ga & gb) / len(union))
        return best

    similarities = []
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            similarities.append((a, b, max_jaccard(a, b)))
    sim_frame = pd.DataFrame(similarities, columns=["disease_a", "disease_b", "jaccard"])
    sim_frame.to_parquet(out_dir / "disease_pair_similarity.parquet", index=False)

    edges = [(a, b) for a, b, value in similarities if value >= SIM_THRESHOLD]
    graph = ig.Graph(n=len(diseases), edges=[(diseases.index(a), diseases.index(b)) for a, b in edges], directed=False)
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=args.seed)
    clusters = {diseases[i]: int(cluster) for cluster, group in enumerate(partition) for i in group}
    log.info("clusters", n=len(partition), sizes=[len(g) for g in partition])

    gene_diseases: dict[str, set[str]] = {}
    for disease, module_ids in disease_modules.items():
        for module_id in module_ids:
            for gene in module_genes[module_id]:
                gene_diseases.setdefault(gene, set()).add(disease)

    rows = []
    for gene, owners in gene_diseases.items():
        clusters_of_owners = {clusters[d] for d in owners}
        rows.append({
            "gene_id": gene,
            "n_diseases": len(owners),
            "n_clusters": len(clusters_of_owners),
            "role": "shared" if len(owners) >= 3 else "specific",
        })
    genes = pd.DataFrame(rows)
    genes.to_parquet(out_dir / "gene_shared_specific.parquet", index=False)

    cluster_frame = pd.DataFrame({"disease_id": list(clusters), "cluster": [clusters[d] for d in clusters]})
    cluster_frame.to_parquet(out_dir / "disease_clusters.parquet", index=False)

    summary = {
        "fold": args.fold,
        "diseases": len(diseases),
        "clusters": int(len(partition)),
        "cluster_sizes": sorted([len(g) for g in partition], reverse=True)[:10],
        "genes_total": int(len(genes)),
        "shared_genes": int((genes["role"] == "shared").sum()),
        "specific_genes": int((genes["role"] == "specific").sum()),
        "frac_shared": round(float((genes["role"] == "shared").mean()), 3),
        "pairs_ge_threshold": int(len(edges)),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "shared_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
