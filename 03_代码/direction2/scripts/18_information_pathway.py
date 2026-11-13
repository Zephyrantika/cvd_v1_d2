"""检验"信息通路"假设：把表型—基因层从"只用训练疾病"扩到"除目标疾病外的全部疾病"，看 MRR 是否改善。

两个变体，除 hpo_gene 边以外所有输入完全相同：
    train_only   只用 fold 0 的 270 个训练疾病构建表型—基因边（现状）
    relaxed      用全部 452 个疾病减去 26 个目标疾病构建（目标疾病自己的边仍然被排除，
                 但同家族、表型相近的疾病可以贡献锚点）

如果 relaxed 的 MRR 明显更高，说明瓶颈是"训练集里没有表型邻居"；如果不动，说明锚点数量不是关键。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}


def metrics(rankings: dict[str, np.ndarray]) -> tuple[float, float]:
    mrr, recall = [], []
    for ranks in rankings.values():
        if len(ranks) == 0:
            continue
        mrr.append(1.0 / ranks.min())
        recall.append(float((ranks <= 10).any()))
    return float(np.mean(mrr)), float(np.mean(recall))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "information_pathway.log", "18_information_pathway")
    started = time.time()
    out_dir = paths.out("pathway")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.data.fold_builder import build_train_phenotype_gene_edges
    from phenotype_network_v0.graph.graph_builder import build_heterogeneous_graph

    processed = paths.v0("data/processed")
    folds_dir = paths.v0("data/folds")
    common = {
        "hpo_nodes": pd.read_parquet(processed / "hpo_nodes.parquet"),
        "hpo_edges": pd.read_parquet(processed / "hpo_edges.parquet"),
        "candidates": pd.read_parquet(folds_dir / "candidate_gene_universe.parquet"),
        "string_edges": pd.read_parquet(processed / "string_gene_edges_700.parquet"),
        "reactome_gene_pathway": pd.read_parquet(processed / "reactome_gene_pathway.parquet"),
        "pathways": pd.read_parquet(processed / "reactome_pathways.parquet"),
        "pathway_hierarchy": pd.read_parquet(processed / "reactome_pathway_hierarchy.parquet"),
        "relation_budget": BUDGET,
    }

    phenotypes = pd.read_parquet(processed / "cardiovascular_disease_phenotypes.parquet")
    genes = pd.read_parquet(processed / "cardiovascular_disease_genes.parquet")
    hpo_ic = pd.read_parquet(folds_dir / f"fold_{args.fold}" / "train_hpo_ic.parquet")
    train_ids = [
        line.strip()
        for line in (folds_dir / f"fold_{args.fold}" / "train_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_ids = [
        line.strip()
        for line in (folds_dir / f"fold_{args.fold}" / "test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    known = (
        genes.assign(disease_id=genes["disease_id"].astype(str), gene_id=genes["gene_id"].astype(str))
        .drop_duplicates()[["disease_id", "gene_id"]]
        .groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s))).to_dict()
    )
    targets = [d for d in test_ids if len(known.get(d, [])) >= 3]
    log.info("targets", n=len(targets))

    variants = {
        "train_only": set(train_ids),
        "relaxed": set(known) - set(targets),
    }

    rows = []
    for name, train_set in variants.items():
        edges = build_train_phenotype_gene_edges(
            train_set, phenotypes, genes, hpo_ic, 0.99,
            use_frequency=True, use_ic=True,
        )
        graph = build_heterogeneous_graph(hpo_gene_edges=edges, **common)
        node_map = graph.node_map
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        gene_ids = gene_rows["node_id"].astype(str).tolist()
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)

        seeds, _ = v0_rwr.build_seed_matrix(tuple(targets), phenotypes, hpo_ic, node_map)
        result = v0_rwr.run_rwr(graph.transition, seeds, alpha=0.2)

        rankings: dict[str, np.ndarray] = {}
        for column, disease in enumerate(targets):
            scores = result.scores[gene_indices, column]
            order = np.argsort(-scores, kind="stable")
            position = {gene_ids[idx]: rank for rank, idx in enumerate(order)}
            ranks = np.array([position[g] + 1 for g in known[disease] if g in position])
            rankings[disease] = ranks
        mrr, recall = metrics(rankings)
        rows.append({
            "variant": name,
            "n_train_diseases": len(train_set),
            "n_hpo_gene_edges": int(len(edges)),
            "n_anchor_genes": int(edges["gene_id"].nunique()) if "gene_id" in edges.columns else None,
            "n_targets": len(targets),
            "mrr": round(mrr, 4),
            "recall_at_10": round(recall, 4),
            "converged": bool(result.converged),
        })
        log.info("variant_done", variant=name, edges=len(edges), mrr=round(mrr, 4))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"information_pathway_fold_{args.fold}.parquet", index=False)
    payload = {"fold": args.fold, "alpha": 0.2, "targets": len(targets),
               "rows": rows, "seconds": round(time.time() - started, 2)}
    (out_dir / f"information_pathway_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
