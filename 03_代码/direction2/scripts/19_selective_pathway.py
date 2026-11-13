"""信息通路修正实验：放宽划分 + 只并入高分表型邻居。

三个变体，都在 fold 0 的同一批测试疾病上评 MRR：
    baseline    现状（表型—基因边只用 270 个训练疾病）
    relaxed_all 用全部 452 个疾病减去 90 个测试疾病（广撒网）
    selective   训练疾病 + 每个测试疾病的表型邻居（Jaccard ≥ 0.2），邻居本身不进测试集

注意：selective 和 relaxed_all 都让"其它疾病的表型—基因关系"进入了图，
其中包括同一折里的其它测试疾病，所以这是**上界实验**，不是可直接部署的设置。
它回答的问题是"如果训练集里有表型邻居，MRR 能到多少"。
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


def mrr_from_scores(scores: np.ndarray, gene_ids: list[str], targets: list[str], known: dict[str, set[str]]) -> float:
    values = []
    for column, disease in enumerate(targets):
        order = np.argsort(-scores[:, column], kind="stable")
        position = {gene_ids[idx]: rank for rank, idx in enumerate(order)}
        ranks = [position[g] + 1 for g in known[disease] if g in position]
        if ranks:
            values.append(float(np.mean([1.0 / r for r in ranks])))
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--jaccard", type=float, default=0.2)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "selective_pathway.log", "19_selective_pathway")
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
    test_ids = [
        line.strip()
        for line in (folds_dir / f"fold_{args.fold}" / "test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    known = (
        genes.assign(disease_id=genes["disease_id"].astype(str), gene_id=genes["gene_id"].astype(str))
        .drop_duplicates().groupby("disease_id")["gene_id"].apply(lambda s: set(s)).to_dict()
    )
    targets = [d for d in test_ids if known.get(d)]
    log.info("targets", n=len(targets))

    sets = {d: set(s) for d, s in phenotypes.groupby("disease_id")["hpo_id"].apply(
        lambda s: set(map(str, s))).items()}

    # 每个测试疾病的表型邻居
    neighbours: set[str] = set()
    for disease in targets:
        target_set = sets.get(disease, set())
        for other, other_set in sets.items():
            if other == disease or other in targets:
                continue
            union = target_set | other_set
            if union and len(target_set & other_set) / len(union) >= args.jaccard:
                neighbours.add(other)
    log.info("neighbours", n=len(neighbours))

    variants = {
        "relaxed_all": set(sets) - set(targets),
        "selective": set(sets) - set(targets) - (set(sets) - set(targets) - neighbours),
    }
    # selective = 训练集 ∪ 邻居（均为非目标疾病）
    train_ids = set(
        line.strip()
        for line in (folds_dir / f"fold_{args.fold}" / "train_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    variants["selective"] = train_ids | (neighbours - set(targets))

    rows = []
    baseline = paths.v0(f"data/graphs/fold_{args.fold}")
    from scipy.sparse import load_npz

    node_map = pd.read_parquet(baseline / "node_map.parquet")
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
    seeds, _ = v0_rwr.build_seed_matrix(tuple(targets), phenotypes, hpo_ic, node_map)
    transition = load_npz(baseline / "transition_matrix.npz").tocsr()
    result = v0_rwr.run_rwr(transition, seeds, alpha=0.2)
    rows.append({"variant": "baseline", "n_edge_diseases": len(train_ids),
                 "n_hpo_gene_edges": None, "mrr": round(mrr_from_scores(result.scores[gene_indices, :], gene_ids, targets, known), 4)})
    log.info("baseline_done", mrr=rows[-1]["mrr"])

    for name, train_set in variants.items():
        edges = build_train_phenotype_gene_edges(train_set, phenotypes, genes, hpo_ic, 0.99)
        graph = build_heterogeneous_graph(hpo_gene_edges=edges, **common)
        node_map = graph.node_map
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        gene_ids = gene_rows["node_id"].astype(str).tolist()
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        seeds, _ = v0_rwr.build_seed_matrix(tuple(targets), phenotypes, hpo_ic, node_map)
        result = v0_rwr.run_rwr(graph.transition, seeds, alpha=0.2)
        score = mrr_from_scores(result.scores[gene_indices, :], gene_ids, targets, known)
        rows.append({"variant": name, "n_edge_diseases": len(train_set),
                     "n_hpo_gene_edges": int(len(edges)), "mrr": round(score, 4)})
        log.info("variant_done", variant=name, mrr=round(score, 4), edges=len(edges))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"selective_pathway_fold_{args.fold}.parquet", index=False)
    payload = {"fold": args.fold, "jaccard_threshold": args.jaccard, "targets": len(targets),
               "neighbours": len(neighbours), "rows": rows,
               "seconds": round(time.time() - started, 2),
               "note": "relaxed_all 与 selective 都含测试集其它疾病的标签，属上界实验"}
    (out_dir / f"selective_pathway_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
