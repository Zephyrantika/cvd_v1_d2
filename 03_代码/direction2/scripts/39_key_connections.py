"""E9 关键连接（文档 §4.6）。

对每个抽样模块：在模块诱导子图上算边介数，取前 10 条边作为关键连接；
分别删掉关键边与随机 10 条内部边，重跑 RWR，比较模块成员的平均分位下降。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

TOP_EDGES = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modules", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "key_connections.log", "39_key_connections")
    started = time.time()
    out_dir = paths.out("key_connections", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    gene_ids = matrix.gene_ids
    index = {g: i for i, g in enumerate(gene_ids)}
    rows, cols, data = [], [], []
    edges_by_pair = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        rows += [ia, ib]
        cols += [ib, ia]
        data += [w, w]
        edges_by_pair[(ia, ib)] = float(w)
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(gene_ids), len(gene_ids)))
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
    BUDGET = {
        "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
        "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
                 "gene_participates_in_pathway": 0.25},
        "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
    }
    graph_dir = paths.v0(f"data/graphs/fold_{args.fold}")
    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr() for p in sorted(graph_dir.glob("A_*.npz"))}
    transition = compose_transition_matrix(relations, BUDGET).tocsr()

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_node = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                        node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(map(str, s))).to_dict()

    modules = pd.read_parquet(paths.out("modules_doc", "modules.parquet"))
    members = pd.read_parquet(paths.out("modules_doc", "members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    candidates = modules.sample(n=min(args.modules, len(modules)), random_state=args.seed)

    def seed_vector(disease: str) -> np.ndarray:
        vector = np.zeros(transition.shape[0])
        indices = [hpo_node[t] for t in terms_of.get(disease, set()) if t in hpo_node]
        if not indices:
            return vector
        unique = np.unique(indices)
        vector[unique] = 1.0 / len(unique)
        return vector

    rows_out = []
    for row in candidates.itertuples():
        genes = [g for g in members.loc[members.module_id == row.module_id, "gene_id"] if g in index]
        if len(genes) < 6:
            continue
        idx = [index[g] for g in genes]
        sub_edges = [(a, b) for a in idx for b in idx if a < b and (a, b) in edges_by_pair]
        if len(sub_edges) < TOP_EDGES + 2:
            continue
        graph = ig.Graph(n=len(gene_ids), edges=sub_edges, directed=False)
        betweenness = graph.edge_betweenness()
        order = np.argsort(betweenness)[::-1][:TOP_EDGES]
        key_edges = [sub_edges[i] for i in order]
        random_edges = [sub_edges[i] for i in rng.choice(len(sub_edges), size=TOP_EDGES, replace=False)]

        seeds = seed_vector(row.disease_id)
        if seeds.sum() == 0:
            continue
        base = v0_rwr.run_rwr(transition, seeds[:, None], alpha=0.3).scores[gene_indices := np.array(idx), 0]

        def drop(removed):
            trans = transition.tolil(copy=True)
            touched = set()
            for a, b in removed:
                na, nb = int(gene_node[a]), int(gene_node[b])
                trans[na, nb] = 0.0
                trans[nb, na] = 0.0
                touched.update((na, nb))
            for node in touched:
                values = np.asarray(trans.data[node], dtype=float)
                total = values.sum()
                if total > 0:
                    trans.data[node] = (values / total).tolist()
                else:
                    trans[node, node] = 1.0
            trans = trans.tocsr()
            scores = v0_rwr.run_rwr(trans, seeds[:, None], alpha=0.3).scores[gene_indices, 0]
            return float(np.mean(scores) - np.mean(base))

        key_drop = drop(key_edges)
        random_drop = drop(random_edges)
        rows_out.append({
            "module_id": row.module_id,
            "disease_id": row.disease_id,
            "size": len(genes),
            "n_internal_edges": len(sub_edges),
            "drop_key_edges": key_drop,
            "drop_random_edges": random_drop,
            "key_minus_random": key_drop - random_drop,
        })
        log.info("module_done", module_id=row.module_id)

    table = pd.DataFrame(rows_out)
    table.to_parquet(out_dir / "key_connections.parquet", index=False)
    summary = {
        "fold": args.fold,
        "modules": int(len(table)),
        "drop_key_median": round(float(table["drop_key_edges"].median()), 6) if len(table) else None,
        "drop_random_median": round(float(table["drop_random_edges"].median()), 6) if len(table) else None,
        "key_more_negative": int((table["drop_key_edges"] < table["drop_random_edges"]).sum()) if len(table) else 0,
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "key_connections.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
