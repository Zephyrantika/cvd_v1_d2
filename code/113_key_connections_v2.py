"""关键连接分析（修正口径）：模块内介数最高的若干条边 vs 随机边，删除后看模块分位与排名保持。

原 39 号脚本用原始分的均值差，量级太小被舍入成 0；这里改用分位变化与 top-200 保持率。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

TOP_EDGES = 3
BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modules", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    out = paths.out("key_connections_v2", f"fold_{args.fold}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    import importlib
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    import igraph as ig
    from scipy.sparse import csr_matrix

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    edges = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        if a in position and b in position:
            i, j = position[a], position[b]
            edges[(min(i, j), max(i, j))] = float(w)

    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr()
                 for p in sorted(paths.v0(f"data/graphs/fold_{args.fold}").glob("A_*.npz"))}
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
    transition = compose_transition_matrix(relations, BUDGET).tocsr()

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_node = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    hpo_node = dict(zip(node_map.loc[node_map["node_type"] == "phenotype", "node_id"].astype(str),
                        node_map.loc[node_map["node_type"] == "phenotype", "node_index"].astype(int)))
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(map(str, s))).to_dict()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])

    members = pd.read_parquet(paths.out("pareto_r1", f"members_fold{args.fold}.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    members["disease_id"] = members["disease_id"].astype(str)
    modules = list(members.groupby("disease_id"))
    rng.shuffle(modules)
    modules = modules[: min(args.modules, len(modules))]

    def seeds_of(disease: str) -> np.ndarray:
        vector = np.zeros(transition.shape[0])
        idx = [hpo_node[t] for t in terms_of.get(disease, set()) if t in hpo_node]
        if not idx:
            return vector
        unique = np.unique(idx)
        vector[unique] = 1.0 / len(unique)
        return vector

    def stats(scores: np.ndarray, gene_idx: np.ndarray, disease: str) -> tuple[float, float]:
        di = matrix.disease_index(disease)
        pct = percentiles[di, gene_idx]
        return float(pct.mean()), float((pct >= 1 - 200 / percentiles.shape[1]).mean())

    rows = []
    for disease, group in modules:
        genes = [g for g in group["gene_id"] if g in position]
        if len(genes) < 6:
            continue
        idx = np.array([position[g] for g in genes])
        sub = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:] if (min(a, b), max(a, b)) in edges]
        if len(sub) < TOP_EDGES + 2:
            continue
        graph = ig.Graph(n=len(idx), edges=[(i, j) for i in range(len(idx)) for j in range(i + 1, len(idx))
                                            if (min(idx[i], idx[j]), max(idx[i], idx[j])) in edges], directed=False)
        betweenness = np.array(graph.edge_betweenness())
        order = np.argsort(betweenness)[::-1][:TOP_EDGES]
        edge_list = [(int(e.source), int(e.target)) for e in graph.es]
        key_edges = [(idx[edge_list[k][0]], idx[edge_list[k][1]]) for k in order]
        pick = rng.choice(len(sub), size=TOP_EDGES, replace=False)
        random_edges = [sub[k] for k in pick]

        def lcc(graph_obj) -> float:
            comps = graph_obj.connected_components()
            return max(len(c) for c in comps) / graph_obj.vcount()

        base_lcc = lcc(graph)

        def after(removed) -> float:
            g2 = graph.copy()
            lookup = {}
            for e in g2.es:
                lookup[(int(e.source), int(e.target))] = e.index
                lookup[(int(e.target), int(e.source))] = e.index
            drop = set()
            for a, b in removed:
                ia, ib = list(idx).index(a), list(idx).index(b)
                eid = lookup.get((ia, ib))
                if eid is not None:
                    drop.add(eid)
            if drop:
                g2.delete_edges(sorted(drop))
            return lcc(g2)

        key_lcc = after(key_edges)
        rnd_lcc = after(random_edges)
        rows.append({"disease_id": disease, "size": len(genes),
                     "d_lcc_key": key_lcc - base_lcc, "d_lcc_random": rnd_lcc - base_lcc})

    table = pd.DataFrame(rows)
    table.to_parquet(out / "key_connections_v2.parquet", index=False)
    payload = {"fold": args.fold, "modules": int(len(table)), "alpha": alpha,
               "seconds": round(time.time() - started, 1)}
    if len(table) > 5:
        a, b = table["d_lcc_key"], table["d_lcc_random"]
        payload["d_lcc"] = {"key_median": round(float(a.median()), 5),
                            "random_median": round(float(b.median()), 5),
                            "key_lower": int((a < b).sum()), "n": int(len(table))}
        if (a != b).any():
            payload["d_lcc"]["p"] = float(wilcoxon(a, b).pvalue)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
