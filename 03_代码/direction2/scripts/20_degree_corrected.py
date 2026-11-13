"""度校正传播：把行随机游走换成对称归一化再加权，看 MRR 能不能上去。

为什么这样做：行随机矩阵 P = D⁻¹A 的稳态分布正比于度数，分数天然偏向 hub。
对称归一化 S = D^(−1/2) A D^(−1/2) 的稳态是均匀分布，hub 的吸力被压掉。
再按行归一化保持随机游走的语义，其余（种子、α、候选集、评价）全不动。

变体：
    baseline    现有转移矩阵（行随机）
    sym         对称归一化后行归一化
    theta25     边权按 1/(d_i^0.25 · d_j^0.75) 缩放
    theta75     边权按 1/(d_i^0.75 · d_j^0.25) 缩放
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
from scipy.sparse import diags, load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

BUDGET = {
    "hpo_hpo": 0.40,
    "hpo_gene": 0.60,
    "gene_hpo": 0.15,
    "gene_gene": 0.60,
    "gene_pathway": 0.25,
    "pathway_gene": 0.70,
    "pathway_pathway": 0.30,
}


def row_normalize(matrix):
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return diags(1.0 / row_sums) @ matrix


def mrr(scores: np.ndarray, gene_ids: list[str], targets: list[str], known: dict[str, set[str]]) -> tuple[float, float]:
    values, hits = [], []
    for column, disease in enumerate(targets):
        order = np.argsort(-scores[:, column], kind="stable")
        position = {gene_ids[idx]: rank for rank, idx in enumerate(order)}
        ranks = [position[g] + 1 for g in known[disease] if g in position]
        if not ranks:
            continue
        values.append(float(np.mean([1.0 / r for r in ranks])))
        hits.append(float(min(ranks) <= 10))
    return float(np.mean(values)), float(np.mean(hits))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "degree_corrected.log", "20_degree_corrected")
    started = time.time()
    out_dir = paths.out("propagation")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    graph_dir = paths.v0(f"data/graphs/fold_{args.fold}")
    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr() for p in sorted(graph_dir.glob("A_*.npz"))}
    node_map = pd.read_parquet(graph_dir / "node_map.parquet")
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_indices = gene_rows["node_index"].to_numpy(dtype=int)

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    test_ids = [
        line.strip()
        for line in paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    genes_table = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"), columns=["disease_id", "gene_id"]
    ).drop_duplicates()
    genes_table["disease_id"] = genes_table["disease_id"].astype(str)
    genes_table["gene_id"] = genes_table["gene_id"].astype(str)
    known = genes_table.groupby("disease_id")["gene_id"].apply(lambda s: set(s)).to_dict()
    targets = [d for d in test_ids if known.get(d)]

    seeds, _ = v0_rwr.build_seed_matrix(tuple(targets), phenotypes, hpo_ic, node_map)
    alpha = 0.2

    # 组合未归一化的加权邻接
    adjacency = None
    for name, weight in BUDGET.items():
        if name not in relations:
            continue
        adjacency = relations[name] * weight if adjacency is None else adjacency + relations[name] * weight
    adjacency = adjacency.tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0

    def variant_matrix(kind: str):
        if kind == "sym":
            scale = 1.0 / np.sqrt(degree)
        elif kind == "theta25":
            scale = degree ** -0.25
        elif kind == "theta75":
            scale = degree ** -0.75
        else:
            raise ValueError(kind)
        scaled = diags(scale) @ adjacency @ diags(scale)
        return row_normalize(scaled)

    rows = []
    baseline = load_npz(graph_dir / "transition_matrix.npz").tocsr()
    result = v0_rwr.run_rwr(baseline, seeds, alpha=alpha)
    value, hit = mrr(result.scores[gene_indices, :], gene_ids, targets, known)
    rows.append({"variant": "baseline", "mrr": round(value, 4), "recall_at_10": round(hit, 4),
                 "converged": bool(result.converged), "nnz": int(baseline.nnz)})
    log.info("variant_done", variant="baseline", mrr=round(value, 4))

    for kind in ("sym", "theta25", "theta75"):
        matrix = variant_matrix(kind)
        result = v0_rwr.run_rwr(matrix, seeds, alpha=alpha)
        value, hit = mrr(result.scores[gene_indices, :], gene_ids, targets, known)
        rows.append({"variant": kind, "mrr": round(value, 4), "recall_at_10": round(hit, 4),
                     "converged": bool(result.converged), "nnz": int(matrix.nnz)})
        log.info("variant_done", variant=kind, mrr=round(value, 4))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"degree_corrected_fold_{args.fold}.parquet", index=False)
    payload = {"fold": args.fold, "alpha": alpha, "targets": len(targets),
               "rows": rows, "seconds": round(time.time() - started, 2)}
    (out_dir / f"degree_corrected_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
