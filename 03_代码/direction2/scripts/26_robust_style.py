"""ROBUST 式多样性枚举（替代实现，向量化版）。

官方 ROBUST 依赖 ILP 求解器，本机装不上。这里实现同一套思路：
    节点 prize = 从种子出发的 RWR 分数（归一化），边 cost = -log(规范化权重)；
    每次从随机子集种子出发做贪心生长，加入"prize 减去连接成本"最大的节点；
    跑 K 次，保留彼此 Jaccard < 0.7 的解；按出现频率排序，取前 K 个非种子基因。

与 DIAMOnD / RWR 在同一批拆分上比较，指标是留出基因的 Recall@K。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

KS = (50, 100, 200)
RUNS = 15
DIVERSITY = 0.85


def build_graph(ppi: pd.DataFrame, genes: list[str]):
    index = {g: i for i, g in enumerate(genes)}
    rows, cols, data = [], [], []
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        rows += [ia, ib]
        cols += [ib, ia]
        data += [w, w]
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(len(genes), len(genes)))
    return adjacency, index


def rwr_scores(adjacency: sparse.csr_matrix, seeds: set[int], alpha: float = 0.3, max_iter: int = 100) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    transition = (sparse.diags(1.0 / degree) @ adjacency).T
    seed_vector = np.zeros(adjacency.shape[0])
    seed_vector[list(seeds)] = 1.0 / len(seeds)
    scores = seed_vector.copy()
    for _ in range(max_iter):
        updated = alpha * seed_vector + (1 - alpha) * (transition @ scores)
        if np.abs(updated - scores).sum() < 1e-9:
            return updated
        scores = updated
    return scores


def grow(adjacency: sparse.csr_matrix, costs: np.ndarray, prizes: np.ndarray,
         start: set[int], max_size: int) -> set[int]:
    n = adjacency.shape[0]
    in_module = np.zeros(n, dtype=bool)
    in_module[list(start)] = True
    best_cost = np.full(n, np.inf)
    for node in start:
        lo, hi = adjacency.indptr[node], adjacency.indptr[node + 1]
        neighbours = adjacency.indices[lo:hi]
        values = costs[lo:hi]
        np.minimum.at(best_cost, neighbours, values)

    module = set(start)
    while len(module) < max_size:
        score = prizes - 0.5 * np.where(np.isfinite(best_cost), best_cost, 1e9)
        score[in_module] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]) or score[best] <= 0:
            break
        in_module[best] = True
        module.add(best)
        lo, hi = adjacency.indptr[best], adjacency.indptr[best + 1]
        np.minimum.at(best_cost, adjacency.indices[lo:hi], costs[lo:hi])
    return module


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "robust_style.log", "26_robust_style")
    started = time.time()
    out_dir = paths.out("gene_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = pd.read_parquet(paths.out("external", "trait_gene_sets.parquet"))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    adjacency, index = build_graph(ppi, universe)
    max_weight = float(ppi["weight"].max())

    # 每条边的成本，按邻居顺序存成与 data 对齐的数组
    weight_to_cost = {}
    costs = np.empty(adjacency.data.shape[0], dtype=float)
    for position, weight in enumerate(adjacency.data):
        key = float(weight)
        value = weight_to_cost.get(key)
        if value is None:
            value = float(-np.log(np.clip(key / max_weight, 1e-6, 1.0)))
            weight_to_cost[key] = value
        costs[position] = value
    log.info("graph_ready", nodes=len(universe), edges=adjacency.nnz // 2)

    rows = []
    for trait in sorted(sets["trait"].unique()):
        genes = [g for g in sets.loc[sets["trait"] == trait, "gene_id"].astype(str) if g in index]
        if len(genes) < 20:
            continue
        for repeat in range(args.repeats):
            order = rng.permutation(len(genes))
            cut = int(len(genes) * 0.7)
            seed_genes = {index[genes[i]] for i in order[:cut]}
            held = {index[genes[i]] for i in order[cut:]}

            prizes = rwr_scores(adjacency, seed_genes)
            prizes = prizes / (prizes.max() + 1e-12)

            solutions: list[set[int]] = []
            for _ in range(RUNS):
                start = set(rng.choice(np.array(sorted(seed_genes)), size=max(3, len(seed_genes) // 3),
                                       replace=False).tolist())
                jitter = prizes * rng.lognormal(0.0, 0.25, size=prizes.shape[0])
                jitter = jitter / (jitter.max() + 1e-12)
                candidate = grow(adjacency, costs, jitter, start, max_size=200)
                if all(jaccard(candidate, existing) < DIVERSITY for existing in solutions):
                    solutions.append(candidate)

            frequency: dict[int, int] = {}
            for solution in solutions:
                for gene in solution:
                    frequency[gene] = frequency.get(gene, 0) + 1
            ranked = [g for g, _ in sorted(frequency.items(), key=lambda kv: -kv[1]) if g not in seed_genes]

            for k in KS:
                top = set(ranked[:k])
                rows.append({"trait": trait, "repeat": repeat, "method": "robust_style", "k": k,
                             "recall": len(top & held) / len(held), "n_solutions": len(solutions)})
        log.info("trait_done", trait=trait, genes=len(genes))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "robust_style_results.parquet", index=False)
    summary = table.groupby("k").agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                                     solutions=("n_solutions", "mean")).round(4).reset_index()
    payload = {"repeats": args.repeats, "runs": RUNS, "diversity": DIVERSITY,
               "table": summary.to_dict(orient="records"), "seconds": round(time.time() - started, 2)}
    (out_dir / "robust_style_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
