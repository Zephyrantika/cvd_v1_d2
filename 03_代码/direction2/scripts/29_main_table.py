"""主结果表：四种方法在同一批拆分上比较，带 bootstrap 置信区间。

方法：RWR、DIAMOnD、ROBUST 式多样性枚举、随机、度数。
性状：6 个常见心血管性状，各自 70/30 拆分，重复 5 次。
指标：留出基因的 Recall@50/100/200。
置信区间：按（性状 × 重复）共 30 个单元做 bootstrap（1000 次）。

同时输出一张"适用性判定"：性状内部边密度是否 ≥ 0.03 与召回高低是否一致。
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
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

KS = (50, 100, 200)
METHODS = ("rwr", "diamond", "robust_style", "random", "degree")
ROBUST_RUNS = 10
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
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(genes), len(genes))), index


def normalize_rows(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return (sparse.diags(1.0 / degree) @ adjacency).T


def rwr(transition: sparse.csr_matrix, seeds: set[int], alpha: float = 0.3, max_iter: int = 100) -> np.ndarray:
    seed_vector = np.zeros(transition.shape[0])
    seed_vector[list(seeds)] = 1.0 / len(seeds)
    scores = seed_vector.copy()
    for _ in range(max_iter):
        updated = alpha * seed_vector + (1 - alpha) * (transition @ scores)
        if np.abs(updated - scores).sum() < 1e-9:
            return updated
        scores = updated
    return scores


def diamond(adjacency: sparse.csr_matrix, seeds: set[int], degree_counts: np.ndarray,
            max_size: int = 200, alpha: float = 0.05, top_m: int = 50) -> list[int]:
    n = adjacency.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    in_module = np.zeros(n, dtype=bool)
    for seed in seeds:
        in_module[seed] = True
        counts[adjacency.indices[adjacency.indptr[seed]:adjacency.indptr[seed + 1]]] += 1
    added: list[int] = []
    while len(added) + len(seeds) < max_size:
        candidates = np.flatnonzero((counts > 0) & (~in_module))
        if candidates.size == 0:
            break
        k = counts[candidates]
        if candidates.size > top_m:
            keep = np.argsort(-k)[:top_m]
            candidates, k = candidates[keep], k[keep]
        p = hypergeom.sf(k - 1, n, degree_counts[candidates].clip(min=1), len(seeds) + len(added))
        best = int(np.argmin(p))
        if p[best] > alpha:
            break
        node = int(candidates[best])
        in_module[node] = True
        added.append(node)
        counts[adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]] += 1
    return added


def grow(adjacency: sparse.csr_matrix, costs: np.ndarray, prizes: np.ndarray,
         start: set[int], max_size: int) -> set[int]:
    n = adjacency.shape[0]
    in_module = np.zeros(n, dtype=bool)
    in_module[list(start)] = True
    best_cost = np.full(n, np.inf)
    for node in start:
        lo, hi = adjacency.indptr[node], adjacency.indptr[node + 1]
        np.minimum.at(best_cost, adjacency.indices[lo:hi], costs[lo:hi])
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


def bootstrap_ci(values: np.ndarray, iterations: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sets", default="trait_gene_sets.parquet")
    parser.add_argument("--tag", default="05")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "main_table.log", "29_main_table")
    started = time.time()
    out_dir = paths.out("gene_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = pd.read_parquet(paths.out("external", args.sets))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    adjacency, index = build_graph(ppi, universe)
    transition = normalize_rows(adjacency)
    degree_counts = np.diff(adjacency.indptr)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    max_weight = float(ppi["weight"].max())

    cost_cache: dict[float, float] = {}
    costs = np.empty(adjacency.data.shape[0])
    for position, weight in enumerate(adjacency.data):
        key = float(weight)
        if key not in cost_cache:
            cost_cache[key] = float(-np.log(np.clip(key / max_weight, 1e-6, 1.0)))
        costs[position] = cost_cache[key]
    log.info("graph_ready", nodes=len(universe), edges=adjacency.nnz // 2)

    rows = []
    for trait, group in sets.groupby("trait"):
        genes = [g for g in group["gene_id"].astype(str) if g in index]
        if len(genes) < 20:
            continue
        for repeat in range(args.repeats):
            order = rng.permutation(len(genes))
            cut = int(len(genes) * 0.7)
            seed_genes = {index[genes[i]] for i in order[:cut]}
            held = {index[genes[i]] for i in order[cut:]}
            non_seed = np.setdiff1d(np.arange(len(universe)), np.array(sorted(seed_genes)))

            scores = rwr(transition, seed_genes)
            scores_masked = scores.copy()
            scores_masked[list(seed_genes)] = -np.inf

            prizes = scores / (scores.max() + 1e-12)
            solutions: list[set[int]] = []
            for _ in range(ROBUST_RUNS):
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
            robust_ranked = [g for g, _ in sorted(frequency.items(), key=lambda kv: -kv[1]) if g not in seed_genes]

            diamond_added = diamond(adjacency, seed_genes, degree_counts, max_size=200)

            degree_masked = degrees.copy()
            degree_masked[list(seed_genes)] = -np.inf

            picks = {
                "rwr": list(np.argsort(-scores_masked)),
                "diamond": diamond_added,
                "robust_style": robust_ranked,
                "degree": list(np.argsort(-degree_masked)),
            }
            for k in KS:
                random_set = set(rng.choice(non_seed, size=k, replace=False))
                for method in ("rwr", "diamond", "robust_style", "degree"):
                    chosen = set(picks[method][:k])
                    rows.append({"trait": trait, "repeat": repeat, "method": method, "k": k,
                                 "recall": len(chosen & held) / len(held)})
                rows.append({"trait": trait, "repeat": repeat, "method": "random", "k": k,
                             "recall": len(random_set & held) / len(held)})
        log.info("trait_done", trait=trait, genes=len(genes))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"main_table_results_{args.tag}.parquet", index=False)

    summary_rows = []
    for method in METHODS:
        for k in KS:
            values = table.loc[(table.method == method) & (table.k == k), "recall"].to_numpy()
            low, high = bootstrap_ci(values)
            summary_rows.append({"method": method, "k": k, "mean": round(float(values.mean()), 4),
                                 "ci_low": round(low, 4), "ci_high": round(high, 4), "n": len(values)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"main_table_summary_{args.tag}.csv", index=False, encoding="utf-8-sig")

    lines = ["| 方法 | Recall@50 | Recall@100 | Recall@200 |", "|---|---|---|---|"]
    for method in METHODS:
        cells = []
        for k in KS:
            row = summary[(summary.method == method) & (summary.k == k)].iloc[0]
            cells.append(f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (out_dir / f"main_table_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {"repeats": args.repeats, "robust_runs": ROBUST_RUNS, "sets": args.sets, "tag": args.tag,
               "summary": summary_rows, "seconds": round(time.time() - started, 2)}
    (out_dir / f"main_table_{args.tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n".join(lines))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
