"""转向后的第一个实验：基因种子驱动的模块扩展。

任务：给定某个常见心血管性状的一部分已知基因（70%）作为种子，在 STRING 700 上扩展出一个模块，
看能不能把留出的 30% 找回来。

对照：
    rwr_gene   以种子基因为初始分布，在 PPI 上跑 RWR，按分数取前 K
    diamond    从种子出发按超几何显著性贪心生长（同 15 号脚本的实现）
    degree     直接取度数最高的 K 个基因（度数基线）
    random     随机取 K 个基因（下界）

指标：留出基因的 Recall@K（K=50/100/200），10 次随机拆分取平均。
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

TRAITS = {
    "coronary_artery_disease": "MONDO_0005010",
    "myocardial_infarction": "MONDO_0005068",
    "heart_failure": "MONDO_0005252",
    "atrial_fibrillation": "MONDO_0004981",
    "hypertension": "MONDO_0005044",
    "ldl_cholesterol": "EFO_0004611",
}
SCORE_THRESHOLD = 0.5
KS = (50, 100, 200)


def build_adjacency(ppi: pd.DataFrame, genes: list[str]) -> tuple[sparse.csr_matrix, dict[str, int]]:
    index = {gene: i for i, gene in enumerate(genes)}
    rows, cols, data = [], [], []
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        rows += [ia, ib]
        cols += [ib, ia]
        data += [w, w]
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(genes), len(genes)))
    return matrix, index


def rwr(adjacency: sparse.csr_matrix, seeds: np.ndarray, alpha: float = 0.3, max_iter: int = 200) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    transition = sparse.diags(1.0 / degree) @ adjacency
    scores = seeds.copy()
    for _ in range(max_iter):
        updated = alpha * seeds + (1 - alpha) * (transition.T @ scores)
        if np.abs(updated - scores).sum() < 1e-10:
            scores = updated
            break
        scores = updated
    return scores


def diamond(adjacency: sparse.csr_matrix, seeds: set[int], degree_counts: np.ndarray,
            max_size: int = 200, alpha: float = 0.05, top_m: int = 50) -> list[int]:
    """增量维护候选基因与模块的连接数，每一步只在连接数最高的 top_m 个候选里做检验。"""
    n_nodes = adjacency.shape[0]
    counts = np.zeros(n_nodes, dtype=np.int32)
    in_module = np.zeros(n_nodes, dtype=bool)
    for seed in seeds:
        in_module[seed] = True
    for seed in seeds:
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
        sizes = degree_counts[candidates].clip(min=1)
        p_values = hypergeom.sf(k - 1, n_nodes, sizes, len(seeds) + len(added))
        best_index = int(np.argmin(p_values))
        if p_values[best_index] > alpha:
            break
        best = int(candidates[best_index])
        in_module[best] = True
        added.append(best)
        counts[adjacency.indices[adjacency.indptr[best]:adjacency.indptr[best + 1]]] += 1
    return added

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "gene_seeded.log", "25_gene_seeded")
    started = time.time()
    out_dir = paths.out("gene_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)

    sets_path = paths.out("external", "trait_gene_sets.parquet")
    if not sets_path.exists():
        print("missing trait_gene_sets.parquet; run 24 first", file=sys.stderr)
        return 1
    sets = pd.read_parquet(sets_path)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    adjacency, index = build_adjacency(ppi, universe)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_counts = np.diff(adjacency.indptr)  # 超几何检验要用邻居个数，不是加权和

    rng = np.random.default_rng(args.seed)
    rows = []
    for trait in sets["trait"].unique():
        genes = sets.loc[sets["trait"] == trait, "gene_id"].astype(str).tolist()
        genes = [g for g in genes if g in index]
        if len(genes) < 20:
            continue
        for repeat in range(args.repeats):
            order = rng.permutation(len(genes))
            cut = int(len(genes) * 0.7)
            seed_genes = {index[genes[i]] for i in order[:cut]}
            held = {index[genes[i]] for i in order[cut:]}
            seed_vector = np.zeros(len(universe))
            seed_vector[list(seed_genes)] = 1.0 / len(seed_genes)

            scores = rwr(adjacency, seed_vector)
            diamond_added = diamond(adjacency, seed_genes, degree_counts, max_size=max(KS))
            degree_top = np.argsort(-degrees)

            # 三种方法都要把种子排除掉再取前 K，否则 RWR 的前几名全是种子，比较不公平
            seed_list = list(seed_genes)
            masked_scores = scores.copy()
            masked_scores[seed_list] = -np.inf
            masked_degree = degrees.copy()
            masked_degree[seed_list] = -np.inf
            non_seed = np.setdiff1d(np.arange(len(universe)), np.array(seed_list))

            for k in KS:
                rwr_top = set(np.argsort(-masked_scores)[:k])
                diamond_top = set(diamond_added[:k])
                degree_set = set(np.argsort(-masked_degree)[:k])
                random_set = set(rng.choice(non_seed, size=k, replace=False))
                for method, chosen in (("rwr_gene", rwr_top), ("diamond", diamond_top),
                                       ("degree", degree_set), ("random", random_set)):
                    rows.append({
                        "trait": trait,
                        "repeat": repeat,
                        "method": method,
                        "k": k,
                        "recall": len(chosen & held) / len(held),
                        "n_seed": len(seed_genes),
                        "n_held": len(held),
                    })
        log.info("trait_done", trait=trait, genes=len(genes))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "gene_seeded_results.parquet", index=False)
    summary = (
        table.groupby(["method", "k"])
        .agg(recall_mean=("recall", "mean"), recall_sd=("recall", "std"))
        .round(4)
        .reset_index()
    )
    summary.to_csv(out_dir / "gene_seeded_summary.csv", index=False, encoding="utf-8-sig")
    per_trait = (
        table.groupby(["trait", "method", "k"])["recall"].mean().round(4)
        .unstack(["method", "k"]).reset_index()
    )
    per_trait.to_csv(out_dir / "gene_seeded_by_trait.csv", index=False, encoding="utf-8-sig")
    payload = {
        "traits": sorted(sets["trait"].unique().tolist()),
        "score_threshold": SCORE_THRESHOLD,
        "repeats": args.repeats,
        "table": summary.to_dict(orient="records"),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "gene_seeded_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
