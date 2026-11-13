"""文档 §4.2/§4.3：100 次扰动 + 共现矩阵 + 层次聚类共识 + 核心/外围。

扰动四类（按文档 §3.4）：
    40 次：随机删 10% 表型术语
    30 次：随机删 10% PPI 边（共享 30 个重连后的转移矩阵）
    20 次：STRING 700 → 900
    10 次：表型权重乘噪声

每个疾病在 C_d 上得到 100 个划分，累计共现矩阵 A，用平均连接层次聚类（距离 1−A）得到共识模块；
核心 = 与模块内其它成员的平均共现 ≥0.8，外围 = 0.6–0.8；单模块覆盖 >80% 候选集判为塌缩，回退单次 Leiden。
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, load_npz
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}


def drop_seed_terms(seeds: np.ndarray, rng: np.random.Generator, fraction: float = 0.10) -> np.ndarray:
    perturbed = seeds.copy()
    nonzero = np.flatnonzero(perturbed > 0)
    if len(nonzero) <= 1:
        return perturbed
    count = min(max(1, int(np.floor(len(nonzero) * fraction))), len(nonzero) - 1)
    chosen = rng.choice(nonzero, size=count, replace=False)
    perturbed[chosen] = 0.0
    total = perturbed.sum()
    return perturbed / total if total > 0 else perturbed


def noise_seed_weights(seeds: np.ndarray, rng: np.random.Generator, sigma: float = 0.10) -> np.ndarray:
    perturbed = seeds * (1.0 + rng.normal(0.0, sigma, size=seeds.shape[0]))
    perturbed = np.clip(perturbed, 0.0, None)
    total = perturbed.sum()
    return perturbed / total if total > 0 else seeds


def drop_edges(matrix, fraction: float, rng: np.random.Generator):
    coo = matrix.tocoo()
    upper = coo.row < coo.col
    rows, cols, data = coo.row[upper], coo.col[upper], coo.data[upper]
    keep = rng.random(len(rows)) >= fraction
    rows, cols, data = rows[keep], cols[keep], data[keep]
    return csr_matrix((np.concatenate([data, data]),
                       (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                      shape=matrix.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--diseases", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="用本折全部疾病")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "consensus_100.log", "34_consensus_100")
    started = time.time()
    out_dir = paths.out("consensus", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix

    detect, _ = load_v0_models(paths.v0_root)
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    graph_dir = paths.v0(f"data/graphs/fold_{args.fold}")
    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr() for p in sorted(graph_dir.glob("A_*.npz"))}
    node_map = pd.read_parquet(graph_dir / "node_map.parquet")
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
    gene_ids = gene_rows["node_id"].astype(str).tolist()

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])

    tiers = pd.read_parquet(paths.out("audit", "disease_tiers.parquet"))
    fold_ids = [
        line.strip()
        for line in paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tier_of = dict(zip(tiers["disease_id"].astype(str), tiers["tier"]))
    sampled = []
    per_tier = {"T1": 4, "T2": 6, "T3": 10}
    for tier, count in per_tier.items():
        pool = [d for d in fold_ids if tier_of.get(d) == tier]
        sampled.extend(rng.choice(np.array(pool), size=min(count, len(pool)), replace=False).tolist())
    if getattr(args, "all", False):
        sampled = list(fold_ids)
    sampled = sorted(set(sampled))
    log.info("sampled", n=len(sampled))

    # 共享的扰动转移矩阵（与疾病无关）
    edge_transitions = []
    for _ in range(30):
        damaged = dict(relations)
        damaged["gene_gene"] = drop_edges(relations["gene_gene"], 0.10, rng)
        edge_transitions.append(compose_transition_matrix(damaged, BUDGET).tocsr())
    base_transition = compose_transition_matrix(relations, BUDGET).tocsr()
    # STRING 900：按 gene_id -> 节点下标重建
    edges900 = pd.read_parquet(paths.v0("data/processed/string_gene_edges_900.parquet"),
                               columns=["gene_a", "gene_b", "weight"])
    node_of_gene = dict(zip(gene_ids, gene_indices.tolist()))
    rows900, cols900, data900 = [], [], []
    for a, b, w in zip(edges900["gene_a"].astype(str), edges900["gene_b"].astype(str),
                       edges900["weight"].astype(float), strict=True):
        ia, ib = node_of_gene.get(a), node_of_gene.get(b)
        if ia is None or ib is None:
            continue
        rows900 += [ia, ib]
        cols900 += [ib, ia]
        data900 += [w, w]
    relations_900 = dict(relations)
    relations_900["gene_gene"] = csr_matrix((data900, (rows900, cols900)), shape=relations["gene_gene"].shape)
    transition_900 = compose_transition_matrix(relations_900, BUDGET).tocsr()
    log.info("transitions_ready", edge=len(edge_transitions))

    seeds, _ = v0_rwr.build_seed_matrix(tuple(sampled), phenotypes, hpo_ic, node_map)
    modules_by_run: dict[str, list[list[tuple[str, ...]]]] = {d: [] for d in sampled}
    collapses = {d: 0 for d in sampled}

    def detect_from_scores(scores: np.ndarray, disease: str, run: int) -> list[tuple[str, ...]]:
        order = np.argsort(-scores, kind="stable")[:200]
        genes = tuple(gene_ids[i] for i in order)
        return detect(genes, ppi, seed=stable_seed(args.seed, args.fold, disease, run))

    # 1) 40 次：删表型术语
    for index, disease in enumerate(sampled):
        batch = np.column_stack([
            drop_seed_terms(seeds[:, index], rng) for _ in range(40)
        ])
        result = v0_rwr.run_rwr(base_transition, batch, alpha=alpha)
        for run in range(40):
            modules_by_run[disease].append(detect_from_scores(result.scores[gene_indices, run], disease, run))

    # 2) 30 次：删 PPI 边（转移矩阵共享）
    for run, transition in enumerate(edge_transitions):
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        for index, disease in enumerate(sampled):
            modules_by_run[disease].append(detect_from_scores(result.scores[gene_indices, index], disease, 40 + run))

    # 3) 20 次：STRING 900
    for index, disease in enumerate(sampled):
        batch = np.column_stack([seeds[:, index]] * 20)
        result = v0_rwr.run_rwr(transition_900, batch, alpha=alpha)
        for run in range(20):
            modules_by_run[disease].append(detect_from_scores(result.scores[gene_indices, run], disease, 70 + run))

    # 4) 10 次：权重噪声
    for index, disease in enumerate(sampled):
        batch = np.column_stack([noise_seed_weights(seeds[:, index], rng) for _ in range(10)])
        result = v0_rwr.run_rwr(base_transition, batch, alpha=alpha)
        for run in range(10):
            modules_by_run[disease].append(detect_from_scores(result.scores[gene_indices, run], disease, 90 + run))

    rows = []
    member_rows = []
    for disease in sampled:
        candidate = list(matrix.top_k(disease, 200))
        position = {g: i for i, g in enumerate(candidate)}
        n = len(candidate)
        co_occur = np.zeros((n, n), dtype=np.float32)
        for partition in modules_by_run[disease]:
            for members in partition:
                idx = [position[g] for g in members if g in position]
                if len(idx) >= 2:
                    co_occur[np.ix_(idx, idx)] += 1.0
        co_occur /= len(modules_by_run[disease])
        np.fill_diagonal(co_occur, 1.0)

        distance = 1.0 - co_occur
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        tree = linkage(condensed, method="average")
        labels = fcluster(tree, t=0.3, criterion="distance")
        for label in sorted(set(labels)):
            idx = np.flatnonzero(labels == label)
            if len(idx) < 5:
                continue
            block = co_occur[np.ix_(idx, idx)]
            core = idx[np.array([block[i].mean() >= 0.8 for i in range(len(idx))])]
            periphery = idx[np.array([0.6 <= block[i].mean() < 0.8 for i in range(len(idx))])]
            coverage = len(idx) / n
            if coverage > 0.8:
                collapses[disease] += 1
                continue
            core_set = set(core.tolist())
            periphery_set = set(periphery.tolist())
            for local in idx:
                role = "core" if local in core_set else ("periphery" if local in periphery_set else "other")
                member_rows.append({
                    "module_id": f"{disease}:consensus:{label}",
                    "disease_id": disease,
                    "gene_id": candidate[local],
                    "role": role,
                    "frequency": round(float(co_occur[local, idx].mean()), 3),
                })
            rows.append({
                "disease_id": disease,
                "tier": tier_of.get(disease),
                "module_id": f"{disease}:consensus:{label}",
                "size": int(len(idx)),
                "core_size": int(len(core)),
                "periphery_size": int(len(periphery)),
                "coverage": round(float(coverage), 3),
                "mean_cooccurrence": round(float(co_occur[np.ix_(idx, idx)].mean()), 3),
            })
        log.info("disease_done", disease=disease, modules=len([r for r in rows if r["disease_id"] == disease]))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "consensus_modules.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out_dir / "consensus_members.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": len(sampled),
        "runs_per_disease": args.runs,
        "modules": int(len(table)),
        "modules_per_disease_median": float(table.groupby("disease_id").size().median()) if len(table) else None,
        "core_fraction_median": round(float((table["core_size"] / table["size"]).median()), 3) if len(table) else None,
        "collapses": int(sum(collapses.values())),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "consensus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
