"""补齐稳健性：PPI 边删除与 Leiden 随机种子变化。

导师要求的四项实验里，"边扰动、表型缺失、随机种子变化"只做了表型缺失。
这里补边删除（5%/10%/20%，各 5 次）和种子变化（20 个种子）。

边删除的做法：在 A_gene_gene 的上三角随机置零，成对删除保证对称，
再按原预算重组转移矩阵、重跑 RWR、重跑 Leiden（种子固定，只让边变）。
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
from scipy.sparse import load_npz

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


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def partition_similarity(base: list[tuple[str, ...]], new: list[tuple[str, ...]]) -> float:
    if not base:
        return float("nan")
    scores = []
    for members in base:
        target = set(members)
        scores.append(max((jaccard(target, set(m)) for m in new), default=0.0))
    return float(np.mean(scores))


def drop_edges(matrix, fraction: float, rng: np.random.Generator):
    coo = matrix.tocoo()
    keep_upper = coo.row < coo.col
    rows, cols, data = coo.row[keep_upper], coo.col[keep_upper], coo.data[keep_upper]
    mask = rng.random(len(rows)) >= fraction
    rows, cols, data = rows[mask], cols[mask], data[mask]
    # 重新对称化
    rows_all = np.concatenate([rows, cols])
    cols_all = np.concatenate([cols, rows])
    data_all = np.concatenate([data, data])
    from scipy.sparse import csr_matrix
    return csr_matrix((data_all, (rows_all, cols_all)), shape=matrix.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--edge-runs", type=int, default=5)
    parser.add_argument("--seed-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "robustness.log", "16_robustness")
    started = time.time()
    out_dir = paths.out("robustness")
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

    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    test_ids = [
        line.strip()
        for line in paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])

    base_members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    base_by_disease = (
        base_members.groupby(["disease_id", "module_id"])["gene_id"]
        .apply(lambda s: tuple(sorted(set(map(str, s)), key=int)))
        .groupby("disease_id").apply(list).to_dict()
    )
    sampled = sorted(base_by_disease)

    def modules_from_scores(percentiles: np.ndarray, seed_value: int) -> dict[str, list[tuple[str, ...]]]:
        result = {}
        for disease in sampled:
            column = test_ids.index(disease)
            order = np.argsort(-percentiles[:, column], kind="stable")[: args.top_k]
            genes = tuple(gene_ids[i] for i in order)
            result[disease] = detect(genes, ppi, seed=stable_seed(seed_value, args.fold, disease, "base"))
        return result

    def percentiles_of(transition) -> np.ndarray:
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        scores = result.scores[gene_indices, :]
        order = np.argsort(-scores, axis=0, kind="stable")
        ranks = np.empty_like(order)
        columns = np.arange(scores.shape[1])[None, :]
        ranks[order, columns] = np.arange(1, scores.shape[0] + 1)[:, None]
        return 1.0 - (ranks - 1) / (scores.shape[0] - 1)

    rows = []
    baseline_transition = compose_transition_matrix(relations, BUDGET).tocsr()
    base_percentiles = percentiles_of(baseline_transition)
    base_modules = {d: base_by_disease[d] for d in sampled}

    # 种子变化
    for run in range(args.seed_runs):
        reference = modules_from_scores(base_percentiles, args.seed)  # 固定参考划分
        perturbed = modules_from_scores(base_percentiles, args.seed + run + 1)
        for disease in sampled:
            rows.append({
                "kind": "seed",
                "level": "seed_variation",
                "run": run,
                "disease_id": disease,
                "similarity": partition_similarity(reference[disease], perturbed[disease]),
            })
        log.info("seed_run_done", run_index=run)

    # 边删除
    for level in (0.05, 0.10, 0.20):
        for run in range(args.edge_runs):
            damaged = dict(relations)
            damaged["gene_gene"] = drop_edges(relations["gene_gene"], level, rng)
            transition = compose_transition_matrix(damaged, BUDGET).tocsr()
            percentiles = percentiles_of(transition)
            perturbed = modules_from_scores(percentiles, args.seed)
            for disease in sampled:
                rows.append({
                    "kind": "edge",
                    "level": f"delete_{int(level * 100)}pct",
                    "run": run,
                    "disease_id": disease,
                    "similarity": partition_similarity(base_modules[disease], perturbed[disease]),
                })
            log.info("edge_run_done", delete_level=level, run_index=run)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"robustness_fold_{args.fold}.parquet", index=False)
    summary = (
        table.groupby(["kind", "level"])
        .agg(jaccard_median=("similarity", "median"),
             q25=("similarity", lambda s: s.quantile(0.25)),
             frac_ge_0_6=("similarity", lambda s: float((s >= 0.6).mean())))
        .round(3)
    )
    summary.to_csv(out_dir / f"robustness_summary_fold_{args.fold}.csv", encoding="utf-8-sig")
    payload = {
        "fold": args.fold,
        "edge_runs": args.edge_runs,
        "seed_runs": args.seed_runs,
        "table": summary.reset_index().to_dict(orient="records"),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / f"robustness_summary_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
