"""M6 对照：真实疾病的模块稳定性，与随机表型查询的模块稳定性，用同一套扰动比。

如果随机查询造出来的模块也同样稳定（Jaccard ≈ 0.75），那"稳定性"只是管线的性质，
不是疾病信号的性质；如果随机模块明显更不稳定，稳定性才有资格作为主结论。

两边用完全相同的扰动：每次随机删 10% 的表型种子项，重跑 RWR → top-200 → Leiden，
按 Jaccard 匹配回基础模块。
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
from phenotype_modules_v0.discovery.baseline import load_v0_models  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.phenotype_null import (  # noqa: E402
    ic_bins,
    sample_matched_query,
    term_neighborhoods,
)
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def module_stability(base_modules: list[tuple[str, ...]], perturbed: list[list[tuple[str, ...]]]) -> list[float]:
    values = []
    for members in base_modules:
        target = set(members)
        scores = [max((jaccard(target, set(m)) for m in draw), default=0.0) for draw in perturbed]
        values.append(float(np.mean(scores)) if scores else 0.0)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--diseases", type=int, default=5)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--drop-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "perturbation_control.log", "10_perturbation_control")
    started = time.time()
    out_dir = paths.out("stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    row_of_gene = {str(nid): i for i, nid in enumerate(node_map.loc[node_map["node_type"].eq("gene"), "node_id"])}
    align = np.array([row_of_gene[g] for g in matrix.gene_ids])
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    edges = pd.read_parquet(paths.v0("data/processed/hpo_edges.parquet"),
                            columns=["child_hpo_id", "parent_hpo_id"])
    ic = hpo_ic.set_index("hpo_id")["ic"].astype(float)
    ic.index = ic.index.astype(str)
    graph_terms = set(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str))
    ic = ic.loc[[t for t in ic.index if t in graph_terms]]
    term_bin, pool = ic_bins(ic, n_bins=5)
    neighborhood = term_neighborhoods(edges, hops=2)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    def query_terms(disease: str) -> tuple[list[str], list[float]] | None:
        frame = (
            phenotypes.loc[phenotypes["disease_id"].astype(str) == disease]
            .groupby("hpo_id", as_index=False)["frequency_weight"].max()
        )
        frame = frame.loc[frame["hpo_id"].astype(str).isin(term_bin)]
        if len(frame) < 3:
            return None
        return [str(t) for t in frame["hpo_id"]], frame["frequency_weight"].astype(float).tolist()

    def seeds_for(terms: list[str], weights: list[float], tag: str) -> np.ndarray:
        fake = pd.DataFrame({"disease_id": tag, "hpo_id": terms, "frequency_weight": weights})
        seeds, _ = v0_rwr.build_seed_matrix((tag,), fake, hpo_ic, node_map)
        return seeds

    def modules_for(seed_values: np.ndarray, seed_value: int) -> list[tuple[str, ...]]:
        result = v0_rwr.run_rwr(transition, seed_values, alpha=alpha)
        pct = result.scores[gene_rows, 0]
        order = np.argsort(-pct, kind="stable")[: args.top_k]
        genes = [matrix.gene_ids[i] for i in order]
        return detect(genes, ppi, seed=int(seed_value))

    rng = np.random.default_rng(args.seed)
    log.info("setup", alpha=alpha, terms=len(ic))

    selected = sorted(pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "modules.parquet"))["disease_id"].astype(str).unique())[: args.diseases]
    results = []
    detail = []

    for disease in selected:
        parsed = query_terms(disease)
        if parsed is None:
            continue
        terms, weights = parsed

        for label in ("real", "random"):
            if label == "real":
                query_terms_used, query_weights = terms, weights
            else:
                query_terms_used = sample_matched_query(terms, term_bin, pool, neighborhood, rng)
                query_weights = sorted(weights, reverse=True)

            tag = f"{disease}#{label}"
            base_seeds = seeds_for(query_terms_used, query_weights, tag)
            base_modules = modules_for(base_seeds, abs(hash(tag)) % (2**31))

            perturbed: list[list[tuple[str, ...]]] = []
            for draw in range(args.draws):
                keep = max(2, int(np.floor(len(query_terms_used) * (1 - args.drop_fraction))))
                chosen = rng.choice(np.arange(len(query_terms_used)), size=keep, replace=False)
                p_terms = [query_terms_used[i] for i in sorted(chosen)]
                p_weights = [query_weights[i] for i in sorted(chosen)]
                p_seeds = seeds_for(p_terms, p_weights, f"{tag}#p{draw:03d}")
                perturbed.append(modules_for(p_seeds, abs(hash((tag, draw))) % (2**31)))

            stability = module_stability(base_modules, perturbed)
            results.append({"disease_id": disease, "query_type": label,
                            "n_modules": len(base_modules),
                            "stability_median": float(np.median(stability)) if stability else float("nan")})
            detail.extend({"disease_id": disease, "query_type": label, "module_index": i, "jaccard": v}
                          for i, v in enumerate(stability))
        log.info("disease_done", disease=disease, draws=args.draws)

    summary_table = pd.DataFrame(results)
    detail_table = pd.DataFrame(detail)
    summary_table.to_parquet(out_dir / "perturbation_control_summary.parquet", index=False)
    detail_table.to_parquet(out_dir / "perturbation_control_detail.parquet", index=False)

    real_values = detail_table.loc[detail_table["query_type"] == "real", "jaccard"]
    random_values = detail_table.loc[detail_table["query_type"] == "random", "jaccard"]
    summary = {
        "fold": args.fold,
        "draws": args.draws,
        "drop_fraction": args.drop_fraction,
        "real": {
            "modules": int(len(real_values)),
            "jaccard_median": round(float(real_values.median()), 3),
            "frac_ge_0.6": round(float((real_values >= 0.6).mean()), 3),
        },
        "random": {
            "modules": int(len(random_values)),
            "jaccard_median": round(float(random_values.median()), 3),
            "frac_ge_0.6": round(float((random_values >= 0.6).mean()), 3),
        },
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "perturbation_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
