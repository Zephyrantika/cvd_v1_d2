"""补主工作第 2 条与消融：三约束模块发现（表型特异性 + 结构紧密性 + 扰动一致性）。

做法：从 V0 的 Leiden 模块出发，在候选集 C_d 内做贪心精修，最大化

    F = 0.40·pct(z_excess) + 0.20·pct(density) + 0.30·stability − 0.10·hub

每一步尝试删一个外围基因或加一个相邻基因，取 ΔF 最大的一步，直到无正增益或 50 步。
stability 用 V0 五十次扰动的共识频率（核心 0.8 以上）作为代理。

消融：把某一项的权重置零重跑同一次精修。
    full / no_phenotype / no_tightness / no_stability / no_hub_penalty
另外 single_seed 直接用原始 Leiden 模块，不精修，作为对照。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.evaluation.specificity import background_by_similarity, phenotype_sets  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency, density_and_hub  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

WEIGHTS = {
    "full": {"z": 0.40, "density": 0.20, "stability": 0.30, "hub": 0.10},
    "no_phenotype": {"z": 0.0, "density": 0.20, "stability": 0.30, "hub": 0.10},
    "no_tightness": {"z": 0.40, "density": 0.0, "stability": 0.30, "hub": 0.10},
    "no_stability": {"z": 0.40, "density": 0.20, "stability": 0.0, "hub": 0.10},
    "no_hub_penalty": {"z": 0.40, "density": 0.20, "stability": 0.30, "hub": 0.0},
}
MIN_SIZE, MAX_SIZE, MAX_STEPS = 5, 100, 10


def module_stats(genes: list[str], adjacency, own_pct, bg_pct, freq: dict[str, float],
                 gene_position: dict[str, int]):
    indices = np.fromiter((gene_position[g] for g in genes), dtype=np.int64, count=len(genes))
    density, hub = density_and_hub(genes, adjacency)
    self_score = float(own_pct[indices].mean())
    background = bg_pct[:, indices].mean(axis=1)
    z = (self_score - float(background.mean())) / (float(background.std(ddof=1)) + 1e-9)
    stability = float(np.mean([freq.get(g, 0.0) for g in genes]))
    return density, hub, z, stability


def objective(weights, density, hub, z, stability, scales):
    return (
        weights["z"] * (z / scales["z"])
        + weights["density"] * (density / scales["density"])
        + weights["stability"] * (stability / scales["stability"])
        - weights["hub"] * (hub / scales["hub"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "three_constraint.log", "17_three_constraint")
    started = time.time()
    out_dir = paths.out("constraints")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    gene_position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    base_by_disease: dict[str, list[list[str]]] = {}
    for (disease, module_id), group in members.groupby(["disease_id", "module_id"]):
        base_by_disease.setdefault(str(disease), []).append(list(dict.fromkeys(group["gene_id"].astype(str))))

    # 共识频率取自 V0 的成员表（我们的模块与 V0 逐成员一致，按 疾病+基因 对齐即可）
    v0_members = pd.read_parquet(
        paths.v0("outputs/modules/module_members.parquet"),
        columns=["disease_id", "gene_id", "consensus_frequency"],
    )
    v0_members["disease_id"] = v0_members["disease_id"].astype(str)
    v0_members["gene_id"] = v0_members["gene_id"].astype(str)
    freq_by_disease: dict[str, dict[str, float]] = {}
    for row in v0_members.itertuples(index=False):
        freq_by_disease.setdefault(row.disease_id, {})[row.gene_id] = float(row.consensus_frequency)

    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "fold"])
    fold_diseases = tuple(sorted(matrix.disease_ids))
    pheno_sets = phenotype_sets(pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id"],
    ))

    scales = {"z": 2.0, "density": 0.4, "stability": 0.8, "hub": 0.2}
    rows = []

    for disease, modules in sorted(base_by_disease.items()):
        own_row = matrix.disease_index(disease)
        bg = background_by_similarity(disease, fold_diseases, pheno_sets)
        bg_rows = [matrix.disease_index(d) for d in bg if d in matrix.disease_ids]
        candidate = list(matrix.top_k(disease, args.top_k))
        candidate_set = set(candidate)
        freq = freq_by_disease.get(disease, {})
        own_pct = percentiles[own_row]
        bg_pct = percentiles[bg_rows]

        for variant, weights in WEIGHTS.items():
            for index, members0 in enumerate(modules):
                current = [g for g in members0 if g in gene_position and g in candidate_set]
                if len(current) < MIN_SIZE:
                    continue
                density, hub, z, stability = module_stats(
                    current, adjacency, own_pct, bg_pct, freq, gene_position
                )
                best = objective(weights, density, hub, z, stability, scales)
                steps = 0
                while steps < MAX_STEPS:
                    best_move, best_value = None, best
                    neighbours = set()
                    for gene in current:
                        neighbours |= set(adjacency.get(gene, {}))
                    neighbours = [g for g in neighbours if g in candidate_set and g not in current]
                    neighbours.sort(key=lambda g: -sum(adjacency.get(g, {}).get(m, 0.0) for m in current))
                    neighbours = neighbours[:15]
                    # 删除
                    if len(current) > MIN_SIZE:
                        for gene in current:
                            trial = [g for g in current if g != gene]
                            d, h, zz, s = module_stats(trial, adjacency, own_pct, bg_pct, freq, gene_position)
                            value = objective(weights, d, h, zz, s, scales)
                            if value > best_value:
                                best_move, best_value = ("remove", gene), value
                    # 加入
                    if len(current) < MAX_SIZE:
                        for gene in neighbours:
                            trial = current + [gene]
                            d, h, zz, s = module_stats(trial, adjacency, own_pct, bg_pct, freq, gene_position)
                            value = objective(weights, d, h, zz, s, scales)
                            if value > best_value:
                                best_move, best_value = ("add", gene), value
                    if best_move is None:
                        break
                    action, gene = best_move
                    current = [g for g in current if g != gene] if action == "remove" else current + [gene]
                    best = best_value
                    steps += 1

                d, h, z, s = module_stats(current, adjacency, own_pct, bg_pct, freq, gene_position)
                rows.append({
                    "variant": variant,
                    "disease_id": disease,
                    "module_index": index,
                    "size_before": len(members0),
                    "size_after": len(current),
                    "steps": steps,
                    "density": d,
                    "hub": h,
                    "z_excess": z,
                    "stability_proxy": s,
                })
        log.info("disease_done", disease=disease, modules=len(modules))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"three_constraint_fold_{args.fold}.parquet", index=False)
    summary = (
        table.groupby("variant")
        .agg(modules=("module_index", "size"),
             size_median=("size_after", "median"),
             z_median=("z_excess", "median"),
             density_median=("density", "median"),
             hub_median=("hub", "median"),
             stability_median=("stability_proxy", "median"),
             steps_median=("steps", "median"))
        .round(3)
    )
    summary.to_csv(out_dir / f"three_constraint_summary_fold_{args.fold}.csv", encoding="utf-8-sig")
    payload = {
        "fold": args.fold,
        "weights": WEIGHTS,
        "table": summary.reset_index().to_dict(orient="records"),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / f"three_constraint_summary_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
