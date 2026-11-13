"""换网络的第一步：用 GTEx 心血管组织表达给 PPI 边加权，重跑 query-swap。

不换候选集、不改折划分、不改种子，只改边权：
    s_g = 该基因在 5 个心血管组织（主动脉、冠脉、胫动脉、心房、左心室）的
          平均 log1p(TPM)，在全体基因内转成百分位，缺失记 0.5；
    基因—基因关系矩阵按 A' = diag(s) A diag(s) 缩放。

三个变体：
    baseline   不加权
    expr_soft  s ∈ [0.5, 1]
    expr_hard  s ∈ [0.05, 1]
    expr_both  expr_hard，并且同样缩放基因—通路关系

判据（预注册）：T1（≥5 个正样本）的 diff 中位数 ≥ 0.08，全体中位 ≥ 0.08。
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

TISSUES = ("Artery_Aorta", "Artery_Coronary", "Artery_Tibial", "Heart_Atrial_Appendage", "Heart_Left_Ventricle")
VARIANTS = ("baseline", "expr_soft", "expr_hard", "expr_both")


def expression_percentiles(paths) -> dict[str, float]:
    frame = pd.read_parquet(
        paths.v0("data/processed/gtex_cardiovascular_expression.parquet"),
        columns=["gene_id", "tissue", "median_tpm"],
    )
    frame["gene_id"] = frame["gene_id"].astype(str)
    frame = frame.loc[frame["tissue"].isin(TISSUES)]
    scores = frame.assign(value=np.log1p(frame["median_tpm"].clip(lower=0))).groupby("gene_id")["value"].mean()
    ranks = scores.rank(pct=True)
    return {gene: float(value) for gene, value in ranks.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--swap-draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "tissue_weighted.log", "14_tissue_weighted")
    started = time.time()

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix

    expression = expression_percentiles(paths)
    log.info("expression_loaded", genes=len(expression))

    budget = {
        "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
        "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
                 "gene_participates_in_pathway": 0.25},
        "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
    }

    assoc = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"), columns=["disease_id", "gene_id"]
    ).drop_duplicates()
    assoc["disease_id"] = assoc["disease_id"].astype(str)
    assoc["gene_id"] = assoc["gene_id"].astype(str)
    known = assoc.groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s))).to_dict()
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "fold"])
    fold_of = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["fold"].astype(int)))
    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )

    rows = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        graph_dir = paths.v0(f"data/graphs/fold_{fold}")
        relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr() for p in sorted(graph_dir.glob("A_*.npz"))}
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        gene_ids = gene_rows["node_id"].astype(str).tolist()
        position = {g: i for i, g in enumerate(gene_ids)}

        raw = np.array([expression.get(g, 0.5) for g in gene_ids], dtype=float)
        node_scale = np.ones(len(node_map), dtype=float)

        hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/train_hpo_ic.parquet"))
        test_ids = [
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
        index = {d: i for i, d in enumerate(test_ids)}
        alpha = float(json.loads(
            paths.v0(f"outputs/rankings/fold_{fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
        )["selected_alpha"])

        for variant in VARIANTS:
            relations_variant = dict(relations)
            if variant != "baseline":
                if variant == "expr_soft":
                    scale = 0.5 + 0.5 * raw
                else:
                    scale = 0.05 + 0.95 * raw
                node_scale[:] = 1.0
                node_scale[gene_indices] = scale
                scaling = diags(node_scale)
                relations_variant["gene_gene"] = (
                    scaling @ relations["gene_gene"] @ scaling
                ).tocsr()
                if variant == "expr_both":
                    relations_variant["gene_pathway"] = (
                        scaling @ relations["gene_pathway"]
                    ).tocsr()
            transition = compose_transition_matrix(relations_variant, budget).tocsr()
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_indices, :]
            order = np.argsort(-scores, axis=0, kind="stable")
            ranks = np.empty_like(order)
            columns = np.arange(scores.shape[1])[None, :]
            ranks[order, columns] = np.arange(1, scores.shape[0] + 1)[:, None]
            percentiles = 1.0 - (ranks - 1) / (scores.shape[0] - 1)

            for disease, genes in sorted(known.items()):
                if len(genes) < args.min_positives or fold_of.get(disease) != fold:
                    continue
                present = [position[g] for g in genes if g in position]
                if not present:
                    continue
                own = float(percentiles[present, index[disease]].mean())
                others = [d for d in test_ids if d != disease]
                picks = rng.choice(np.array(others), size=min(args.swap_draws, len(others)), replace=False)
                swap = np.array([percentiles[present, index[str(d)]].mean() for d in picks])
                rows.append({
                    "fold": fold,
                    "variant": variant,
                    "disease_id": disease,
                    "n_known": len(genes),
                    "own": own,
                    "swap_mean": float(swap.mean()),
                    "diff": own - float(swap.mean()),
                    "empirical_p": (1 + int((swap >= own).sum())) / (len(swap) + 1),
                })
            log.info("variant_done", fold=fold, variant=variant, alpha=alpha, converged=bool(result.converged))

    table = pd.DataFrame(rows)
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "tissue_weighted.parquet", index=False)

    summary = []
    for variant in VARIANTS:
        group = table.loc[table["variant"] == variant]
        t1 = group.loc[group["n_known"] >= 5]
        summary.append({
            "variant": variant,
            "n_diseases": int(len(group)),
            "median_diff": round(float(group["diff"].median()), 4),
            "t1_median_diff": round(float(t1["diff"].median()), 4) if len(t1) else None,
            "frac_diff_positive": round(float((group["diff"] > 0).mean()), 3),
            "frac_p_le_0.05": round(float((group["empirical_p"] <= 0.05).mean()), 3),
        })
    (out_dir / "tissue_weighted_summary.json").write_text(
        json.dumps({"folds": args.folds, "criterion": {"t1_median_diff": 0.08, "all_median_diff": 0.08},
                    "rows": summary, "seconds": round(time.time() - started, 2)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
