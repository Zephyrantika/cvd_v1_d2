"""检验"是不是关系预算的问题"：提高基因节点上表型边的权重，重跑 query-swap。

现在基因节点上的转移预算：表型 0.15、PPI 0.60、通路 0.25。
表型边只占一成半，种子对分数的影响被 PPI 稀释。这里扫三档：

    base      0.15 / 0.60 / 0.25  （现状）
    mid       0.25 / 0.50 / 0.25
    high_hpo  0.35 / 0.45 / 0.20
    max_hpo   0.50 / 0.30 / 0.20

只重建转移矩阵，其余（种子、α、top-200、Leiden）都不动。
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
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

VARIANTS = {
    "base": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "mid": {"gene_associated_with_phenotype": 0.25, "gene_interacts_with_gene": 0.50,
            "gene_participates_in_pathway": 0.25},
    "high_hpo": {"gene_associated_with_phenotype": 0.35, "gene_interacts_with_gene": 0.45,
                 "gene_participates_in_pathway": 0.20},
    "max_hpo": {"gene_associated_with_phenotype": 0.50, "gene_interacts_with_gene": 0.30,
                "gene_participates_in_pathway": 0.20},
}
OTHER_BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--swap-draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "relation_budget.log", "13_relation_budget")
    started = time.time()

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix

    assoc = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"),
        columns=["disease_id", "gene_id"],
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
        relations = {
            p.stem.removeprefix("A_"): load_npz(p).tocsr()
            for p in sorted(graph_dir.glob("A_*.npz"))
        }
        node_map = pd.read_parquet(graph_dir / "node_map.parquet")
        gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
        gene_ids = node_map.loc[node_map["node_type"].eq("gene"), "node_id"].astype(str).tolist()
        position = {g: i for i, g in enumerate(gene_ids)}
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

        for name, budget in VARIANTS.items():
            full_budget = {**OTHER_BUDGET, "gene": budget}
            transition = compose_transition_matrix(relations, full_budget).tocsr()
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows, :]
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
                    "variant": name,
                    "disease_id": disease,
                    "n_known": len(genes),
                    "own": own,
                    "swap_mean": float(swap.mean()),
                    "diff": own - float(swap.mean()),
                    "empirical_p": (1 + int((swap >= own).sum())) / (len(swap) + 1),
                })
            log.info("variant_done", fold=fold, variant=name, alpha=alpha, converged=bool(result.converged))
    table = pd.DataFrame(rows)
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "relation_budget.parquet", index=False)

    summary = []
    for name, group in table.groupby("variant"):
        t1 = group.loc[group["n_known"] >= 5]
        summary.append({
            "variant": name,
            "budget": VARIANTS[name],
            "n_diseases": int(len(group)),
            "median_diff": round(float(group["diff"].median()), 4),
            "t1_median_diff": round(float(t1["diff"].median()), 4) if len(t1) else None,
            "frac_diff_positive": round(float((group["diff"] > 0).mean()), 3),
            "frac_p_le_0.05": round(float((group["empirical_p"] <= 0.05).mean()), 3),
        })
    order = ["base", "mid", "high_hpo", "max_hpo"]
    summary = sorted(summary, key=lambda r: order.index(r["variant"]))
    (out_dir / "relation_budget_summary.json").write_text(
        json.dumps({"folds": args.folds, "rows": summary,
                    "seconds": round(time.time() - started, 2)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
