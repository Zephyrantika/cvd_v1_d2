"""M4 修正第四步：用已知基因做正对照（query-swap）。

问题：这个管线的表型查询，能不能至少把**自己疾病的已知基因**排在别的疾病查询之前？
这一步是唯一直接用到标签的检验，只用 T1/T2（每病 ≥3 个正样本），并且标签只用于评价，
不进入种子、不进入构图、不参与任何选择。

做法（对每个有 ≥3 个已知基因的疾病 d）：
    own    = 该病已知基因在 d 自己的排名里的平均分位；
    swap   = 同一批基因在同一折另外 30 个疾病的排名里的平均分位；
    diff   = own - swap。

若 diff 显著大于 0，说明表型查询确实指向自己的基因，管线有信号；
若 diff 接近 0，说明查到自己基因和查到别人的基因一样，问题在表型端。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--swap-draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    assoc = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"),
        columns=["disease_id", "gene_id"],
    ).drop_duplicates()
    assoc["disease_id"] = assoc["disease_id"].astype(str)
    assoc["gene_id"] = assoc["gene_id"].astype(str)
    known = assoc.groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s))).to_dict()

    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "fold"])
    fold_of = dict(zip(folds["disease_id"].astype(str), folds["fold"].astype(int)))

    rows = []
    for disease, genes in sorted(known.items()):
        if len(genes) < args.min_positives or disease not in fold_of:
            continue
        fold = fold_of[disease]
        matrix = load_saved_scores(paths.out("scores", f"fold_{fold}", "score_matrix.npz"))
        percentiles = matrix.percentiles()
        position = {gene: i for i, gene in enumerate(matrix.gene_ids)}
        present = [g for g in genes if g in position]
        if not present:
            continue
        rows_idx = [position[g] for g in present]
        own_row = matrix.disease_index(disease)
        own = float(percentiles[own_row, rows_idx].mean())

        others = [d for d in matrix.disease_ids if d != disease]
        picks = rng.choice(np.array(others), size=min(args.swap_draws, len(others)), replace=False)
        swap_values = np.array([percentiles[matrix.disease_index(str(d)), rows_idx].mean() for d in picks])

        rows.append({
            "disease_id": disease,
            "fold": fold,
            "n_known": len(genes),
            "n_evaluable": len(present),
            "own_percentile": own,
            "swap_mean": float(swap_values.mean()),
            "swap_sd": float(swap_values.std(ddof=1)),
            "diff": own - float(swap_values.mean()),
            "z": (own - float(swap_values.mean())) / (swap_values.std(ddof=1) + 1e-9),
            "empirical_p": (1 + int((swap_values >= own).sum())) / (len(swap_values) + 1),
            "swap_draws": len(swap_values),
        })

    table = pd.DataFrame(rows).sort_values(["fold", "n_known"], ascending=[True, False])
    table.to_parquet(out_dir / "query_swap_t1t2.parquet", index=False)

    summary = {
        "min_positives": args.min_positives,
        "diseases": int(len(table)),
        "median_own_percentile": round(float(table["own_percentile"].median()), 3),
        "median_swap_mean": round(float(table["swap_mean"].median()), 3),
        "median_diff": round(float(table["diff"].median()), 4),
        "frac_diff_positive": round(float((table["diff"] > 0).mean()), 3),
        "frac_p_le_0.05": round(float((table["empirical_p"] <= 0.05).mean()), 3),
        "wilcoxon_p_diff_gt_0": (
            float(wilcoxon(table["diff"], alternative="greater").pvalue) if len(table) > 5 else None
        ),
        "by_tier": {
            "n_ge5": int((table["n_known"] >= 5).sum()),
            "median_diff_ge5": round(float(table.loc[table["n_known"] >= 5, "diff"].median()), 4) if (table["n_known"] >= 5).any() else None,
            "median_diff_3_4": round(float(table.loc[table["n_known"] < 5, "diff"].median()), 4) if (table["n_known"] < 5).any() else None,
        },
    }
    (out_dir / "query_swap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
