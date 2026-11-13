"""M4 修正第五步：度归一化之后，query-swap 的差距会不会变大。

11 节发现已知基因在任意疾病查询下平均分位就有 0.843，怀疑是度数（rich club）在起作用。
这里把分位改成"同度数区间内的分位"：先按基因在 STRING 700 上的加权度数分 50 层，
每层内单独排名再转分位，然后重跑 query-swap。

如果度归一化之后 diff 明显变大，说明传播的度数偏向是可以修的；如果不变，说明信号本身不在。
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
from phenotype_modules_v0.nulls.structure_null import build_adjacency, gene_degrees  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def degree_normalized_percentiles(scores: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    """scores: 疾病 × 基因；bins: 每个基因的度数层。返回同形状的层内分位。"""
    result = np.full(scores.shape, np.nan, dtype=np.float32)
    for label in range(n_bins):
        columns = np.flatnonzero(bins == label)
        if len(columns) == 0:
            continue
        block = scores[:, columns]
        order = np.argsort(block, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(block.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, block.shape[1] + 1)[None, :]
        if block.shape[1] == 1:
            result[:, columns] = 1.0
        else:
            result[:, columns] = 1.0 - (ranks - 1) / (block.shape[1] - 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--swap-draws", type=int, default=30)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)
    degrees = gene_degrees(adjacency)

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
    for fold in sorted(set(fold_of.values())):
        matrix = load_saved_scores(paths.out("scores", f"fold_{fold}", "score_matrix.npz"))
        gene_degree = np.array([float(degrees.get(g, 0.0)) for g in matrix.gene_ids])
        edges = np.unique(np.quantile(gene_degree, np.linspace(0, 1, args.bins + 1)))
        bins = np.digitize(gene_degree, edges[1:-1], right=False)
        normalized = degree_normalized_percentiles(matrix.scores, bins, len(edges) - 1)
        position = {gene: i for i, gene in enumerate(matrix.gene_ids)}

        for disease, genes in sorted(known.items()):
            if len(genes) < args.min_positives or fold_of.get(disease) != fold:
                continue
            present = [g for g in genes if g in position]
            if not present:
                continue
            idx = [position[g] for g in present]
            own_row = matrix.disease_index(disease)
            own = float(normalized[own_row, idx].mean())
            others = [d for d in matrix.disease_ids if d != disease]
            picks = rng.choice(np.array(others), size=min(args.swap_draws, len(others)), replace=False)
            swap = np.array([normalized[matrix.disease_index(str(d)), idx].mean() for d in picks])
            rows.append({
                "disease_id": disease,
                "fold": fold,
                "n_known": len(genes),
                "n_evaluable": len(present),
                "own_dn": own,
                "swap_mean_dn": float(swap.mean()),
                "diff_dn": own - float(swap.mean()),
                "empirical_p_dn": (1 + int((swap >= own).sum())) / (len(swap) + 1),
            })

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "query_swap_degree_normalized.parquet", index=False)
    summary = {
        "bins": args.bins,
        "diseases": int(len(table)),
        "median_own": round(float(table["own_dn"].median()), 3),
        "median_swap": round(float(table["swap_mean_dn"].median()), 3),
        "median_diff": round(float(table["diff_dn"].median()), 4),
        "frac_diff_positive": round(float((table["diff_dn"] > 0).mean()), 3),
        "frac_p_le_0.05": round(float((table["empirical_p_dn"] <= 0.05).mean()), 3),
        "wilcoxon_p": float(wilcoxon(table["diff_dn"], alternative="greater").pvalue) if len(table) > 5 else None,
        "t1_median_diff": round(float(table.loc[table["n_known"] >= 5, "diff_dn"].median()), 4) if (table["n_known"] >= 5).any() else None,
        "t2_median_diff": round(float(table.loc[table["n_known"] < 5, "diff_dn"].median()), 4) if (table["n_known"] < 5).any() else None,
    }
    (out_dir / "query_swap_degree_normalized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # 与原始（未归一化）结果并排比较
    raw_path = out_dir / "query_swap_t1t2.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)[["disease_id", "diff", "empirical_p"]]
        merged = table.merge(raw, on="disease_id", how="inner", suffixes=("", "_raw"))
        merged.to_parquet(out_dir / "query_swap_degree_vs_raw.parquet", index=False)
        summary["raw_for_comparison"] = {
            "median_diff": round(float(merged["diff"].median()), 4),
            "frac_p_le_0.05": round(float((merged["empirical_p"] <= 0.05).mean()), 3),
            "improved_diseases": int((merged["diff_dn"] > merged["diff"]).sum()),
            "n": int(len(merged)),
        }
        (out_dir / "query_swap_degree_normalized_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
