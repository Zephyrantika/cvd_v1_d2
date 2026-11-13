"""检验"是不是 RWR 架构的限制"：扫重启概率 α，看 query-swap 的差距能不能被救回来。

α 大 → 质量留在种子附近，查询影响更局部；α 小 → 长程扩散，容易被 rich club 吸走。
如果某个 α 下 T1 的 diff 明显转正，说明是超参数/架构层面的问题；如果扫遍都不行，
说明问题在图的边上而不在传播算子。

做法：每个 fold 一次性对全部测试疾病批量跑 RWR，得到 疾病 × 基因 的分位矩阵，
再对 26 个有 ≥3 个已知基因的疾病算 own - swap。
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

ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument("--swap-draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "alpha_sweep.log", "12_alpha_sweep")
    started = time.time()

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

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

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )

    rows = []
    for fold in sorted(set(fold_of.values())):
        node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{fold}/node_map.parquet"))
        gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
        gene_ids = node_map.loc[node_map["node_type"].eq("gene"), "node_id"].astype(str).tolist()
        transcript = load_npz(paths.v0(f"data/graphs/fold_{fold}/transition_matrix.npz")).tocsr()
        hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/train_hpo_ic.parquet"))
        test_ids = [
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
        position = {g: i for i, g in enumerate(gene_ids)}

        for alpha in ALPHAS:
            result = v0_rwr.run_rwr(transcript, seeds, alpha=alpha)
            scores = result.scores[gene_rows, :]
            order = np.argsort(-scores, axis=0, kind="stable")
            ranks = np.empty_like(order)
            columns = np.arange(scores.shape[1])[None, :]
            ranks[order, columns] = np.arange(1, scores.shape[0] + 1)[:, None]
            percentiles = 1.0 - (ranks - 1) / (scores.shape[0] - 1)
            index = {d: i for i, d in enumerate(test_ids)}

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
                    "alpha": alpha,
                    "disease_id": disease,
                    "n_known": len(genes),
                    "own": own,
                    "swap_mean": float(swap.mean()),
                    "diff": own - float(swap.mean()),
                    "empirical_p": (1 + int((swap >= own).sum())) / (len(swap) + 1),
                    "converged": bool(result.converged),
                })
        log.info("fold_done", fold=fold, diseases=len(test_ids))

    table = pd.DataFrame(rows)
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "alpha_sweep.parquet", index=False)

    summary = []
    for alpha, group in table.groupby("alpha"):
        t1 = group.loc[group["n_known"] >= 5]
        summary.append({
            "alpha": float(alpha),
            "n_diseases": int(len(group)),
            "median_diff": round(float(group["diff"].median()), 4),
            "t1_median_diff": round(float(t1["diff"].median()), 4) if len(t1) else None,
            "frac_diff_positive": round(float((group["diff"] > 0).mean()), 3),
            "frac_p_le_0.05": round(float((group["empirical_p"] <= 0.05).mean()), 3),
            "converged": bool(group["converged"].all()),
        })
    (out_dir / "alpha_sweep_summary.json").write_text(
        json.dumps({"rows": summary, "seconds": round(time.time() - started, 2)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
