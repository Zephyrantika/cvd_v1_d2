"""M4 修正第二步：按基因热度分层，比较真实模块与随机模块的 z。

为什么需要这一步：逐基因超额分位在数学上对 z 没有影响。
模块内每个基因减掉的是同一个常数 baseline(g)，而 z 比的是同一模块在不同疾病间的差，
基因级常数在分子和分母里同时出现、直接相消。真正需要的是**跨模块**条件化：
热门模块和冷门模块的 z 分布本来就不同，必须按热度分层或回归后再比。

产出：outputs/nulls_corrected/fold_0/stratified_comparison.parquet 与 summary_stratified.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def module_popularity(members: pd.DataFrame, id_column: str, popularity: dict[str, float]) -> pd.Series:
    grouped = members.groupby(id_column)["gene_id"].apply(
        lambda s: float(np.mean([popularity.get(str(g), 0.0) for g in s]))
    )
    return grouped.rename("popularity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "residualize.log", "05_residualize")
    out_dir = paths.out("nulls_corrected", f"fold_{args.fold}")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    top = (matrix.ranks <= args.top_k)
    popularity = {gene: float(top[:, i].mean()) for i, gene in enumerate(matrix.gene_ids)}

    real = pd.read_parquet(out_dir / "specificity_excess.parquet")
    real_members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    real = real.merge(module_popularity(real_members, "module_id", popularity), left_on="module_id", right_index=True)

    random_modules = pd.read_parquet(out_dir / "random_modules.parquet")
    random_members = pd.read_parquet(out_dir / "random_module_members.parquet")
    random_modules = random_modules.merge(
        module_popularity(random_members, "random_module_id", popularity),
        left_on="random_module_id", right_index=True,
    )

    rows = []
    for frame, kind in ((real, "real"), (random_modules, "random")):
        rho, p_value = spearmanr(frame["z_excess"], frame["popularity"])
        rows.append({"kind": kind, "n": len(frame),
                     "z_median": round(float(frame["z_excess"].median()), 3),
                     "popularity_median": round(float(frame["popularity"].median()), 4),
                     "spearman_z_popularity": round(float(rho), 3),
                     "spearman_p": float(p_value)})
    correlation = pd.DataFrame(rows)

    pooled = pd.concat([
        real.assign(kind="real")[["kind", "z_excess", "popularity"]],
        random_modules.assign(kind="random")[["kind", "z_excess", "popularity"]],
    ], ignore_index=True)
    try:
        pooled["stratum"] = pd.qcut(pooled["popularity"], q=args.bins, labels=False, duplicates="drop")
    except ValueError:
        pooled["stratum"] = 0

    stratified = []
    for stratum, group in pooled.groupby("stratum"):
        real_z = group.loc[group["kind"] == "real", "z_excess"].to_numpy()
        random_z = group.loc[group["kind"] == "random", "z_excess"].to_numpy()
        p_value = (
            float(mannwhitneyu(real_z, random_z, alternative="greater").pvalue)
            if len(real_z) and len(random_z) else float("nan")
        )
        stratified.append({
            "stratum": int(stratum),
            "popularity_low": round(float(group["popularity"].min()), 4),
            "popularity_high": round(float(group["popularity"].max()), 4),
            "n_real": len(real_z),
            "n_random": len(random_z),
            "z_median_real": round(float(np.median(real_z)), 3) if len(real_z) else None,
            "z_median_random": round(float(np.median(random_z)), 3) if len(random_z) else None,
            "mannwhitney_p_real_greater": p_value,
        })
    stratified_table = pd.DataFrame(stratified)
    stratified_table.to_parquet(out_dir / "stratified_comparison.parquet", index=False)

    real_all = real["z_excess"].to_numpy()
    random_all = random_modules["z_excess"].to_numpy()
    summary = {
        "fold": args.fold,
        "bins": args.bins,
        "ungrouped": {
            "z_median_real": round(float(np.median(real_all)), 3),
            "z_median_random": round(float(np.median(random_all)), 3),
            "mannwhitney_p_real_greater": float(mannwhitneyu(real_all, random_all, alternative="greater").pvalue),
        },
        "correlation": correlation.to_dict(orient="records"),
        "stratified": stratified,
        "n_strata_real_gt_random": int((stratified_table["z_median_real"] > stratified_table["z_median_random"]).sum()),
        "n_strata": int(len(stratified_table)),
    }
    (out_dir / "summary_stratified.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
