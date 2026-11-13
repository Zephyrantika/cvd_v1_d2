"""M4 修正第三步：先按模块大小分层，再看真实模块与随机模块的差别有多少是尺寸造成的。

真实模块每个疾病并集约 182 个基因，随机查询造的模块只有 26 个，直接比 z 和密度都不公平。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    out_dir = paths.out("nulls_corrected", f"fold_{args.fold}")

    real = pd.read_parquet(out_dir / "specificity_excess.parquet")[["module_id", "disease_id", "module_size", "z_excess"]]
    real_modules = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "modules.parquet"))
    real = real.merge(real_modules[["module_id", "internal_density"]], on="module_id", how="left")
    real = real.rename(columns={"internal_density": "density"})

    random_modules = pd.read_parquet(out_dir / "random_modules.parquet")[
        ["random_module_id", "disease_id", "module_size", "density", "z_excess"]
    ]
    random_modules = random_modules.rename(columns={"random_module_id": "module_id"})
    random_modules["source"] = "random"
    real["source"] = "real"
    real = real.reset_index(drop=True)
    random_modules = random_modules.reset_index(drop=True)
    pooled = pd.concat([real, random_modules], ignore_index=True)

    edges = np.quantile(real["module_size"], np.linspace(0, 1, args.bins + 1))
    edges = np.unique(edges.astype(int))
    pooled["size_bin"] = pd.cut(pooled["module_size"], bins=edges, include_lowest=True, duplicates="drop")

    rows = []
    for size_bin, group in pooled.groupby("size_bin", observed=True):
        real_z = group.loc[group["source"] == "real", "z_excess"].to_numpy()
        random_z = group.loc[group["source"] == "random", "z_excess"].to_numpy()
        real_d = group.loc[group["source"] == "real", "density"].dropna().to_numpy()
        random_d = group.loc[group["source"] == "random", "density"].dropna().to_numpy()
        rows.append({
            "size_bin": str(size_bin),
            "n_real": len(real_z),
            "n_random": len(random_z),
            "z_real": round(float(np.median(real_z)), 3) if len(real_z) else None,
            "z_random": round(float(np.median(random_z)), 3) if len(random_z) else None,
            "p_z_real_greater": (
                float(mannwhitneyu(real_z, random_z, alternative="greater").pvalue)
                if len(real_z) and len(random_z) else None
            ),
            "density_real": round(float(np.median(real_d)), 3) if len(real_d) else None,
            "density_random": round(float(np.median(random_d)), 3) if len(random_d) else None,
            "p_density_real_greater": (
                float(mannwhitneyu(real_d, random_d, alternative="greater").pvalue)
                if len(real_d) and len(random_d) else None
            ),
        })
    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "size_matched_comparison.parquet", index=False)

    summary = {
        "fold": args.fold,
        "size_edges": [int(e) for e in edges],
        "overall": {
            "z_real_median": round(float(real["z_excess"].median()), 3),
            "z_random_median": round(float(random_modules["z_excess"].median()), 3),
            "density_real_median": round(float(real["density"].median()), 3),
            "density_random_median": round(float(random_modules["density"].median()), 3),
        },
        "stratified": rows,
        "n_strata_z_real_gt_random": int(sum(
            1 for r in rows if r["z_real"] is not None and r["z_random"] is not None and r["z_real"] > r["z_random"]
        )),
        "n_strata": len(rows),
    }
    (out_dir / "size_matched_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
