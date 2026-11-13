"""表型驱动线的外部证据：模块基因的 GWAS 支持与组织表达，和同大小随机集比。

导师要求的第四项实验（未参与模块构建的证据）里，GO 已经做过；这里补 GWAS 与 GTEx。
GWAS 只用基因级支持，不升级到模块级因果；GTEx 只做组织表达的支持说明。
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
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

DRAWS = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "external_evidence.log", "31_external_evidence")
    started = time.time()
    out_dir = paths.out("external_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    universe_set = set(universe)

    gwas = pd.read_parquet(paths.v0("data/processed/gwas_gene_support.parquet"),
                           columns=["gene_id", "p_value"])
    gwas["gene_id"] = gwas["gene_id"].astype(str)
    gwas_any = set(gwas["gene_id"]) & universe_set
    gwas_strict = set(gwas.loc[gwas["p_value"] <= 5e-8, "gene_id"]) & universe_set
    log.info("gwas_loaded", any_n=len(gwas_any), strict_n=len(gwas_strict))

    expression = pd.read_parquet(paths.v0("data/processed/gtex_cardiovascular_expression.parquet"),
                                 columns=["gene_id", "median_tpm"])
    expression["gene_id"] = expression["gene_id"].astype(str)
    expr_mean = expression.assign(v=np.log1p(expression["median_tpm"].clip(lower=0))).groupby("gene_id")["v"].mean()
    expr_pct = expr_mean.rank(pct=True)
    expr_lookup = {g: float(expr_pct.get(g, np.nan)) for g in universe}

    modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"), columns=["module_id", "disease_id", "fold"])
    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"),
                              columns=["module_id", "gene_id"])
    members["gene_id"] = members["gene_id"].astype(str)
    tiers = pd.read_parquet(paths.out("audit", "disease_tiers.parquet"))[["disease_id", "tier"]]
    module_members = members.groupby("module_id")["gene_id"].apply(list).to_dict()
    module_tier = modules.merge(tiers, on="disease_id", how="left").set_index("module_id")["tier"].to_dict()

    rows = []
    for module_id, genes in module_members.items():
        genes = [g for g in genes if g in universe_set]
        if len(genes) < 5:
            continue
        target = set(genes)
        frac_any = len(target & gwas_any) / len(target)
        frac_strict = len(target & gwas_strict) / len(target)
        expr_values = [expr_lookup[g] for g in target if np.isfinite(expr_lookup.get(g, np.nan))]

        random_any, random_strict, random_expr = [], [], []
        for _ in range(DRAWS):
            sample = set(rng.choice(universe, size=len(target), replace=False).tolist())
            random_any.append(len(sample & gwas_any) / len(sample))
            random_strict.append(len(sample & gwas_strict) / len(sample))
            values = [expr_lookup[g] for g in sample if np.isfinite(expr_lookup.get(g, np.nan))]
            if values:
                random_expr.append(float(np.mean(values)))

        rows.append({
            "module_id": module_id,
            "tier": module_tier.get(module_id),
            "module_size": len(target),
            "gwas_frac": frac_any,
            "gwas_frac_strict": frac_strict,
            "gwas_z": (frac_any - np.mean(random_any)) / (np.std(random_any, ddof=1) + 1e-9),
            "expr_mean": float(np.mean(expr_values)) if expr_values else np.nan,
            "expr_vs_random": (float(np.mean(expr_values)) - np.mean(random_expr)) if expr_values and random_expr else np.nan,
        })

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "module_external_evidence.parquet", index=False)
    summary = (
        table.groupby("tier")
        .agg(modules=("module_id", "size"),
             gwas_frac_median=("gwas_frac", "median"),
             gwas_frac_strict_median=("gwas_frac_strict", "median"),
             gwas_z_median=("gwas_z", "median"),
             expr_mean_median=("expr_mean", "median"),
             expr_vs_random_median=("expr_vs_random", "median"))
        .round(4)
    )
    summary.to_csv(out_dir / "module_external_evidence_summary.csv", encoding="utf-8-sig")
    payload = {
        "modules": int(len(table)),
        "frac_modules_gwas_z_gt_2": round(float((table["gwas_z"] > 2).mean()), 4),
        "frac_modules_expr_above_random": round(float((table["expr_vs_random"] > 0).mean()), 4),
        "table": summary.reset_index().to_dict(orient="records"),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "module_external_evidence.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string())
    print(json.dumps({k: v for k, v in payload.items() if k != "table"}, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
