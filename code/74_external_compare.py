"""外部证据对比：基线模块 vs 精修模块的 GWAS 支持与 GTEx 表达。

指标与 31_external_evidence.py 一致：
  gwas_frac        模块基因中出现在 GWAS Catalog 基因集的比例
  gwas_frac_strict 同上，但只取 p<=5e-8
  gwas_z           相对同大小随机基因集的标准化差异
  expr_vs_random   模块基因表达分位均值减去随机集均值
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

DRAWS = 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alt-members", default=None)
    parser.add_argument("--alt-label", default="refined")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "external_compare.log", "74_external_compare")
    started = time.time()
    out_dir = paths.out("external_compare", f"fold_{args.fold}_{args.alt_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    universe_set = set(universe)
    gwas = pd.read_parquet(paths.v0("data/processed/gwas_gene_support.parquet"), columns=["gene_id", "p_value"])
    gwas["gene_id"] = gwas["gene_id"].astype(str)
    gwas_any = set(gwas["gene_id"]) & universe_set
    gwas_strict = set(gwas.loc[gwas["p_value"] <= 5e-8, "gene_id"]) & universe_set

    expression = pd.read_parquet(paths.v0("data/processed/gtex_cardiovascular_expression.parquet"),
                                 columns=["gene_id", "median_tpm"])
    expression["gene_id"] = expression["gene_id"].astype(str)
    expr_mean = expression.assign(v=np.log1p(expression["median_tpm"].clip(lower=0))).groupby("gene_id")["v"].mean()
    expr_pct = expr_mean.rank(pct=True)
    expr_lookup = {g: float(expr_pct.get(g, np.nan)) for g in universe}

    mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    if args.alt_members:
        ref = pd.read_parquet(args.alt_members)
    else:
        ref = pd.read_parquet(paths.out("lam_refine", f"fold_{args.fold}_lam1.0_mu1.0", "lam_refine_members.parquet"))
    ref["gene_id"] = ref["gene_id"].astype(str)
    ref_sets = ref.groupby("disease_id")["gene_id"].apply(set).to_dict()

    rows = []
    for disease, module_id in largest.items():
        if disease not in ref_sets:
            continue
        for label, genes in (("baseline", base_sets.get(module_id, set())), (args.alt_label, ref_sets[disease])):
            target = {g for g in genes if g in universe_set}
            if len(target) < 5:
                continue
            frac_any = len(target & gwas_any) / len(target)
            frac_strict = len(target & gwas_strict) / len(target)
            expr_values = [expr_lookup[g] for g in target if np.isfinite(expr_lookup.get(g, np.nan))]
            random_any, random_expr = [], []
            for _ in range(args.draws):
                sample = set(rng.choice(universe, size=len(target), replace=False).tolist())
                random_any.append(len(sample & gwas_any) / len(sample))
                values = [expr_lookup[g] for g in sample if np.isfinite(expr_lookup.get(g, np.nan))]
                if values:
                    random_expr.append(float(np.mean(values)))
            rows.append({
                "disease_id": disease, "variant": label, "size": len(target),
                "gwas_frac": frac_any, "gwas_frac_strict": frac_strict,
                "gwas_z": (frac_any - np.mean(random_any)) / (np.std(random_any, ddof=1) + 1e-9),
                "expr_mean": float(np.mean(expr_values)) if expr_values else np.nan,
                "expr_vs_random": (float(np.mean(expr_values)) - np.mean(random_expr)) if expr_values and random_expr else np.nan,
            })
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "external_compare.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant",
                              values=["gwas_z", "expr_vs_random", "gwas_frac"])
    payload = {"fold": args.fold, "diseases": int(pivot.shape[0])}
    variants = list(dict.fromkeys(["baseline", args.alt_label]))
    for metric in ["gwas_z", "expr_vs_random", "gwas_frac"]:
        a, b = pivot[(metric, "baseline")], pivot[(metric, args.alt_label)]
        paired = pd.concat([a, b], axis=1).dropna()
        payload[metric] = {
            "baseline_median": round(float(paired.iloc[:, 0].median()), 4),
            "refined_median": round(float(paired.iloc[:, 1].median()), 4),
            "refined_better": int((paired.iloc[:, 1] > paired.iloc[:, 0]).sum()),
            "n": int(len(paired)),
            "p": float(wilcoxon(paired.iloc[:, 1], paired.iloc[:, 0]).pvalue),
        }
    payload["frac_modules_gwas_z_gt2"] = {
        v: round(float((table[table.variant == v].gwas_z > 2).mean()), 4) for v in variants}
    payload["frac_expr_above_random"] = {
        v: round(float((table[table.variant == v].expr_vs_random > 0).mean()), 4) for v in variants}
    payload["seconds"] = round(time.time() - started, 1)
    (out_dir / "external_compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
