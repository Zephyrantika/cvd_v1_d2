"""M6/M7 试点：扰动稳定性与外部功能一致性（H2）。

两件事：
    1. 用 V0 已有的 50 次扰动结果，看稳定性在 T1–T3 上的分布，并抽出稳定核心（频率 ≥0.8）；
    2. 用未参与构图的 GO 注释做超几何富集，比较"稳定核心 / 完整模块 / 同大小随机基因集"的
       最佳 FDR。这一步直接检验 H2：核心是否比完整模块更有一致的功能。

不重跑扰动：V0 的 modules.parquet 和 module_members.parquet 里已经有 50 次扰动的
stability_mean_jaccard 和 consensus_frequency，重建一次只会得到同一批数。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

CORE_THRESHOLD = 0.8
PERIPHERY_THRESHOLD = 0.6
MIN_TERM_SIZE = 10
MAX_TERM_SIZE = 500
EXCLUDE_EVIDENCE = {"IEA"}


def go_index(paths, universe: set[str]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    annotations = pd.read_parquet(
        paths.v0("data/processed/go_gene_annotations.parquet"),
        columns=["gene_id", "go_id", "evidence_code"],
    )
    annotations["gene_id"] = annotations["gene_id"].astype(str)
    annotations = annotations.loc[
        annotations["gene_id"].isin(universe)
        & ~annotations["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)
    ]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(annotations["gene_id"], annotations["go_id"], strict=True):
        term_genes.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes = {
        term: genes
        for term, genes in term_genes.items()
        if MIN_TERM_SIZE <= len(genes) <= MAX_TERM_SIZE
    }
    gene_terms = {g: [t for t in terms if t in term_genes] for g, terms in gene_terms.items()}
    return term_genes, gene_terms


def best_fdr(genes: set[str], term_genes: dict[str, set[str]], gene_terms: dict[str, list[str]], n_universe: int) -> tuple[float, str | None, int]:
    if not genes:
        return float("nan"), None, 0
    candidate_terms: set[str] = set()
    for gene in genes:
        candidate_terms.update(gene_terms.get(gene, ()))
    if not candidate_terms:
        return float("nan"), None, 0
    terms = sorted(candidate_terms)
    ks = np.array([len(term_genes[t]) for t in terms])
    xs = np.array([len(term_genes[t] & genes) for t in terms])
    n = len(genes)
    p_values = hypergeom.sf(xs - 1, n_universe, ks, n)
    _, adjusted, _, _ = multipletests(p_values, method="fdr_bh")
    best = int(np.argmin(adjusted))
    return float(adjusted[best]), terms[best], len(terms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--modules", type=int, default=100)
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "stability_go.log", "09_stability_go")
    started = time.time()
    out_dir = paths.out("stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"))
    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"))
    tiers = pd.read_parquet(paths.out("audit", "disease_tiers.parquet"))[["disease_id", "tier"]]
    modules = modules.merge(tiers, on="disease_id", how="left")
    members = members.merge(tiers, on="disease_id", how="left")

    # ---- 1. 稳定性分层 ----
    by_tier = (
        modules.groupby("tier")
        .agg(n_modules=("module_id", "size"),
             stability_median=("stability_mean_jaccard", "median"),
             stability_q25=("stability_mean_jaccard", lambda s: s.quantile(0.25)),
             stability_q75=("stability_mean_jaccard", lambda s: s.quantile(0.75)),
             size_median=("module_size", "median"),
             hub_median=("hub_dominance", "median"))
        .reindex(["T1", "T2", "T3"])
        .round(3)
    )
    by_tier["frac_stability_ge_0.6"] = [
        round(float((modules.loc[modules["tier"] == tier, "stability_mean_jaccard"] >= 0.6).mean()), 3)
        if (modules["tier"] == tier).any() else None
        for tier in by_tier.index
    ]
    by_tier.to_csv(out_dir / "stability_by_tier.csv", encoding="utf-8-sig")

    # ---- 2. 核心/外围 ----
    members["role"] = np.where(
        members["consensus_frequency"] >= CORE_THRESHOLD, "core",
        np.where(members["consensus_frequency"] >= PERIPHERY_THRESHOLD, "periphery", "unstable"),
    )
    core_stats = (
        members.groupby(["module_id", "role"])["gene_id"].size()
        .unstack(fill_value=0)
        .reindex(columns=["core", "periphery", "unstable"], fill_value=0)
    )
    core_stats["module_size"] = core_stats.sum(axis=1)
    core_stats["core_fraction"] = (core_stats["core"] / core_stats["module_size"]).round(3)
    core_stats = core_stats.join(modules.set_index("module_id")[["tier", "stability_mean_jaccard"]])
    core_stats.to_parquet(out_dir / "core_composition.parquet")

    role_scores = (
        members.groupby("role")
        .agg(n=("gene_id", "size"),
             raw_rwr_median=("raw_rwr_score", "median"),
             freq_median=("consensus_frequency", "median"))
        .round(5)
    )
    role_scores.to_csv(out_dir / "role_scores.csv", encoding="utf-8-sig")

    # ---- 3. GO 一致性：核心 vs 完整模块 vs 同大小随机 ----
    universe_df = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))
    universe = set(universe_df["gene_id"].astype(str))
    term_genes, gene_terms = go_index(paths, universe)
    n_universe = len(universe)
    log.info("go_index_built", terms=len(term_genes), genes=len(gene_terms), universe=n_universe)

    rng = np.random.default_rng(args.seed)
    module_ids = sorted(core_stats.index)
    if len(module_ids) > args.modules:
        module_ids = list(rng.choice(np.array(module_ids), size=args.modules, replace=False))

    member_map = members.groupby("module_id").apply(
        lambda frame: {
            "core": set(frame.loc[frame["role"] == "core", "gene_id"].astype(str)),
            "full": set(frame["gene_id"].astype(str)),
        },
        include_groups=False,
    ).to_dict()

    rows = []
    gene_pool = np.array(sorted(universe))
    for module_id in module_ids:
        sets = member_map.get(module_id)
        if not sets or not sets["full"]:
            continue
        size = len(sets["full"])
        core_fdr, core_term, _ = best_fdr(sets["core"], term_genes, gene_terms, n_universe)
        full_fdr, full_term, _ = best_fdr(sets["full"], term_genes, gene_terms, n_universe)
        random_fdrs = []
        for _ in range(args.random_draws):
            sample = set(rng.choice(gene_pool, size=size, replace=False).tolist())
            fdr, _, _ = best_fdr(sample, term_genes, gene_terms, n_universe)
            if np.isfinite(fdr):
                random_fdrs.append(fdr)
        rows.append({
            "module_id": module_id,
            "module_size": size,
            "core_size": len(sets["core"]),
            "core_best_fdr": core_fdr,
            "core_best_term": core_term,
            "full_best_fdr": full_fdr,
            "full_best_term": full_term,
            "random_best_fdr_median": float(np.median(random_fdrs)) if random_fdrs else float("nan"),
            "random_frac_fdr_le_0.05": float(np.mean(np.array(random_fdrs) <= 0.05)) if random_fdrs else float("nan"),
        })
        if len(rows) % 25 == 0:
            log.info("go_progress", done=len(rows), total=len(module_ids))

    go_table = pd.DataFrame(rows)
    go_table.to_parquet(out_dir / "go_consistency.parquet", index=False)

    def frac_le(frame, column, threshold=0.05):
        values = frame[column].dropna()
        return round(float((values <= threshold).mean()), 3) if len(values) else None

    summary = {
        "stability": {
            "modules": int(len(modules)),
            "by_tier": by_tier.reset_index().to_dict(orient="records"),
            "frac_stability_ge_0.6": round(float((modules["stability_mean_jaccard"] >= 0.6).mean()), 3),
        },
        "core": {
            "modules": int(len(core_stats)),
            "core_fraction_median": round(float(core_stats["core_fraction"].median()), 3),
            "modules_with_core": int((core_stats["core"] > 0).sum()),
        },
        "go": {
            "modules_evaluated": int(len(go_table)),
            "core_frac_fdr_le_0.05": frac_le(go_table, "core_best_fdr"),
            "full_frac_fdr_le_0.05": frac_le(go_table, "full_best_fdr"),
            "random_frac_fdr_le_0.05_median": round(float(go_table["random_frac_fdr_le_0.05"].median()), 3)
            if len(go_table) else None,
            "core_better_than_full": int((go_table["core_best_fdr"] < go_table["full_best_fdr"]).sum()),
            "core_better_than_random": int((go_table["core_best_fdr"] < go_table["random_best_fdr_median"]).sum()),
        },
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
