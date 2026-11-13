"""E1：raw 与 calibrated 两套分数的对照（文档 §6）。

calibrated 分数来自 V0 的 outputs/rankings/fold_*/calibrated/rankings.parquet，
用 empirical_percentile 作为 r_d；同样取 top-200、Leiden、算 z_spec（IC 背景 20）与 GO。
与 raw 的模块逐疾病比较。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
BACKDROP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "e1_calibrated.log", "40_e1_calibrated")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    detect, _ = load_v0_models(paths.v0_root)
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    rank = pd.read_parquet(paths.v0(f"outputs/rankings/fold_{args.fold}/calibrated/rankings.parquet"),
                           columns=["disease_id", "gene_id", "empirical_percentile", "corrected_rank"])
    rank["disease_id"] = rank["disease_id"].astype(str)
    rank["gene_id"] = rank["gene_id"].astype(str)
    percentiles = rank.pivot(index="disease_id", columns="gene_id", values="empirical_percentile")
    percentiles = percentiles.fillna(0.0)
    genes = list(percentiles.columns)
    position = {g: i for i, g in enumerate(genes)}

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))

    go = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                         columns=["gene_id", "go_id", "evidence_code"])
    go["gene_id"] = go["gene_id"].astype(str)
    go = go.loc[~go["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(go["gene_id"], go["go_id"], strict=True):
        term_genes.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes = {t: g for t, g in term_genes.items() if MIN_TERM <= len(g) <= MAX_TERM}
    gene_terms = {g: [t for t in ts if t in term_genes] for g, ts in gene_terms.items()}
    universe = len(genes)

    def go_fdr(members: list[str]) -> float:
        target = set(members)
        terms: set[str] = set()
        for gene in target:
            terms.update(gene_terms.get(gene, ()))
        if not terms:
            return float("nan")
        ordered = sorted(terms)
        ks = np.array([len(term_genes[t]) for t in ordered])
        xs = np.array([len(term_genes[t] & target) for t in ordered])
        return float(multipletests(hypergeom.sf(xs - 1, universe, ks, len(target)), method="fdr_bh")[1].min())

    diseases = sorted(percentiles.index)
    rows = []
    for disease in diseases:
        row = percentiles.loc[disease].to_numpy()
        idx = np.argsort(-row, kind="stable")[:200]
        genes_top = tuple(genes[i] for i in idx)
        modules = detect(genes_top, ppi, seed=stable_seed(args.seed, args.fold, disease, "e1"))
        bg = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            bg.append((sim, other))
        bg = [d for _, d in sorted(bg, reverse=True)[:BACKDROP]]
        for members in modules:
            if len(members) < 5:
                continue
            cols = [position[g] for g in members if g in position]
            if not cols:
                continue
            own = float(row[cols].mean())
            background = np.array([percentiles.loc[b].to_numpy()[cols].mean() for b in bg])
            z = (own - background.mean()) / (background.std(ddof=1) + 1e-9)
            rows.append({"disease_id": disease, "size": len(cols), "z_spec": float(z),
                         "go_fdr": go_fdr(list(members))})

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "e1_calibrated.parquet", index=False)

    raw = pd.read_parquet(paths.out("modules_doc", "modules.parquet"))
    raw_members = pd.read_parquet(paths.out("modules_doc", "members.parquet"))
    raw_members["gene_id"] = raw_members["gene_id"].astype(str)
    raw_groups = raw_members.groupby("module_id")["gene_id"].apply(list).to_dict()
    raw["go_fdr"] = [go_fdr(raw_groups.get(m, [])) for m in raw["module_id"]]
    summary = {
        "fold": args.fold,
        "calibrated_modules": int(len(table)),
        "calibrated_z_median": round(float(table["z_spec"].median()), 3),
        "calibrated_go_median": float(table["go_fdr"].median()),
        "raw_modules": int(len(raw)),
        "raw_z_median": round(float(raw["z_spec"].median()), 3),
        "raw_go_median": float(raw["go_fdr"].median()),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "e1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
