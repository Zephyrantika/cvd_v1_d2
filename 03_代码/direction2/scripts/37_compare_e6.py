"""E6 的配对比较：同疾病、同大小下，DIAMOnD 与 Leiden 共识模块的 z_spec 与 GO。

DIAMOnD 用 10 个表型种子生长到 25 个基因；Leiden 模块按 r_d 取前 25 个成员做同样大小。
配对的 Wilcoxon 检验。
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
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
SIZE = 25
BACKDROP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "compare_e6.log", "37_compare_e6")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    universe = len(matrix.gene_ids)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()

    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))
    norms = {d: float(np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms))) for d, terms in disease_terms.items()}

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

    def go_fdr(genes: list[str]) -> float:
        target = set(genes)
        terms: set[str] = set()
        for gene in target:
            terms.update(gene_terms.get(gene, ()))
        if not terms:
            return float("nan")
        ordered = sorted(terms)
        ks = np.array([len(term_genes[t]) for t in ordered])
        xs = np.array([len(term_genes[t] & target) for t in ordered])
        return float(multipletests(hypergeom.sf(xs - 1, universe, ks, len(target)), method="fdr_bh")[1].min())

    def z_of(disease: str, genes: list[str], diseases: list[str]) -> float:
        idx = [position[g] for g in genes if g in position]
        if not idx:
            return float("nan")
        bg = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = disease_terms.get(disease, set()) & disease_terms.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                norms.get(disease, 1.0) * norms.get(other, 1.0) + 1e-9)
            bg.append((sim, other))
        bg = [d for _, d in sorted(bg, reverse=True)[:BACKDROP]]
        self_score = float(percentiles[matrix.disease_index(disease), idx].mean())
        background = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in bg])
        return float((self_score - background.mean()) / (background.std(ddof=1) + 1e-9))

    diamond_members = pd.read_parquet(out_dir / "diamond_members.parquet")
    diamond_members["gene_id"] = diamond_members["gene_id"].astype(str)
    diamond_sets = diamond_members.groupby("disease_id")["gene_id"].apply(lambda s: list(s)[:SIZE]).to_dict()
    diseases = sorted(diamond_sets)

    leiden_members = pd.read_parquet(paths.out("modules_doc", "members.parquet"))
    leiden_members["gene_id"] = leiden_members["gene_id"].astype(str)
    leiden_modules = pd.read_parquet(paths.out("modules_doc", "modules.parquet"))
    best_module = (
        leiden_modules.sort_values("z_spec", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    )

    rows = []
    for disease in diseases:
        if disease not in best_module:
            continue
        module_id = best_module[disease]
        leiden_genes = list(leiden_members.loc[leiden_members.module_id == module_id]
                            .sort_values("r_d", ascending=False)["gene_id"])[:SIZE]
        diamond_genes = list(diamond_sets[disease])[:SIZE]
        if len(leiden_genes) < 5 or len(diamond_genes) < 5:
            continue
        rows.append({
            "disease_id": disease,
            "z_diamond": z_of(disease, diamond_genes, diseases),
            "z_leiden": z_of(disease, leiden_genes, diseases),
            "go_diamond": go_fdr(diamond_genes),
            "go_leiden": go_fdr(leiden_genes),
        })

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "e6_size_matched.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": int(len(table)),
        "size": SIZE,
        "z_diamond_median": round(float(table["z_diamond"].median()), 3),
        "z_leiden_median": round(float(table["z_leiden"].median()), 3),
        "go_diamond_median": float(table["go_diamond"].median()),
        "go_leiden_median": float(table["go_leiden"].median()),
        "wilcoxon_z_p": float(wilcoxon(table["z_diamond"], table["z_leiden"]).pvalue) if len(table) >= 5 else None,
        "diamond_better_go": int((table["go_diamond"] < table["go_leiden"]).sum()),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "e6_size_matched.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary)
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
