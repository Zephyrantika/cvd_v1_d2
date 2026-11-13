"""E6 补充：ROBUST 式多样性枚举（表型种子）。

与 26 号脚本同一套替代实现：prize 用疾病 RWR 分数，cost 用负对数权重，随机起点 + prize 抖动，
保留 Jaccard < 0.85 的解，按出现频率排序取前 25 个基因，与 DIAMOnD、Leiden 同大小比较 z_spec 与 GO。
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
from scipy import sparse
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
SIZE = 25
RUNS = 15
DIVERSITY = 0.85
BACKDROP = 20


def grow(adjacency: sparse.csr_matrix, costs: np.ndarray, prizes: np.ndarray,
         start: set[int], max_size: int) -> set[int]:
    n = adjacency.shape[0]
    in_module = np.zeros(n, dtype=bool)
    in_module[list(start)] = True
    best_cost = np.full(n, np.inf)
    for node in start:
        lo, hi = adjacency.indptr[node], adjacency.indptr[node + 1]
        np.minimum.at(best_cost, adjacency.indices[lo:hi], costs[lo:hi])
    module = set(start)
    while len(module) < max_size:
        score = prizes - 0.5 * np.where(np.isfinite(best_cost), best_cost, 1e9)
        score[in_module] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]) or score[best] <= 0:
            break
        in_module[best] = True
        module.add(best)
        lo, hi = adjacency.indptr[best], adjacency.indptr[best + 1]
        np.minimum.at(best_cost, adjacency.indices[lo:hi], costs[lo:hi])
    return module


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "robust_phenotype.log", "41_robust_phenotype")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe = list(matrix.gene_ids)
    index = {g: i for i, g in enumerate(universe)}
    rows, cols, data = [], [], []
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        rows += [ia, ib]
        cols += [ib, ia]
        data += [w, w]
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(len(universe), len(universe)))
    max_weight = float(ppi["weight"].max())
    costs = np.empty(adjacency.data.shape[0])
    cache: dict[float, float] = {}
    for position_, weight in enumerate(adjacency.data):
        key = float(weight)
        if key not in cache:
            cache[key] = float(-np.log(np.clip(key / max_weight, 1e-6, 1.0)))
        costs[position_] = cache[key]

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
    universe_size = len(universe)

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
        return float(multipletests(hypergeom.sf(xs - 1, universe_size, ks, len(target)), method="fdr_bh")[1].min())

    diseases = sorted(matrix.disease_ids)
    if args.limit:
        keep_path = paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet")
        if keep_path.exists():
            keep = set(pd.read_parquet(keep_path)["disease_id"].astype(str))
            diseases = [d for d in diseases if d in keep]

    rows_out = []
    member_rows = []
    for disease in diseases:
        row = percentiles[matrix.disease_index(disease)]
        prizes = row / (row.max() + 1e-12)
        seeds = set(np.argsort(-row)[:50].tolist())
        solutions: list[set[int]] = []
        for _ in range(RUNS):
            start = set(rng.choice(np.array(sorted(seeds)), size=max(3, len(seeds) // 3), replace=False).tolist())
            jitter = prizes * rng.lognormal(0.0, 0.25, size=prizes.shape[0])
            jitter = jitter / (jitter.max() + 1e-12)
            candidate = grow(adjacency, costs, jitter, start, SIZE)
            if all(jaccard(candidate, existing) < DIVERSITY for existing in solutions):
                solutions.append(candidate)
        frequency: dict[int, int] = {}
        for solution in solutions:
            for gene in solution:
                frequency[gene] = frequency.get(gene, 0) + 1
        ranked = [g for g, _ in sorted(frequency.items(), key=lambda kv: -kv[1])][:SIZE]
        members = [universe[i] for i in ranked]

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
        cols = [position[g] for g in members if g in position]
        self_score = float(percentiles[matrix.disease_index(disease), cols].mean())
        background = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in bg])
        z = float((self_score - background.mean()) / (background.std(ddof=1) + 1e-9))
        for gene in members:
            member_rows.append({"module_id": f"{disease}:robust", "disease_id": disease, "gene_id": gene})
        rows_out.append({"disease_id": disease, "size": len(cols), "z_spec": z,
                         "go_fdr": go_fdr(members), "n_solutions": len(solutions)})

    table = pd.DataFrame(rows_out)
    pd.DataFrame(member_rows).to_parquet(out_dir / "robust_members.parquet", index=False)
    table.to_parquet(out_dir / "robust_phenotype.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": int(len(table)),
        "z_spec_median": round(float(table["z_spec"].median()), 3),
        "go_fdr_median": float(table["go_fdr"].median()),
        "solutions_mean": round(float(table["n_solutions"].mean()), 1),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "robust_phenotype.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
