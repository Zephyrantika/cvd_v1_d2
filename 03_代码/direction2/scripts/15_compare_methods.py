"""补齐对照实验：与 Louvain、DIAMOnD 在同一候选集上比较。

三种模块发现器，输入完全相同（fold 0、同一批 19 个疾病、同一份 RWR top-200、同一张 STRING 700 子图）：
    leiden      V0 的基线（已有）
    louvain     igraph 的 community_multilevel，分辨率扫描后取默认
    diamond     以训练图表型—基因边推出的基因为种子，按超几何显著性生长

评价只报结构特征与特异性（z_excess），不报基因恢复——T1 只有 9 个疾病，恢复类指标没有统计意义。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.evaluation.specificity import background_by_similarity, phenotype_sets  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency, density_and_hub  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def louvain_modules(genes: tuple[str, ...], ppi: pd.DataFrame, minimum_size: int, maximum_size: int) -> list[tuple[str, ...]]:
    index = {g: i for i, g in enumerate(genes)}
    selected = ppi.loc[ppi["gene_a"].astype(str).isin(index) & ppi["gene_b"].astype(str).isin(index)]
    graph = ig.Graph(n=len(genes), directed=False)
    graph.vs["name"] = list(genes)
    if not selected.empty:
        graph.add_edges([(index[str(r.gene_a)], index[str(r.gene_b)]) for r in selected.itertuples(index=False)])
        graph.es["weight"] = selected["weight"].astype(float).tolist()
    if graph.ecount() == 0:
        return []
    membership = graph.community_multilevel(weights="weight" if graph.es else None)
    modules = []
    for cluster in membership:
        members = tuple(sorted((genes[i] for i in cluster), key=int))
        if minimum_size <= len(members) <= maximum_size:
            modules.append(members)
    return sorted(modules, key=lambda m: (-len(m), tuple(map(int, m))))


def diamond_module(
    genes: tuple[str, ...],
    ppi: pd.DataFrame,
    seeds: set[str],
    full_degrees: pd.Series,
    universe_size: int,
    maximum_size: int = 100,
    alpha: float = 0.05,
) -> tuple[tuple[str, ...], int]:
    index = {g: i for i, g in enumerate(genes)}
    adjacency: dict[str, set[str]] = {g: set() for g in genes}
    for a, b in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), strict=True):
        if a in index and b in index:
            adjacency[a].add(b)
            adjacency[b].add(a)
    module = {g for g in seeds if g in index}
    if not module:
        return (), 0

    added = 0
    while len(module) < maximum_size:
        candidates = set()
        for gene in module:
            candidates |= adjacency[gene]
        candidates -= module
        if not candidates:
            break
        best, best_p = None, 1.0
        for candidate in sorted(candidates):
            k = len(adjacency[candidate] & module)
            if k == 0:
                continue
            degree = float(full_degrees.get(candidate, 0.0))
            if degree <= 0:
                continue
            p_value = float(hypergeom.sf(k - 1, universe_size, degree, len(module)))
            if p_value < best_p:
                best, best_p = candidate, p_value
        if best is None or best_p > alpha:
            break
        module.add(best)
        added += 1
    return tuple(sorted(module, key=int)), added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "compare_methods.log", "15_compare_methods")
    started = time.time()
    out_dir = paths.out("compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)
    degrees = pd.Series({g: float(sum(v.values())) for g, v in adjacency.items()})
    percentiles = matrix.percentiles()
    gene_position = {g: i for i, g in enumerate(matrix.gene_ids)}
    baseline_modules = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "modules.parquet"))
    base_members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    base_by_disease = (
        base_members.groupby(["disease_id", "module_id"])["gene_id"]
        .apply(lambda s: tuple(sorted(set(map(str, s)), key=int)))
        .groupby("disease_id")
        .apply(list)
        .to_dict()
    )

    phenotype_edges = pd.read_parquet(
        paths.v0(f"data/folds/fold_{args.fold}/phenotype_gene_edges.parquet"),
        columns=["hpo_id", "gene_id"],
    )
    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id"],
    )
    disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(map(str, s))).to_dict()
    term_genes: dict[str, set[str]] = {}
    for term, gene in zip(phenotype_edges["hpo_id"].astype(str), phenotype_edges["gene_id"].astype(str), strict=True):
        term_genes.setdefault(term, set()).add(gene)

    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "fold", "disease_family_id"])
    family = dict(zip(folds["disease_id"].astype(str), folds["disease_family_id"].astype(str)))
    fold_diseases = tuple(sorted(matrix.disease_ids))
    pheno_sets = phenotype_sets(pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"), columns=["disease_id", "hpo_id"]))

    rows = []
    for disease in sorted(baseline_modules["disease_id"].astype(str).unique()):
        genes = matrix.top_k(disease, args.top_k)
        bg = background_by_similarity(disease, fold_diseases, pheno_sets)
        bg_rows = [matrix.disease_index(d) for d in bg if d in matrix.disease_ids]
        own_row = matrix.disease_index(disease)

        seeds = set()
        for term in disease_terms.get(disease, set()):
            seeds |= term_genes.get(term, set())
        seeds = {g for g in seeds if g in gene_position}

        candidates = {
            "leiden": [tuple(sorted(set(map(str, m)), key=int)) for m in base_by_disease.get(disease, [])],
            "louvain": louvain_modules(genes, ppi, 5, 100),
            "diamond": [diamond_module(genes, ppi, seeds, degrees, len(matrix.gene_ids))[0]],
        }
        for method, modules in candidates.items():
            modules = [m for m in modules if len(m) >= 5]
            for index, members in enumerate(modules):
                indices = [gene_position[g] for g in members if g in gene_position]
                if len(indices) < 5:
                    continue
                score_self = float(percentiles[own_row, indices].mean())
                background = np.array([percentiles[row, indices].mean() for row in bg_rows])
                z = (score_self - float(background.mean())) / (float(background.std(ddof=1)) + 1e-9)
                density, hub = density_and_hub(list(members), adjacency)
                rows.append({
                    "method": method,
                    "disease_id": disease,
                    "module_index": index,
                    "module_size": len(members),
                    "density": density,
                    "hub_dominance": hub,
                    "z_excess": z,
                    "n_seeds": len(seeds) if method == "diamond" else np.nan,
                })
        log.info("disease_done", disease=disease, seeds=len(seeds))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / f"methods_fold_{args.fold}.parquet", index=False)

    summary = (
        table.groupby("method")
        .agg(modules=("module_id" if "module_id" in table else "module_size", "size"),
             size_median=("module_size", "median"),
             density_median=("density", "median"),
             hub_median=("hub_dominance", "median"),
             z_median=("z_excess", "median"),
             z_q75=("z_excess", lambda s: s.quantile(0.75)))
        .round(3)
    )
    summary.to_csv(out_dir / f"methods_summary_fold_{args.fold}.csv", encoding="utf-8-sig")
    payload = {
        "fold": args.fold,
        "top_k": args.top_k,
        "diseases": int(table["disease_id"].nunique()) if len(table) else 0,
        "table": summary.reset_index().to_dict(orient="records"),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / f"methods_summary_fold_{args.fold}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
