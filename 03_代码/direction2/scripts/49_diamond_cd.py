"""重跑 DIAMOnD：把生长限制在候选集 C_d 内（top-200）。

之前 36 号脚本在全部 16,992 个基因上生长，会跑到候选集外，把模块稀释；
文献说明（literature/04_方向二模块.md）要求「必须在同一候选集上跑」。
这里只保留 C_d 内的节点和边，再比较 z_spec。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25
SEED_SIZE = 10
BACKDROP = 20


def diamond(adjacency: sparse.csr_matrix, seeds: set[int], degree_counts: np.ndarray,
            max_size: int, alpha: float = 0.05, top_m: int = 50) -> list[int]:
    n = adjacency.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    in_module = np.zeros(n, dtype=bool)
    for seed in seeds:
        in_module[seed] = True
        counts[adjacency.indices[adjacency.indptr[seed]:adjacency.indptr[seed + 1]]] += 1
    added: list[int] = []
    while len(added) + len(seeds) < max_size:
        candidates = np.flatnonzero((counts > 0) & (~in_module))
        if candidates.size == 0:
            break
        k = counts[candidates]
        if candidates.size > top_m:
            keep = np.argsort(-k)[:top_m]
            candidates, k = candidates[keep], k[keep]
        p = hypergeom.sf(k - 1, n, degree_counts[candidates].clip(min=1), len(seeds) + len(added))
        best = int(np.argmin(p))
        if p[best] > alpha:
            break
        node = int(candidates[best])
        in_module[node] = True
        added.append(node)
        counts[adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]] += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "diamond_cd.log", "49_diamond_cd")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    edges = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/phenotype_gene_edges.parquet"),
                            columns=["hpo_id", "gene_id", "weight"])
    edges["hpo_id"] = edges["hpo_id"].astype(str)
    edges["gene_id"] = edges["gene_id"].astype(str)
    term_genes = {t: dict(zip(g.gene_id, g.weight.astype(float))) for t, g in edges.groupby("hpo_id")}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))

    def z_of(disease: str, genes: list[str]) -> float:
        cols = [position[g] for g in genes if g in position]
        if not cols:
            return np.nan
        bg = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            bg.append((sim, other))
        bg = [d for _, d in sorted(bg, reverse=True)[:BACKDROP]]
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in bg])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    modules = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    largest = modules.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    baseline = {}
    for disease, module_id in largest.items():
        genes = list(members.loc[members.module_id == module_id].sort_values("r_d", ascending=False)["gene_id"])[:SIZE]
        if len(genes) >= 5:
            baseline[disease] = z_of(disease, genes)

    rows = []
    for disease in matrix.disease_ids:
        candidate = list(matrix.top_k(disease, 200))
        index = {g: i for i, g in enumerate(candidate)}
        rows_e, cols_e, data_e = [], [], []
        for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
            ia, ib = index.get(a), index.get(b)
            if ia is None or ib is None:
                continue
            rows_e += [ia, ib]
            cols_e += [ib, ia]
            data_e += [w, w]
        adjacency = sparse.csr_matrix((data_e, (rows_e, cols_e)), shape=(len(candidate), len(candidate)))
        degree_counts = np.diff(adjacency.indptr)

        score: dict[str, float] = {}
        for term in terms_of.get(disease, set()):
            for gene, weight in term_genes.get(term, {}).items():
                if gene in index:
                    score[gene] = score.get(gene, 0.0) + float(weight)
        seeds = [index[g] for g, _ in sorted(score.items(), key=lambda kv: -kv[1])[:SEED_SIZE] if g in index]
        if len(seeds) < 5:
            continue
        added = diamond(adjacency, set(seeds), degree_counts, SIZE)
        genes = [candidate[i] for i in set(seeds) | set(added)]
        rows.append({"disease_id": disease, "z_diamond": z_of(disease, genes),
                     "z_baseline": baseline.get(disease, np.nan), "size": len(genes)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "diamond_cd.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": int(len(table)),
        "z_diamond_median": round(float(table.z_diamond.median()), 3),
        "z_baseline_median": round(float(table.z_baseline.median()), 3),
        "wins": int((table.z_diamond > table.z_baseline).sum()),
        "p": float(wilcoxon(table.z_diamond, table.z_baseline).pvalue),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "diamond_cd.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
