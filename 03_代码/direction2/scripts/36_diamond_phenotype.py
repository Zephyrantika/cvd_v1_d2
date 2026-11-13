"""E6 方法对照之一：DIAMOnD 按表型种子生长。

文档 §4.2：DIAMOnD 的种子用训练图表型边推出。这里取疾病 HPO 术语在训练图里连到的基因，
按边权和排序取前 50 个作种子，在 STRING 700 上按超几何显著性生长到 200 个基因。

评价与 35 号的分级模块同口径：z_spec（IC 相似度 20 个背景）、GO FDR、density、hub、稳定性。
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
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
SEED_SIZE = 50
MAX_SIZE = 200
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
    parser.add_argument("--limit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-size", type=int, default=200)
    parser.add_argument("--seed-size", type=int, default=50)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "diamond_phenotype.log", "36_diamond_phenotype")
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
    disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(s)).to_dict()

    edges = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/phenotype_gene_edges.parquet"),
                            columns=["hpo_id", "gene_id", "weight"])
    edges["hpo_id"] = edges["hpo_id"].astype(str)
    edges["gene_id"] = edges["gene_id"].astype(str)
    term_genes = {term: dict(zip(g.gene_id, g.weight.astype(float))) for term, g in edges.groupby("hpo_id")}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
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
    degree_counts = np.diff(adjacency.indptr)

    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "fold", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))
    norms = {d: float(np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms))) for d, terms in disease_terms.items()}

    go = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                         columns=["gene_id", "go_id", "evidence_code"])
    go["gene_id"] = go["gene_id"].astype(str)
    go = go.loc[~go["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)]
    term_genes_go: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(go["gene_id"], go["go_id"], strict=True):
        term_genes_go.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes_go = {t: g for t, g in term_genes_go.items() if MIN_TERM <= len(g) <= MAX_TERM}
    gene_terms = {g: [t for t in ts if t in term_genes_go] for g, ts in gene_terms.items()}

    def go_fdr(genes: set[str]) -> float:
        terms: set[str] = set()
        for gene in genes:
            terms.update(gene_terms.get(gene, ()))
        if not terms:
            return float("nan")
        ordered = sorted(terms)
        ks = np.array([len(term_genes_go[t]) for t in ordered])
        xs = np.array([len(term_genes_go[t] & genes) for t in ordered])
        return float(multipletests(hypergeom.sf(xs - 1, len(universe), ks, len(genes)), method="fdr_bh")[1].min())

    diseases = sorted(matrix.disease_ids)
    if args.limit:
        keep_path = paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet")
        if keep_path.exists():
            keep = set(pd.read_parquet(keep_path)["disease_id"].astype(str))
            diseases = [d for d in diseases if d in keep]

    rows = []
    member_rows = []
    for disease in diseases:
        score: dict[str, float] = {}
        for term in disease_terms.get(disease, set()):
            for gene, weight in term_genes.get(term, {}).items():
                score[gene] = score.get(gene, 0.0) + float(weight)
        seeds = [index[g] for g, _ in sorted(score.items(), key=lambda kv: -kv[1])[:args.seed_size] if g in index]
        if len(seeds) < 5:
            continue
        added = diamond(adjacency, set(seeds), degree_counts, args.max_size)
        members = [universe[i] for i in set(seeds) | set(added)]

        bg = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = disease_terms.get(disease, set()) & disease_terms.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                norms.get(disease, 1.0) * norms.get(other, 1.0) + 1e-9)
            bg.append((sim, other))
        bg = [d for _, d in sorted(bg, reverse=True)[:BACKDROP]]
        idx = [position[g] for g in members if g in position]
        self_score = float(percentiles[matrix.disease_index(disease), idx].mean())
        background = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in bg])
        z_spec = (self_score - background.mean()) / (background.std(ddof=1) + 1e-9)

        for gene in members:
            member_rows.append({"module_id": f"{disease}:diamond", "disease_id": disease, "gene_id": gene,
                                "role": "member"})
        rows.append({
            "disease_id": disease,
            "method": "diamond_phenotype",
            "size": len(members),
            "z_spec": float(z_spec),
            "go_fdr": go_fdr(set(members)),
            "n_seeds": len(seeds),
        })
        log.info("disease_done", disease=disease, size=len(members))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "diamond_phenotype.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out_dir / "diamond_members.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": int(len(table)),
        "mean_size": round(float(table["size"].mean()), 1),
        "z_spec_median": round(float(table["z_spec"].median()), 3),
        "z_spec_gt_2": round(float((table["z_spec"] > 2).mean()), 3),
        "go_fdr_median": float(table["go_fdr"].median()),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "diamond_phenotype.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
