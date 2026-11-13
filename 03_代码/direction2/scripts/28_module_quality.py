"""模块质量：功能一致性与扰动稳定性。

对每个性状：固定一次 70/30 拆分，用 RWR 取前 50 个非种子基因作为模块，然后
    1. GO 超几何富集（排除 IEA），看最佳 FDR，和同大小随机集比；
    2. 删掉 10% 的 PPI 边后重跑，看模块 Jaccard。
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
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
TOP_K = 50


def build_graph(ppi: pd.DataFrame, genes: list[str]):
    index = {g: i for i, g in enumerate(genes)}
    rows, cols, data = [], [], []
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        rows += [ia, ib]
        cols += [ib, ia]
        data += [w, w]
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(genes), len(genes))), index


def rwr(adjacency: sparse.csr_matrix, seeds: set[int], alpha: float = 0.3, max_iter: int = 100) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    transition = (sparse.diags(1.0 / degree) @ adjacency).T
    seed_vector = np.zeros(adjacency.shape[0])
    seed_vector[list(seeds)] = 1.0 / len(seeds)
    scores = seed_vector.copy()
    for _ in range(max_iter):
        updated = alpha * seed_vector + (1 - alpha) * (transition @ scores)
        if np.abs(updated - scores).sum() < 1e-9:
            return updated
        scores = updated
    return scores


def best_fdr(genes: set[str], term_genes: dict[str, set[str]], gene_terms: dict[str, list[str]], universe: int):
    if not genes:
        return float("nan"), None
    terms: set[str] = set()
    for gene in genes:
        terms.update(gene_terms.get(gene, ()))
    if not terms:
        return float("nan"), None
    ordered = sorted(terms)
    ks = np.array([len(term_genes[t]) for t in ordered])
    xs = np.array([len(term_genes[t] & genes) for t in ordered])
    p = hypergeom.sf(xs - 1, universe, ks, len(genes))
    adjusted = multipletests(p, method="fdr_bh")[1]
    index = int(np.argmin(adjusted))
    return float(adjusted[index]), ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "module_quality.log", "28_module_quality")
    started = time.time()
    out_dir = paths.out("gene_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = pd.read_parquet(paths.out("external", "trait_gene_sets.parquet"))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    universe_list = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
    universe = set(universe_list)
    adjacency, index = build_graph(ppi, universe_list)

    annotations = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                                  columns=["gene_id", "go_id", "evidence_code"])
    annotations["gene_id"] = annotations["gene_id"].astype(str)
    annotations = annotations.loc[annotations["gene_id"].isin(universe)
                                  & ~annotations["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(annotations["gene_id"], annotations["go_id"], strict=True):
        term_genes.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes = {t: g for t, g in term_genes.items() if MIN_TERM <= len(g) <= MAX_TERM}
    gene_terms = {g: [t for t in terms if t in term_genes] for g, terms in gene_terms.items()}
    log.info("go_ready", terms=len(term_genes))

    rows = []
    for trait, group in sets.groupby("trait"):
        genes = [g for g in group["gene_id"].astype(str) if g in index]
        if len(genes) < 20:
            continue
        order = rng.permutation(len(genes))
        cut = int(len(genes) * 0.7)
        seed_genes = {index[genes[i]] for i in order[:cut]}
        held = {index[genes[i]] for i in order[cut:]}

        scores = rwr(adjacency, seed_genes)
        scores[list(seed_genes)] = -np.inf
        top = list(np.argsort(-scores)[:TOP_K])
        module_names = {universe_list[i] for i in top}
        fdr, term = best_fdr(module_names, term_genes, gene_terms, len(universe_list))

        random_fdrs = []
        for _ in range(10):
            sample = set(rng.choice(np.array(sorted(universe - set(universe_list[i] for i in seed_genes))),
                                    size=TOP_K, replace=False).tolist())
            random_fdr, _ = best_fdr(sample, term_genes, gene_terms, len(universe_list))
            if np.isfinite(random_fdr):
                random_fdrs.append(random_fdr)

        # 稳定性：删 10% 边后重跑
        coo = adjacency.tocoo()
        upper = coo.row < coo.col
        rows_e, cols_e, data_e = coo.row[upper], coo.col[upper], coo.data[upper]
        keep = rng.random(len(rows_e)) >= 0.10
        r2 = np.concatenate([rows_e[keep], cols_e[keep]])
        c2 = np.concatenate([cols_e[keep], rows_e[keep]])
        d2 = np.concatenate([data_e[keep], data_e[keep]])
        damaged = sparse.csr_matrix((d2, (r2, c2)), shape=adjacency.shape)
        scores2 = rwr(damaged, seed_genes)
        scores2[list(seed_genes)] = -np.inf
        top2 = set(np.argsort(-scores2)[:TOP_K])
        overlap = len(set(top) & top2) / TOP_K

        rows.append({
            "trait": trait,
            "module_size": TOP_K,
            "held_out_recall": len(set(top) & held) / len(held),
            "go_best_fdr": fdr,
            "go_best_term": term,
            "random_fdr_median": float(np.median(random_fdrs)) if random_fdrs else float("nan"),
            "stability_jaccard_10pct_edges": round(overlap, 3),
        })
        log.info("trait_done", trait=trait, fdr=round(fdr, 4) if np.isfinite(fdr) else None)

    table = pd.DataFrame(rows).sort_values("held_out_recall", ascending=False)
    table.to_csv(out_dir / "module_quality.csv", index=False, encoding="utf-8-sig")
    payload = {
        "traits": table.to_dict(orient="records"),
        "frac_go_fdr_le_0.05": round(float((table["go_best_fdr"] <= 0.05).mean()), 3),
        "median_stability": round(float(table["stability_jaccard_10pct_edges"].median()), 3),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "module_quality.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(table.to_string(index=False))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
