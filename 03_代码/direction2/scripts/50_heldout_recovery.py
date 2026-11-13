"""留出基因回收测试：用 T1/T2 疾病的已知基因做独立评价。

z_spec 由 RWR 分位定义，而 Leiden 模块正是从同一分位的 top-200 里切出来的，
这个指标天然偏向基线。换成独立的评价：把疾病已知基因留出 30%，看模块能找回多少。
已知基因不参与分数构建（V0 已做泄漏控制），所以这是干净的留出评价。

方法：基线（Leiden 最大模块 top-25）与 DIAMOnD（候选集内生长到 25）。
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
from scipy.stats import hypergeom, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25
SEED_SIZE = 10
REPEATS = 5


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "heldout_recovery.log", "50_heldout_recovery")
    started = time.time()
    out_dir = paths.out("heldout", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    detect, _ = load_v0_models(paths.v0_root)
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    genes_table = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_genes.parquet"),
                                  columns=["disease_id", "gene_id"]).drop_duplicates()
    genes_table["disease_id"] = genes_table["disease_id"].astype(str)
    genes_table["gene_id"] = genes_table["gene_id"].astype(str)
    known = genes_table.groupby("disease_id")["gene_id"].apply(set).to_dict()

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

    rows = []
    for disease in matrix.disease_ids:
        labels = known.get(disease, set())
        labels = {g for g in labels if g in position}
        if len(labels) < 3:
            continue
        candidate = list(matrix.top_k(disease, 200))
        index = {g: i for i, g in enumerate(candidate)}
        r_d = percentiles[matrix.disease_index(disease)][[position[g] for g in candidate]]

        rows_e, cols_e, data_e = [], [], []
        for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str),
                           ppi["weight"].astype(float), strict=True):
            ia, ib = index.get(a), index.get(b)
            if ia is None or ib is None:
                continue
            rows_e += [ia, ib]
            cols_e += [ib, ia]
            data_e += [w, w]
        adjacency = sparse.csr_matrix((data_e, (rows_e, cols_e)), shape=(len(candidate), len(candidate)))
        degree_counts = np.diff(adjacency.indptr)

        # 基线模块：C_d 上 Leiden 最大模块，按 r_d 取前 25
        detections = detect(tuple(candidate), ppi, seed=stable_seed(args.seed, args.fold, disease, "heldout"))
        base_genes: list[str] = []
        if detections:
            best = max(detections, key=len)
            base_genes = sorted(best, key=lambda g: -r_d[index[g]])[:SIZE]

        # DIAMOnD 模块：表型边推种子，C_d 内生长
        score: dict[str, float] = {}
        for term in terms_of.get(disease, set()):
            for gene, weight in term_genes.get(term, {}).items():
                if gene in index:
                    score[gene] = score.get(gene, 0.0) + float(weight)
        seeds = [index[g] for g, _ in sorted(score.items(), key=lambda kv: -kv[1])[:SEED_SIZE] if g in index]
        dia_genes: list[str] = []
        if len(seeds) >= 5:
            added = diamond(adjacency, set(seeds), degree_counts, SIZE)
            dia_genes = [candidate[i] for i in set(seeds) | set(added)]

        for repeat in range(REPEATS):
            order = sorted(labels)
            rng.shuffle(order)
            cut = max(1, int(len(order) * 0.3))
            held = set(order[:cut])
            for method, module in (("leiden", set(base_genes)), ("diamond", set(dia_genes))):
                if not module:
                    continue
                rows.append({"disease_id": disease, "repeat": repeat, "method": method,
                             "recall": len(module & held) / len(held),
                             "n_held": len(held), "module_size": len(module)})
        log.info("disease_done", disease=disease, labels=len(labels))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "heldout_recovery.parquet", index=False)
    summary = {}
    for method, group in table.groupby("method"):
        summary[method] = {"recall_mean": round(float(group["recall"].mean()), 4),
                           "recall_median": round(float(group["recall"].median()), 4)}
    pivot = table.pivot_table(index=["disease_id", "repeat"], columns="method", values="recall")
    if {"leiden", "diamond"} <= set(pivot.columns):
        paired = pivot.dropna()
        summary["leiden_wins"] = int((paired["leiden"] > paired["diamond"]).sum())
        summary["diamond_wins"] = int((paired["diamond"] > paired["leiden"]).sum())
        summary["ties"] = int((paired["leiden"] == paired["diamond"]).sum())
        summary["mwu_p"] = float(mannwhitneyu(paired["leiden"], paired["diamond"]).pvalue)
        summary["paired_n"] = int(len(paired))
    summary["diseases"] = int(table.disease_id.nunique())
    summary["seconds"] = round(time.time() - started, 2)
    (out_dir / "heldout_recovery.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
