"""表型驱动的另一条走法：先用表型把「锚点基因」选出来，再从基因往外扩展。

原来的做法是把 HPO 术语当种子，在异质图上走，信号要先经过表型节点、再被 PPI 稀释。
这里改成：疾病 → 其在训练图里连接的表型术语 → 这些术语连到的基因（锚点）→
以锚点基因为种子在 PPI 上传播。

如果 MRR 比原来的 0.0462 高，说明表型这条路的问题在"种子放错层"，不是表型没用。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--top-anchors", type=int, default=30)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "phenotype_gene_seeds.log", "30_phenotype_gene_seeds")
    started = time.time()
    out_dir = paths.out("phenotype_seeds")
    out_dir.mkdir(parents=True, exist_ok=True)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    genes_table = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"), columns=["disease_id", "gene_id"]
    ).drop_duplicates()
    genes_table["disease_id"] = genes_table["disease_id"].astype(str)
    genes_table["gene_id"] = genes_table["gene_id"].astype(str)
    known = genes_table.groupby("disease_id")["gene_id"].apply(lambda s: set(s)).to_dict()

    rows = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        matrix = None
        # 基线排名直接读现成的
        rankings = pd.read_parquet(
            paths.v0(f"outputs/rankings/fold_{fold}/rwr/rankings.parquet"),
            columns=["disease_id", "gene_id", "rank"],
        )
        rankings["disease_id"] = rankings["disease_id"].astype(str)
        rankings["gene_id"] = rankings["gene_id"].astype(str)
        rank_lookup = {(r.disease_id, r.gene_id): int(r.rank) for r in rankings.itertuples(index=False)}

        test_ids = [
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        phenotypes = pd.read_parquet(
            paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"), columns=["disease_id", "hpo_id"]
        )
        phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
        phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
        disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(s)).to_dict()

        edges = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/phenotype_gene_edges.parquet"),
                                columns=["hpo_id", "gene_id", "weight"])
        edges["hpo_id"] = edges["hpo_id"].astype(str)
        edges["gene_id"] = edges["gene_id"].astype(str)
        term_genes = {term: dict(zip(g.gene_id, g.weight.astype(float))) for term, g in edges.groupby("hpo_id")}

        universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"))["gene_id"].astype(str).tolist()
        adjacency, index = build_graph(ppi, universe)

        for disease in test_ids:
            genes = known.get(disease, set())
            if not genes:
                continue
            score: dict[str, float] = {}
            for term in disease_terms.get(disease, set()):
                for gene, weight in term_genes.get(term, {}).items():
                    score[gene] = score.get(gene, 0.0) + float(weight)
            anchors = sorted(score.items(), key=lambda kv: -kv[1])[: args.top_anchors]
            anchor_genes = {index[g] for g, _ in anchors if g in index}
            if not anchor_genes:
                continue

            scores = rwr(adjacency, anchor_genes)
            order = np.argsort(-scores)
            position = {universe[i]: r + 1 for r, i in enumerate(order)}
            new_ranks = [position[g] for g in genes if g in position]
            base_ranks = [rank_lookup[(disease, g)] for g in genes if (disease, g) in rank_lookup]
            if not new_ranks or not base_ranks:
                continue
            rows.append({
                "fold": fold,
                "disease_id": disease,
                "n_known": len(genes),
                "n_anchors": len(anchor_genes),
                "mrr_new": float(np.mean([1.0 / r for r in new_ranks])),
                "mrr_base": float(np.mean([1.0 / r for r in base_ranks])),
                "recall10_new": float(any(r <= 10 for r in new_ranks)),
                "recall10_base": float(any(r <= 10 for r in base_ranks)),
            })
        log.info("fold_done", fold=fold)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "phenotype_gene_seeds.parquet", index=False)
    summary = {
        "diseases": int(len(table)),
        "mrr_base": round(float(table["mrr_base"].mean()), 4),
        "mrr_new": round(float(table["mrr_new"].mean()), 4),
        "recall10_base": round(float(table["recall10_base"].mean()), 4),
        "recall10_new": round(float(table["recall10_new"].mean()), 4),
        "mrr_base_t1t2": round(float(table.loc[table.n_known >= 3, "mrr_base"].mean()), 4),
        "mrr_new_t1t2": round(float(table.loc[table.n_known >= 3, "mrr_new"].mean()), 4),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "phenotype_gene_seeds_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
