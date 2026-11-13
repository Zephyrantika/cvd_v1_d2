"""浅层重排器：用多特征把 RWR 排序重新打一遍。

训练用该折的训练疾病（需要为它们单独跑一次 RWR，V0 只存了测试疾病的排名），
评价用该折的测试疾病，对照同一折的 RWR 基线。

特征：rwr 分位、log 分数、excess（分位减全折均值）、log 度数、度数分位、
表型—基因边权和、命中术语数、通路成员数、组织表达分位。

训练：正样本 = 训练疾病的已知基因，负样本 = 每病按度数分层抽 150 个未标注基因。
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
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency, gene_degrees  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

FEATURES = ["rwr_pct", "rwr_log", "excess", "degree_log", "degree_pct",
            "hpo_weight", "hpo_hits", "pathway_log", "expr_pct"]


def percentiles_of(scores: np.ndarray) -> np.ndarray:
    """行内（疾病内）分位，1 = 最高分。"""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return 1.0 - (ranks - 1) / (scores.shape[1] - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--negatives", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "reranker.log", "21_reranker")
    started = time.time()
    out_dir = paths.out("reranker")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)
    degrees = gene_degrees(adjacency)
    pathways = pd.read_parquet(paths.v0("data/processed/reactome_gene_pathway.parquet"),
                               columns=["gene_id", "pathway_id"])
    pathway_count = pathways.assign(gene_id=pathways["gene_id"].astype(str)).groupby("gene_id")["pathway_id"].nunique()
    expr = pd.read_parquet(paths.v0("data/processed/gtex_cardiovascular_expression.parquet"),
                           columns=["gene_id", "median_tpm"])
    expr["gene_id"] = expr["gene_id"].astype(str)
    expr_scores = expr.assign(v=np.log1p(expr["median_tpm"].clip(lower=0))).groupby("gene_id")["v"].mean()
    expr_pct = expr_scores.rank(pct=True).to_dict()

    genes_table = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"), columns=["disease_id", "gene_id"]
    ).drop_duplicates()
    genes_table["disease_id"] = genes_table["disease_id"].astype(str)
    genes_table["gene_id"] = genes_table["gene_id"].astype(str)
    known = genes_table.groupby("disease_id")["gene_id"].apply(lambda s: set(s)).to_dict()

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(s)).to_dict()

    rows = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        matrix = load_saved_scores(paths.out("scores", f"fold_{fold}", "score_matrix.npz"))
        gene_ids = matrix.gene_ids
        position = {g: i for i, g in enumerate(gene_ids)}
        n_genes = len(gene_ids)

        node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{fold}/node_map.parquet"))
        gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
        graph_gene_ids = gene_rows["node_id"].astype(str).tolist()
        gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
        order = [graph_gene_ids.index(g) for g in gene_ids]
        transition = load_npz(paths.v0(f"data/graphs/fold_{fold}/transition_matrix.npz")).tocsr()
        hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/train_hpo_ic.parquet"))
        alpha = float(json.loads(
            paths.v0(f"outputs/rankings/fold_{fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
        )["selected_alpha"])

        train_diseases = [d for d in sorted(known) if d in set(node_map["node_id"]) or True]
        train_ids = [
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/train_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        test_ids = [
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_diseases = [d for d in train_ids if known.get(d)]
        test_diseases = [d for d in test_ids]

        seeds, _ = v0_rwr.build_seed_matrix(tuple(train_diseases), phenotypes, hpo_ic, node_map)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        train_scores = result.scores[gene_indices, :].T[:, order]
        train_pct = percentiles_of(train_scores)

        test_pct = matrix.percentiles()
        bg_mean = np.vstack([train_pct, test_pct]).mean(axis=0)

        edges = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/phenotype_gene_edges.parquet"),
                                columns=["hpo_id", "gene_id", "weight"])
        edges["hpo_id"] = edges["hpo_id"].astype(str)
        edges["gene_id"] = edges["gene_id"].astype(str)
        weight_by_term = {term: dict(zip(g.gene_id, g.weight.astype(float)))
                          for term, g in edges.groupby("hpo_id")}

        degree_vec = np.array([float(degrees.get(g, 0.0)) for g in gene_ids])
        degree_pct = pd.Series(degree_vec).rank(pct=True).to_numpy()
        pathway_vec = np.array([np.log1p(float(pathway_count.get(g, 0))) for g in gene_ids])
        expr_vec = np.array([expr_pct.get(g, 0.5) for g in gene_ids])
        degree_log = np.log1p(degree_vec)

        def hpo_vectors(disease: str) -> tuple[np.ndarray, np.ndarray]:
            weight = np.zeros(n_genes, dtype=np.float32)
            hits = np.zeros(n_genes, dtype=np.float32)
            for term in disease_terms.get(disease, set()):
                for gene, value in weight_by_term.get(term, {}).items():
                    idx = position.get(gene)
                    if idx is not None:
                        weight[idx] += float(value)
                        hits[idx] += 1.0
            return weight, hits

        def feature_block(disease: str, pct_row: np.ndarray, log_scores: np.ndarray) -> np.ndarray:
            weight, hits = hpo_vectors(disease)
            return np.column_stack([
                pct_row, np.log1p(log_scores), pct_row - bg_mean,
                degree_log, degree_pct, weight, hits, pathway_vec, expr_vec,
            ]).astype(np.float32)

        X_train = np.vstack([feature_block(d, train_pct[i], train_scores[i])
                             for i, d in enumerate(train_diseases)])
        y_train = np.zeros(len(X_train), dtype=np.int8)
        for i, disease in enumerate(train_diseases):
            for gene in known[disease]:
                idx = position.get(gene)
                if idx is not None:
                    y_train[i * n_genes + idx] = 1

        keep = np.zeros(len(X_train), dtype=bool)
        keep[y_train == 1] = True
        unlabeled = np.flatnonzero(y_train == 0)
        for i in range(len(train_diseases)):
            block = unlabeled[(unlabeled >= i * n_genes) & (unlabeled < (i + 1) * n_genes)]
            picks = rng.choice(block, size=min(args.negatives, len(block)), replace=False)
            keep[picks] = True

        scaler = StandardScaler().fit(X_train[keep])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.transform(X_train[keep]), y_train[keep])

        X_test = np.vstack([feature_block(d, test_pct[matrix.disease_index(d)], matrix.scores[matrix.disease_index(d)])
                            for d in test_diseases])
        scores = model.predict_proba(scaler.transform(X_test))[:, 1]

        mrr_model, mrr_base, hits = [], [], []
        for i, disease in enumerate(test_diseases):
            genes = known.get(disease, set())
            if not genes:
                continue
            for values, bucket in ((scores[i * n_genes:(i + 1) * n_genes], mrr_model),
                                   (test_pct[matrix.disease_index(disease)], mrr_base)):
                order_idx = np.argsort(-values, kind="stable")
                pos = {gene_ids[j]: r + 1 for r, j in enumerate(order_idx)}
                ranks = [pos[g] for g in genes if g in pos]
                if ranks:
                    bucket.append(float(np.mean([1.0 / r for r in ranks])))
                    if values is not test_pct[matrix.disease_index(disease)]:
                        hits.append(float(min(ranks) <= 10))
        rows.append({
            "fold": fold,
            "mrr_reranker": round(float(np.mean(mrr_model)), 4),
            "mrr_rwr": round(float(np.mean(mrr_base)), 4),
            "recall_at_10_reranker": round(float(np.mean(hits)), 4),
            "coef": {name: round(float(c), 3) for name, c in zip(FEATURES, model.coef_[0], strict=True)},
        })
        log.info("fold_done", fold=fold, mrr_reranker=rows[-1]["mrr_reranker"], mrr_rwr=rows[-1]["mrr_rwr"])

    payload = {
        "folds": args.folds,
        "rows": rows,
        "mean_mrr_reranker": round(float(np.mean([r["mrr_reranker"] for r in rows])), 4),
        "mean_mrr_rwr": round(float(np.mean([r["mrr_rwr"] for r in rows])), 4),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "reranker_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
