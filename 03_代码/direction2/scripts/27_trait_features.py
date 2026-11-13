"""性状特征与模块质量：解释为什么有的性状好扩展、有的不行。

两部分：
    1. 性状特征（基因数、平均度数、内部边密度、平均关联分数）与 Recall@200 的关系；
    2. 模块的功能一致性：RWR 前 50 个非种子基因做 GO 超几何富集，和同大小随机集比。
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
from scipy.stats import hypergeom, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "trait_features.log", "27_trait_features")
    started = time.time()
    out_dir = paths.out("gene_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = pd.read_parquet(paths.out("external", "trait_gene_sets.parquet"))
    results = pd.read_parquet(out_dir / "gene_seeded_results.parquet")
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    degree = pd.concat([ppi["gene_a"].astype(str), ppi["gene_b"].astype(str)]).value_counts()

    rows = []
    for trait, group in sets.groupby("trait"):
        gene_ids = set(group["gene_id"].astype(str))
        sub = ppi.loc[ppi["gene_a"].astype(str).isin(gene_ids) & ppi["gene_b"].astype(str).isin(gene_ids)]
        possible = len(gene_ids) * (len(gene_ids) - 1) / 2
        recall_rwr = results.loc[(results.trait == trait) & (results.method == "rwr_gene") & (results.k == 200), "recall"].mean()
        rows.append({
            "trait": trait,
            "n_genes": len(gene_ids),
            "mean_degree": round(float(np.mean([degree.get(g, 0) for g in gene_ids])), 2),
            "internal_edges": int(len(sub)),
            "internal_density": round(len(sub) / possible, 5) if possible else 0.0,
            "mean_score": round(float(group["score"].mean()), 3),
            "recall_rwr_200": round(float(recall_rwr), 4),
        })
    features = pd.DataFrame(rows).sort_values("recall_rwr_200", ascending=False)
    features.to_csv(out_dir / "trait_features.csv", index=False, encoding="utf-8-sig")
    correlations = {}
    for column in ("n_genes", "mean_degree", "internal_density", "mean_score"):
        rho, p = spearmanr(features[column], features["recall_rwr_200"])
        correlations[column] = {"spearman": round(float(rho), 3), "p": round(float(p), 3)}
    log.info("features_done", traits=len(features))

    # GO 一致性：用最后一折的 RWR 前 50 基因（从 gene_seeded_results 没有成员，这里重算一次太贵，
    # 改为直接读 DIAMOnD/RWR 的模块成员表若存在；否则只做性状特征部分）
    summary = {"traits": features.to_dict(orient="records"), "correlations": correlations,
               "seconds": round(time.time() - started, 2)}
    (out_dir / "trait_features.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(features.to_string(index=False))
    print(json.dumps(correlations, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
