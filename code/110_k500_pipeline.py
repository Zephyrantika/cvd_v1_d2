"""迭代二：K=500 候选集操作点。

产出两套模块（baseline@500 与精修@500 由 60 号脚本生成 baseline 侧需要重算复现率）：
  1. top-500 → Leiden → 最大模块 → 取前 25（按分位）
  2. 20 次独立扰动的复现率
输出 members 与 selected，供 73 号脚本分级。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "k500.log", "110_k500")
    started = time.time()
    out = paths.out("k500", f"fold_{args.fold}_top{args.topk}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    hpo_rows = node_map.loc[node_map["node_type"].eq("phenotype")]
    hpo_index = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    members: dict[str, set[str]] = {}
    for disease in test_ids:
        di = matrix.disease_index(disease)
        order = np.argsort(-percentiles[di, :])[: args.topk]
        candidate = tuple(matrix.gene_ids[i] for i in order)
        detections = detect(candidate, ppi, seed=stable_seed(110, args.fold, disease, args.topk))
        if not detections:
            continue
        largest = max(detections, key=len)
        ranked = sorted(largest, key=lambda g: -percentiles[di, position[g]])[: args.size]
        members[disease] = set(ranked)
    log.info("modules_ready", n=len(members))

    partitions: dict[str, list[list[set[str]]]] = {}
    for disease in members:
        parts = []
        for run in range(args.runs):
            kept = [t for t in terms_of.get(disease, []) if rng.random() > 0.10]
            indices = [hpo_index[t] for t in kept if t in hpo_index]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[: args.topk]
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(110, args.fold, disease, run))
            parts.append([set(m) for m in detections])
        partitions[disease] = parts
    log.info("partitions_ready")

    rows, member_rows = [], []
    for disease, module in members.items():
        parts = partitions.get(disease, [])
        covered = sum(1 for p in parts if any(len(module & m) / len(module) >= COVER for m in p))
        rows.append({"disease_id": disease, "replication": covered / max(len(parts), 1)})
        for gene in module:
            member_rows.append({"module_id": f"{disease}:k{args.topk}", "disease_id": disease, "gene_id": gene})

    pd.DataFrame(rows).to_parquet(out / "selected.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out / "members.parquet", index=False)
    summary = {"fold": args.fold, "topk": args.topk, "diseases": len(members),
               "rep_mean": round(float(pd.DataFrame(rows).replication.mean()), 3),
               "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
