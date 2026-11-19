"""G3 验收门：空模块 p 值分布检验与假阳性率。

做法：
  1. 对每个疾病抽 N 组随机表型（与真实表型同数量），跑 RWR → top-200 → Leiden → 取最大模块，
     得到「空模块」（不由真实表型驱动）；
  2. 用真实分数分位矩阵给每个空模块算 z_spec（同样的背景疾病集）；
  3. 空模块之间做留一排名得到 p 值——零模型若校准，这些 p 应服从 U(0,1)；
  4. 对 p 分布做 KS 检验（H0: 均匀），并统计 p≤0.01 的比例即假阳性率。
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
from scipy.stats import kstest, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = __import__("phenotype_modules_v0.logging_utils", fromlist=["JsonlLogger"]).JsonlLogger(
        paths.out("qc") / "g3_ks.log", "101_g3_ks")
    started = time.time()
    out_dir = paths.out("g3", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    all_terms = list(hpo_index)
    n_nodes = int(node_map["node_index"].max()) + 1
    transition = load_npz = __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(
        paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    def background(disease: str) -> list[str]:
        scored = []
        for other in test_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_of(disease: str, genes: set[str]) -> float:
        idx = [position[g] for g in genes if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), idx].mean())
        back = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in background(disease)])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    rows = []
    for disease in test_ids:
        size = len(terms_of.get(disease, set()))
        if size < 3:
            continue
        zs = []
        done = 0
        while done < args.draws:
            count = min(args.batch, args.draws - done)
            seeds = np.zeros((n_nodes, count))
            for b in range(count):
                sample = rng.choice(all_terms, size=min(size, len(all_terms)), replace=False)
                indices = [hpo_index[t] for t in sample if t in hpo_index]
                if not indices:
                    continue
                unique = np.unique(indices)
                seeds[unique, b] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
            for b in range(count):
                top = np.argsort(-scores[:, b])[:200]
                candidate = tuple(gene_ids[i] for i in top)
                detections = detect(candidate, ppi, seed=stable_seed(args.seed, args.fold, disease, done + b))
                if not detections:
                    continue
                module = max(detections, key=len)
                value = z_of(disease, set(module))
                if not np.isnan(value):
                    zs.append(value)
            done += count
        if len(zs) < 10:
            continue
        arr = np.array(zs)
        ranks = rankdata(arr)
        pvals = ranks / (len(arr) + 1)          # 留一式 p，零模型下应近似均匀
        rows.extend({"disease_id": disease, "z": float(z), "p": float(p)} for z, p in zip(arr, pvals))
        log.info("disease_done", disease=disease, n=len(arr))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "null_modules.parquet", index=False)
    ks = kstest(table["p"], "uniform") if len(table) > 5 else None
    fpr001 = float((table["p"] <= 0.01).mean())
    fpr005 = float((table["p"] <= 0.05).mean())
    payload = {"fold": args.fold, "draws": args.draws, "null_modules": int(len(table)),
               "diseases": int(table.disease_id.nunique()) if len(table) else 0,
               "ks_stat": round(float(ks.statistic), 4) if ks else None,
               "ks_p": float(ks.pvalue) if ks else None,
               "fpr_001": round(fpr001, 4), "fpr_005": round(fpr005, 4),
               "gate_g3_pass": bool(fpr001 <= 0.05),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "g3.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
