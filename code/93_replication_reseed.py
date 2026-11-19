"""换一批随机种子的独立扰动复现率：检验 R1 的选点是否只对原来那批扰动有效。

输入：两个模块集合（基线 / R1），同一折。
输出：每个疾病两种模块的复现率（覆盖 60% 成员）。
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "reseed_replication.log", "93_reseed")
    started = time.time()
    out_dir = paths.out("reseed", f"fold_{args.fold}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    base_mem = pd.read_parquet(paths.out("pareto", f"base_members_fold{args.fold}.parquet"))
    r1_mem = pd.read_parquet(paths.out("pareto_r1", f"members_fold{args.fold}.parquet"))
    sets = {}
    for name, frame in (("baseline", base_mem), ("r1", r1_mem)):
        frame["gene_id"] = frame["gene_id"].astype(str)
        sets[name] = frame.groupby("disease_id")["gene_id"].apply(set).to_dict()

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_index = {str(t): int(i) for t, i in zip(node_map.loc[node_map["node_type"] == "phenotype", "node_id"],
                                                node_map.loc[node_map["node_type"] == "phenotype", "node_index"])}
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    rows = []
    for disease in sorted(set(sets["baseline"]) & set(sets["r1"])):
        terms = terms_of.get(disease, [])
        if len(terms) < 5:
            continue
        partitions = []
        for run in range(args.runs):
            kept = [t for t in terms if rng.random() > 0.10]
            indices = [hpo_index[t] for t in kept if t in hpo_index]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[:200]
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(args.seed, args.fold, disease, run))
            partitions.append([set(m) for m in detections])
        for name in ("baseline", "r1"):
            module = sets[name][disease]
            covered = sum(1 for p in partitions if any(len(module & m) / len(module) >= COVER for m in p))
            rows.append({"disease_id": disease, "variant": name,
                         "replication_reseed": covered / max(len(partitions), 1)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "reseed.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant", values="replication_reseed").dropna()
    payload = {"fold": args.fold, "seed": args.seed, "runs": args.runs, "diseases": int(len(pivot)),
               "baseline": round(float(pivot.baseline.mean()), 3),
               "r1": round(float(pivot.r1.mean()), 3),
               "wins": int((pivot.r1 > pivot.baseline).sum()),
               "losses": int((pivot.r1 < pivot.baseline).sum()),
               "p": float(wilcoxon(pivot.r1, pivot.baseline).pvalue),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "reseed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
