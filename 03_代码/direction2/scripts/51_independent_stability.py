"""独立扰动下的复现率：给共识模块与基线模块一个公平的稳定性比较。

共识模块的「稳定性」用的是构建它的那 100 次扰动的共现，属于自证。
这里另跑 10 次全新扰动（不同随机数种子、不同删术语比例），对每个模块算复现率：
新扰动里存在某个模块、覆盖它至少 60% 成员的比例。

比较对象：15 个疾病上的共识模块（119 个）与同疾病的 V0 基线模块。
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

RUNS = 10
COVER = 0.6


def main() -> int:
    global RUNS, COVER, COVERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--covers", default="0.6")
    args = parser.parse_args()

    RUNS = args.runs
    COVERS = [float(x) for x in args.covers.split(',') if x.strip()]
    COVER = COVERS[0]
    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "independent_stability.log", "51_independent_stability")
    started = time.time()
    out_dir = paths.out("stability_independent", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores = None
    from phenotype_modules_v0.scores.provider import load_saved_scores as _load
    matrix = _load(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))

    consensus = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"))
    consensus["gene_id"] = consensus["gene_id"].astype(str)
    consensus_sets = consensus.groupby("module_id")["gene_id"].apply(set).to_dict()

    baseline = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    baseline_members = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    baseline_members["gene_id"] = baseline_members["gene_id"].astype(str)
    baseline_sets = baseline_members.groupby("module_id")["gene_id"].apply(set).to_dict()

    diseases = sorted(set(consensus["disease_id"].astype(str)))
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                        node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    rows = []
    for disease in diseases:
        terms = list(terms_of.get(disease, set()))
        if len(terms) < 5:
            continue
        module_sets = (
            [(mid, s, "consensus") for mid, s in consensus_sets.items()
             if mid in set(consensus.loc[consensus.disease_id.astype(str) == disease, "module_id"])]
            + [(bid, s, "baseline") for bid, s in baseline_sets.items()
               if bid in set(baseline.loc[baseline.disease_id.astype(str) == disease, "module_id"])]
        )
        if not module_sets:
            continue

        runs_partitions = []
        for run in range(RUNS):
            kept = [t for t in terms if rng.random() > 0.10]
            indices = [hpo_node[t] for t in kept if t in hpo_node]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[:200]
            candidate = [gene_ids[i] for i in top]
            detections = detect(tuple(candidate), ppi, seed=stable_seed(args.seed, args.fold, disease, run))
            runs_partitions.append([set(m) for m in detections])

        for module_id, members, kind in module_sets:
            if len(members) < 5:
                continue
            n_runs = max(len(runs_partitions), 1)
            for cover in COVERS:
                covered = 0
                for partition in runs_partitions:
                    if any(len(members & m) / len(members) >= cover for m in partition):
                        covered += 1
                rows.append({"module_id": module_id, "disease_id": disease, "kind": kind,
                             "cover": cover, "size": len(members),
                             "replication": covered / n_runs})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "independent_stability.parquet", index=False)
    summary = table.groupby(["kind", "cover"]).agg(modules=("module_id", "size"),
                                        replication_median=("replication", "median"),
                                        replication_mean=("replication", "mean")).round(3)
    summary.to_csv(out_dir / "independent_stability.csv", encoding="utf-8-sig")
    payload = {"fold": args.fold, "runs": RUNS, "cover": COVER,
               "table": summary.reset_index().to_dict(orient="records"),
               "seconds": round(time.time() - started, 2)}
    (out_dir / "independent_stability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
