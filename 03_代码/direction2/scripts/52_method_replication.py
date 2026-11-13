"""五种模块集的独立扰动复现率对比：基线、共识、DIAMOnD、ROBUST 式、Heinz 贪心。

另跑 20 次全新扰动（未参与任何模块构建），对每个模块算复现率：
新扰动中存在某个模块、覆盖它至少 x% 成员的比例。三个覆盖率阈值一起报。
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

RUNS = 20
COVERS = (0.5, 0.6, 0.7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "method_replication.log", "52_method_replication")
    started = time.time()
    out_dir = paths.out("method_replication", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    from phenotype_modules_v0.scores.provider import load_saved_scores
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))

    def load_sets(path: Path, kind: str) -> dict[str, tuple[str, set[str]]]:
        frame = pd.read_parquet(path)
        frame["gene_id"] = frame["gene_id"].astype(str)
        if "disease_id" not in frame.columns:
            mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"),
                                   columns=["module_id", "disease_id"])
            frame = frame.merge(mods, on="module_id", how="left")
        frame["disease_id"] = frame["disease_id"].astype(str)
        return {module_id: (str(group["disease_id"].iloc[0]), set(group["gene_id"]))
                for module_id, group in frame.groupby("module_id")}

    sources = {
        "baseline": paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"),
        "consensus": paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"),
        "diamond": paths.out("methods", f"fold_{args.fold}", "diamond_members.parquet"),
        "robust": paths.out("methods", f"fold_{args.fold}", "robust_members.parquet"),
        "heinz": paths.out("methods", f"fold_{args.fold}", "heinz_members.parquet"),
    }
    sets_by_kind = {kind: load_sets(path, kind) for kind, path in sources.items() if path.exists()}
    log.info("loaded", **{k: len(v) for k, v in sets_by_kind.items()})

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
    for disease in matrix.disease_ids:
        terms = list(terms_of.get(disease, set()))
        if len(terms) < 5:
            continue
        module_sets = [(mid, members, kind) for kind, table in sets_by_kind.items()
                       for mid, (d, members) in table.items() if d == disease and len(members) >= 5]
        if not module_sets:
            continue

        partitions = []
        for run in range(args.runs):
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
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(args.seed, args.fold, disease, run))
            partitions.append([set(m) for m in detections])

        n_runs = max(len(partitions), 1)
        for module_id, members, kind in module_sets:
            for cover in COVERS:
                covered = sum(1 for p in partitions
                              if any(len(members & m) / len(members) >= cover for m in p))
                rows.append({"module_id": module_id, "disease_id": disease, "kind": kind,
                             "cover": cover, "size": len(members), "replication": covered / n_runs})
        log.info("disease_done", disease=disease, modules=len(module_sets))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "method_replication.parquet", index=False)
    summary = (table.groupby(["kind", "cover"])
               .agg(modules=("module_id", "size"), mean=("replication", "mean"),
                    median=("replication", "median"))
               .round(3).reset_index())
    summary.to_csv(out_dir / "method_replication.csv", index=False, encoding="utf-8-sig")
    payload = {"fold": args.fold, "runs": args.runs,
               "table": summary.to_dict(orient="records"),
               "seconds": round(time.time() - started, 2)}
    (out_dir / "method_replication.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.pivot(index="kind", columns="cover", values="mean").to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
