"""M3-2：复现 V0 的「RWR top-K + Leiden」基线，并与既有模块表逐成员比对。

验收标准是模块集合完全一致，不是"跑通了"。任何差异都写进 comparison.json，不静默跳过。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import (  # noqa: E402
    detect_base_modules,
    drop_single_hub_modules,
    load_v0_models,
    module_sets,
)
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_fold_scores, load_saved_scores  # noqa: E402


def stratified_diseases(tiers: pd.DataFrame, fold: int, per_tier: dict[str, int], rng: np.random.Generator) -> list[str]:
    pool = tiers.loc[tiers["fold"] == fold]
    chosen: list[str] = []
    for tier, count in per_tier.items():
        candidates = np.array(sorted(pool.loc[pool["tier"] == tier, "disease_id"].astype(str)))
        if len(candidates) == 0:
            continue
        take = min(count, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return sorted(chosen)


def v0_modules_for(paths, fold: int) -> dict[str, set[frozenset[str]]]:
    modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"),
                              columns=["module_id", "disease_id", "fold"])
    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"),
                              columns=["module_id", "disease_id", "gene_id"])
    modules = modules.loc[modules["fold"] == fold]
    keep = set(modules["module_id"].astype(str))
    members = members.loc[members["module_id"].astype(str).isin(keep)]
    grouped = members.groupby("module_id")["gene_id"].apply(lambda s: frozenset(map(str, s)))
    result: dict[str, set[frozenset[str]]] = {}
    for module_id, members_set in grouped.items():
        disease = str(module_id).split(":")[0] + ":" + str(module_id).split(":")[1]
        result.setdefault(disease, set()).add(frozenset(members_set))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--t1", type=int, default=2)
    parser.add_argument("--t2", type=int, default=4)
    parser.add_argument("--t3", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "baselines.log", "02_baselines")
    started = time.time()

    tiers = pd.read_parquet(paths.out("audit", "disease_tiers.parquet"))
    diseases = stratified_diseases(tiers, args.fold, {"T1": args.t1, "T2": args.t2, "T3": args.t3}, rng)
    log.info("sampled_diseases", fold=args.fold, n=len(diseases), diseases=",".join(diseases))

    matrix_path = paths.out("scores", f"fold_{args.fold}", "score_matrix.npz")
    matrix = load_saved_scores(matrix_path) if matrix_path.exists() else load_fold_scores(paths.v0_root, args.fold)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    detect, topology = load_v0_models(paths.v0_root)

    genes_by_disease = {disease: matrix.top_k(disease, args.top_k) for disease in diseases}
    modules = detect_base_modules(
        diseases, genes_by_disease, ppi, detect=detect, seed_base=args.seed, fold=args.fold,
    )
    kept, rejected = drop_single_hub_modules(modules, ppi, topology=topology)

    rows = []
    for disease in diseases:
        for index, members in enumerate(sorted(kept[disease], key=lambda m: (-len(m), tuple(map(int, m))))):
            info = topology(members, ppi)
            rows.append({
                "module_id": f"{disease}:leiden:{args.fold}:M{index + 1:02d}",
                "disease_id": disease,
                "fold": args.fold,
                "method": "leiden_topk",
                "module_size": len(members),
                **{k: info[k] for k in ("internal_edge_count", "internal_density", "mean_internal_weight", "hub_dominance")},
            })
    table = pd.DataFrame(rows)
    out_dir = paths.out("baselines", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "modules.parquet", index=False)

    member_rows = [
        {"module_id": f"{disease}:leiden:{args.fold}:M{index + 1:02d}", "disease_id": disease,
         "fold": args.fold, "gene_id": gene}
        for disease in diseases
        for index, members in enumerate(sorted(kept[disease], key=lambda m: (-len(m), tuple(map(int, m)))))
        for gene in members
    ]
    pd.DataFrame(member_rows).to_parquet(out_dir / "module_members.parquet", index=False)

    reference = v0_modules_for(paths, args.fold)
    ours = module_sets(kept)
    comparison = []
    for disease in diseases:
        mine = ours.get(disease, set())
        theirs = reference.get(disease, set())
        comparison.append({
            "disease_id": disease,
            "n_ours": len(mine),
            "n_v0": len(theirs),
            "identical": mine == theirs,
            "only_ours": len(mine - theirs),
            "only_v0": len(theirs - mine),
        })
    identical = sum(1 for c in comparison if c["identical"])
    summary = {
        "fold": args.fold,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_diseases": len(diseases),
        "n_identical": identical,
        "n_modules_ours": int(sum(c["n_ours"] for c in comparison)),
        "n_modules_v0": int(sum(c["n_v0"] for c in comparison)),
        "n_rejected_single_hub": len(rejected),
        "seconds": round(time.time() - started, 2),
        "diseases": comparison,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"diseases={len(diseases)} identical={identical} modules_ours={summary['n_modules_ours']} "
          f"modules_v0={summary['n_modules_v0']} rejected={len(rejected)} seconds={summary['seconds']}")
    if identical != len(diseases):
        print("mismatched diseases:")
        for row in comparison:
            if not row["identical"]:
                print(f"  {row['disease_id']}: ours={row['n_ours']} v0={row['n_v0']} "
                      f"only_ours={row['only_ours']} only_v0={row['only_v0']}")
    log.info("comparison", fold=args.fold, identical=identical, total=len(diseases),
             seconds=summary["seconds"])
    log.close()
    return 0 if identical == len(diseases) else 1


if __name__ == "__main__":
    sys.exit(main())
