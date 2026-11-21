"""按文档口径计算 core Jaccard 稳定性，并与复现率并排对比（fold 3）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DIR = Path(__file__).resolve().parents[1]
COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    args = parser.parse_args()

    parts = pd.read_parquet(DIR / f"outputs/partitions/fold_{args.fold}/partitions.parquet")
    parts["gene_id"] = parts["gene_id"].astype(str)
    parts["disease_id"] = parts["disease_id"].astype(str)

    modules_by_run: dict[tuple[str, int], list[set[str]]] = {}
    for (disease, run), group in parts.groupby(["disease_id", "run"]):
        mods = [set(block["gene_id"]) for _, block in group.groupby("module_index")]
        modules_by_run[(disease, run)] = mods

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    refined = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0/lam_refine_members.parquet")
    refined["gene_id"] = refined["gene_id"].astype(str)
    r1 = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    r1["gene_id"] = r1["gene_id"].astype(str)

    def sets_of(frame: pd.DataFrame, key: str) -> dict[str, set[str]]:
        return frame.groupby(key)["gene_id"].apply(set).to_dict()

    candidates = {
        "baseline": {d: set(list(base_sets.get(m, set()))[:25]) for d, m in largest.items()},
        "refined": sets_of(refined, "disease_id"),
        "r1": sets_of(r1, "disease_id"),
    }

    rows = []
    for disease in sorted(candidates["baseline"]):
        runs = [k[1] for k in modules_by_run if k[0] == disease]
        if len(runs) < 10:
            continue
        for name, mapping in candidates.items():
            module = mapping.get(disease)
            if not module or len(module) < 5:
                continue
            jac, cov = [], []
            for run in runs:
                best_j, best_c = 0.0, 0.0
                for other in modules_by_run[(disease, run)]:
                    inter = len(module & other)
                    if not inter:
                        continue
                    best_j = max(best_j, inter / len(module | other))
                    best_c = max(best_c, inter / len(module))
                jac.append(best_j)
                cov.append(best_c)
            rows.append({"disease_id": disease, "variant": name,
                         "jaccard": float(np.mean(jac)),
                         "replication": float(np.mean(np.array(cov) >= COVER))})

    table = pd.DataFrame(rows)
    out = DIR / f"outputs/jaccard_stability/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "jaccard_stability.parquet", index=False)

    pivot = table.pivot_table(index="disease_id", columns="variant", values=["jaccard", "replication"])
    summary = {}
    for metric in ("jaccard", "replication"):
        for a, b in (("baseline", "refined"), ("baseline", "r1")):
            pair = pivot[metric][[a, b]].dropna()
            summary[f"{metric}_{a}_vs_{b}"] = {
                "n": int(len(pair)),
                "a": round(float(pair[a].mean()), 4),
                "b": round(float(pair[b].mean()), 4),
                "wins": int((pair[b] > pair[a]).sum()),
                "p": float(wilcoxon(pair[b], pair[a]).pvalue),
            }
    payload = {"fold": args.fold, "runs": int(parts["run"].nunique()), "summary": summary}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
