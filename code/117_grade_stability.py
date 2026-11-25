"""等级稳定性：对 100 次扰动做 bootstrap，看重抽样下 A 级判定会不会翻转。

A 级四项判据里只有复现率随扰动变化，其余（p_spec、hub、GO）固定，
因此 P(A) 等于「其余三项已满足」前提下的 P(复现率 ≥ 0.7)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]
COVER = 0.6
REPS = 500


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    parts = pd.read_parquet(DIR / f"outputs/partitions/fold_{args.fold}/partitions.parquet")
    parts["gene_id"] = parts["gene_id"].astype(str)
    parts["disease_id"] = parts["disease_id"].astype(str)
    modules_by_run: dict[tuple[str, int], list[set[str]]] = {}
    for (disease, run), group in parts.groupby(["disease_id", "run"]):
        modules_by_run[(disease, run)] = [set(b["gene_id"]) for _, b in group.groupby("module_index")]

    members = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    members["gene_id"] = members["gene_id"].astype(str)
    members["disease_id"] = members["disease_id"].astype(str)
    modules = members.groupby("disease_id")["gene_id"].apply(set).to_dict()

    graded = pd.read_parquet(DIR / f"outputs/refined_grading/fold_{args.fold}_r1/refined_graded.parquet")
    graded["disease_id"] = graded["disease_id"].astype(str)
    others_ok = dict(zip(graded["disease_id"],
                         (graded["p_spec"] <= 0.01) & (graded["hub"] <= 0.5) & (graded["go_fdr"] <= 0.05)))

    rows = []
    for disease, module in modules.items():
        runs = sorted(r for (d, r) in modules_by_run if d == disease)
        if len(runs) < 20:
            continue
        hits = np.array([1.0 if any(len(module & m) / len(module) >= COVER for m in modules_by_run[(disease, r)])
                         else 0.0 for r in runs])
        idx = rng.integers(0, len(hits), size=(REPS, len(hits)))
        reps = hits[idx].mean(axis=1)
        p_a = float((reps >= 0.7).mean()) if others_ok.get(disease, False) else 0.0
        rows.append({"disease_id": disease, "replication": float(hits.mean()),
                     "p_grade_a": p_a, "others_ok": bool(others_ok.get(disease, False))})

    table = pd.DataFrame(rows)
    out = DIR / f"outputs/grade_stability/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "grade_stability.parquet", index=False)
    unstable = table[(table.p_grade_a > 0.05) & (table.p_grade_a < 0.95)]
    payload = {"fold": args.fold, "diseases": int(len(table)),
               "mean_p_grade_a": round(float(table.p_grade_a.mean()), 4),
               "unstable_fraction": round(float(len(unstable) / max(len(table), 1)), 4),
               "firm_A": int((table.p_grade_a >= 0.95).sum()),
               "never_A": int((table.p_grade_a <= 0.05).sum())}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
