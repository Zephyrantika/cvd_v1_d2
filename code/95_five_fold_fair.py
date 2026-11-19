"""五折汇总：R1 选点 vs 公平基线（允许在 15/20/25/30/40 里选点）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto"


def main() -> int:
    rows, per_fold = [], []
    for fold in range(5):
        fair_path = DIR / f"outputs/refined_grading/fold_{fold}_base_fair/refined_graded.parquet"
        r1_path = DIR / f"outputs/refined_grading/fold_{fold}_r1/refined_graded.parquet"
        if not (fair_path.exists() and r1_path.exists()):
            print(f"[warn] fold {fold} incomplete")
            continue
        fair = pd.read_parquet(fair_path)[["disease_id", "grade"]].rename(columns={"grade": "grade_fair"})
        r1 = pd.read_parquet(r1_path)[["disease_id", "grade", "p_spec", "replication"]]
        merged = fair.merge(r1, on="disease_id")
        merged["fold"] = fold
        rows.append(merged)
        per_fold.append({"fold": fold, "diseases": int(len(merged)),
                         "fair_A": int((merged.grade_fair == "A").sum()),
                         "r1_A": int((merged.grade == "A").sum())})

    table = pd.concat(rows, ignore_index=True)
    table["fA"] = table.grade_fair == "A"
    table["rA"] = table.grade == "A"
    tab = pd.crosstab(table.fA, table.rA)
    payload = {"diseases": int(len(table)),
               "fair_A": int(table.fA.sum()), "r1_A": int(table.rA.sum()),
               "fair_frac": round(float(table.fA.mean()), 4),
               "r1_frac": round(float(table.rA.mean()), 4),
               "gained": int(((~table.fA) & table.rA).sum()),
               "lost": int((table.fA & (~table.rA)).sum()),
               "mcnemar_p": float(mcnemar(tab, exact=True).pvalue),
               "per_fold": per_fold}
    (OUT / "five_fold_fair.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
