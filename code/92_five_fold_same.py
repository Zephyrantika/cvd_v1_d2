"""五折同口径对比：R1 选点 vs 基线（都用 100 次零模型的 p + 独立扰动复现率）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto"


def main() -> int:
    rows = []
    per_fold = []
    for fold in range(5):
        base_path = DIR / f"outputs/refined_grading/fold_{fold}_base_same/refined_graded.parquet"
        r1_path = DIR / f"outputs/refined_grading/fold_{fold}_r1/refined_graded.parquet"
        if not (base_path.exists() and r1_path.exists()):
            print(f"[warn] fold {fold} incomplete")
            continue
        base = pd.read_parquet(base_path)[["disease_id", "grade"]].rename(columns={"grade": "grade_base"})
        r1 = pd.read_parquet(r1_path)[["disease_id", "grade", "z_spec", "replication"]]
        merged = base.merge(r1, on="disease_id")
        merged["fold"] = fold
        rows.append(merged)
        per_fold.append({"fold": fold, "diseases": int(len(merged)),
                         "base_A": int((merged.grade_base == "A").sum()),
                         "r1_A": int((merged.grade == "A").sum())})

    table = pd.concat(rows, ignore_index=True)
    table["bA"] = table.grade_base == "A"
    table["rA"] = table.grade == "A"
    tab = pd.crosstab(table.bA, table.rA)
    payload = {"diseases": int(len(table)),
               "base_A": int(table.bA.sum()), "r1_A": int(table.rA.sum()),
               "base_frac": round(float(table.bA.mean()), 4),
               "r1_frac": round(float(table.rA.mean()), 4),
               "gained": int(((~table.bA) & table.rA).sum()),
               "lost": int((table.bA & (~table.rA)).sum()),
               "mcnemar_p": float(mcnemar(tab, exact=True).pvalue),
               "per_fold": per_fold}
    (OUT / "five_fold_same_protocol.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
