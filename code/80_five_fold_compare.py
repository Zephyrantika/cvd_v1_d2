"""五折合并对比：Pareto 选择 vs 基线（每病最大模块），A 级比例与配对检验。

输入：
  outputs/modules_doc/fold_{N}/modules.parquet        基线模块（含 reliability_tier、z_spec）
  outputs/refined_grading/fold_{N}_pareto/refined_graded.parquet  Pareto 模块分级结果
输出：
  outputs/pareto/five_fold_compare.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto"


def main() -> int:
    frames = []
    per_fold = []
    for fold in range(5):
        base = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/modules.parquet")
        largest = base.sort_values("size", ascending=False).groupby("disease_id").first().reset_index()
        graded_path = DIR / f"outputs/refined_grading/fold_{fold}_pareto/refined_graded.parquet"
        if not graded_path.exists():
            print(f"[warn] fold {fold} missing grading")
            continue
        graded = pd.read_parquet(graded_path)
        merged = largest[["disease_id", "reliability_tier", "z_spec"]].merge(
            graded[["disease_id", "grade", "z_spec", "replication"]], on="disease_id",
            suffixes=("_base", "_pareto"))
        merged["fold"] = fold
        frames.append(merged)
        per_fold.append({
            "fold": fold, "diseases": int(len(merged)),
            "base_A": int((merged.reliability_tier == "A").sum()),
            "pareto_A": int((merged.grade == "A").sum()),
            "base_frac": round(float((merged.reliability_tier == "A").mean()), 3),
            "pareto_frac": round(float((merged.grade == "A").mean()), 3),
        })

    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(OUT / "five_fold_compare.parquet", index=False)
    table["base_A"] = table.reliability_tier == "A"
    table["pareto_A"] = table.grade == "A"
    tab = pd.crosstab(table.base_A, table.pareto_A)
    payload = {
        "diseases": int(len(table)),
        "baseline_A": int(table.base_A.sum()),
        "pareto_A": int(table.pareto_A.sum()),
        "baseline_frac": round(float(table.base_A.mean()), 4),
        "pareto_frac": round(float(table.pareto_A.mean()), 4),
        "gained": int(((~table.base_A) & table.pareto_A).sum()),
        "lost": int((table.base_A & (~table.pareto_A)).sum()),
        "mcnemar_p": float(mcnemar(tab, exact=True).pvalue),
        "base_z_median": round(float(table.z_spec_base.median()), 3),
        "pareto_z_median": round(float(table.z_spec_pareto.median()), 3),
        "pareto_rep_mean": round(float(table.replication.mean()), 3),
        "per_fold": per_fold,
    }
    (OUT / "five_fold_compare.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
