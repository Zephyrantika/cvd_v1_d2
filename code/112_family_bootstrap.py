"""家族 cluster bootstrap：按疾病家族整簇重抽样，估计 A 级比例差的置信区间。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]
V0 = DIR.parent / "phenotype-network-v0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for fold in range(5):
        base = pd.read_parquet(DIR / f"outputs/refined_grading/fold_{fold}_base_same/refined_graded.parquet")
        r1 = pd.read_parquet(DIR / f"outputs/refined_grading/fold_{fold}_r1/refined_graded.parquet")
        merged = base[["disease_id", "grade"]].rename(columns={"grade": "base"}).merge(
            r1[["disease_id", "grade"]].rename(columns={"grade": "r1"}), on="disease_id")
        merged["fold"] = fold
        rows.append(merged)
    table = pd.concat(rows, ignore_index=True)
    table["disease_id"] = table["disease_id"].astype(str)
    table["A_base"] = (table["base"] == "A").astype(float)
    table["A_r1"] = (table["r1"] == "A").astype(float)

    fam = pd.read_parquet(V0 / "data/folds/disease_fold_assignments.parquet",
                          columns=["disease_id", "disease_family_id"])
    fam["disease_id"] = fam["disease_id"].astype(str)
    table = table.merge(fam, on="disease_id", how="left")
    table["disease_family_id"] = table["disease_family_id"].fillna(table["disease_id"])

    clusters = [group for _, group in table.groupby("disease_family_id")]
    n = len(table)
    deltas, base_rates, r1_rates = [], [], []
    for _ in range(args.reps):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        sample = pd.concat([clusters[i] for i in idx], ignore_index=True)
        base_rates.append(float(sample["A_base"].mean()))
        r1_rates.append(float(sample["A_r1"].mean()))
        deltas.append(float(sample["A_r1"].mean() - sample["A_base"].mean()))

    deltas = np.array(deltas)
    observed = float(table["A_r1"].mean() - table["A_base"].mean())
    payload = {"diseases": int(n), "families": int(len(clusters)), "reps": args.reps,
               "observed_delta": round(observed, 4),
               "observed_base": round(float(table["A_base"].mean()), 4),
               "observed_r1": round(float(table["A_r1"].mean()), 4),
               "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
               "ci_high": round(float(np.percentile(deltas, 97.5)), 4),
               "p_two_sided": round(float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())), 4),
               "frac_resamples_positive": round(float((deltas > 0).mean()), 4)}
    out = DIR / "outputs/family_bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"delta": deltas}).to_parquet(out / "deltas.parquet", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
