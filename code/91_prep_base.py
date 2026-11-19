"""为同口径对比准备基线的模块成员与复现率（每折一份）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    fold = args.fold
    out = DIR / "outputs/pareto"
    out.mkdir(parents=True, exist_ok=True)

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    msets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    rows = []
    for disease, module_id in largest.items():
        for gene in list(msets.get(module_id, set()))[:25]:
            rows.append({"module_id": f"{disease}:base", "disease_id": disease, "gene_id": gene})
    pd.DataFrame(rows).to_parquet(out / f"base_members_fold{fold}.parquet", index=False)

    lam = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{fold}_lam1.0_mu1.0/lam_refine.parquet")
    base = lam[lam.variant == "baseline"][["disease_id", "replication"]]
    base.to_parquet(out / f"base_rep_fold{fold}.parquet", index=False)
    print(f"fold {fold}: {len(base)} diseases, {len(rows)} member rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
