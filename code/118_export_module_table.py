"""导出供审稿人使用的模块总表：每个疾病一行，含等级、四项判据与基因列表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for fold in range(5):
        graded = pd.read_parquet(DIR / f"outputs/refined_grading/fold_{fold}_r1/refined_graded.parquet")
        graded["disease_id"] = graded["disease_id"].astype(str)
        members = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{fold}.parquet")
        members["gene_id"] = members["gene_id"].astype(str)
        members["disease_id"] = members["disease_id"].astype(str)
        genes = members.groupby("disease_id")["gene_id"].apply(lambda s: ",".join(sorted(s))).to_dict()
        for row in graded.itertuples(index=False):
            rows.append({"disease_id": row.disease_id, "fold": fold,
                         "n_genes": len(genes.get(row.disease_id, "").split(",")) if genes.get(row.disease_id) else 0,
                         "grade": row.grade, "z_spec": round(float(row.z_spec), 4),
                         "p_spec": round(float(row.p_spec), 4), "replication": round(float(row.replication), 3),
                         "hub": round(float(row.hub), 3), "go_fdr": float(row.go_fdr),
                         "gene_ids": genes.get(row.disease_id, "")})

    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else DIR / "outputs/module_table"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "module_table.csv", index=False, encoding="utf-8-sig")
    table.to_parquet(out / "module_table.parquet", index=False)
    summary = {"diseases": int(len(table)), "grade_counts": table.grade.value_counts().to_dict(),
               "columns": table.columns.tolist()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
