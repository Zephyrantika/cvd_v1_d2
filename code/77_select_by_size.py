"""按疾病选模块规模：在 15/20/25/30/40 五个规模中，要求复现率≥0.7，取 z_spec 最高的那个。

规模本身是方法的自由度；规则在看分级结果之前固定，对所有疾病一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/selection"
OUT.mkdir(parents=True, exist_ok=True)

FOLDERS = {
    15: "fold_0_lam1.0_mu1.0_size15",
    20: "fold_0_lam1.0_mu1.0_size20",
    25: "fold_0_lam1.0_mu1.0",
    30: "fold_0_lam1.0_mu1.0_size30",
    40: "fold_0_lam1.0_mu1.0_size40",
}


def main() -> int:
    rows = []
    members: dict[tuple[str, int], set[str]] = {}
    for size, folder in FOLDERS.items():
        path = DIR / "outputs/lam_refine" / folder
        if not path.exists():
            print(f"[warn] missing {folder}")
            continue
        table = pd.read_parquet(path / "lam_refine.parquet")
        table = table[table.variant == "refined"]
        mem = pd.read_parquet(path / "lam_refine_members.parquet")
        mem["gene_id"] = mem["gene_id"].astype(str)
        sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()
        for row in table.itertuples(index=False):
            rows.append({"disease_id": row.disease_id, "size": size, "z": float(row.z),
                         "replication": float(row.replication)})
            members[(row.disease_id, size)] = sets.get(row.disease_id, set())

    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "size_candidates_fold0.parquet", index=False)

    selected = []
    for disease, group in frame.groupby("disease_id"):
        eligible = group[group.replication >= 0.7]
        pool = eligible if len(eligible) else group
        best = pool.sort_values("z", ascending=False).iloc[0]
        selected.append({"disease_id": disease, "size": int(best["size"]), "z": float(best.z),
                         "replication": float(best.replication)})
    chosen = pd.DataFrame(selected)
    chosen.to_parquet(OUT / "selected_by_size_fold0.parquet", index=False)

    member_rows = []
    for row in chosen.itertuples(index=False):
        for gene in members.get((row.disease_id, row.size), set()):
            member_rows.append({"module_id": f"{row.disease_id}:size{row.size}",
                                "disease_id": row.disease_id, "gene_id": gene})
    pd.DataFrame(member_rows).to_parquet(OUT / "selected_by_size_members_fold0.parquet", index=False)

    summary = {"diseases": int(len(chosen)),
               "size_counts": chosen["size"].value_counts().sort_index().to_dict(),
               "z_median": round(float(chosen.z.median()), 3),
               "rep_mean": round(float(chosen.replication.mean()), 3),
               "frac_rep_ge_0.7": round(float((chosen.replication >= 0.7).mean()), 3)}
    (OUT / "size_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
