"""Pareto 前沿选模块：每个疾病在全部候选（基线、共识、各 λ/μ/规模的精修）里，
按 (z_spec, 复现率) 取帕累托前沿，再在「复现率≥0.7」的点里取 z 最大者。

选点规则事先固定，与分级无关；候选池包含已跑过的所有配置。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto"
OUT.mkdir(parents=True, exist_ok=True)

REFINE_DIRS = {
    "lam0.5_mu1": "fold_0_lam0.5_mu1.0",
    "lam0.7_mu1": "fold_0_lam0.7_mu1.0",
    "lam1.0_mu1": "fold_0_lam1.0_mu1.0",
    "lam1.0_mu2": "fold_0_lam1.0_mu2.0",
    "size15": "fold_0_lam1.0_mu1.0_size15",
    "size20": "fold_0_lam1.0_mu1.0_size20",
    "size30": "fold_0_lam1.0_mu1.0_size30",
    "size40": "fold_0_lam1.0_mu1.0_size40",
}


def main() -> int:
    candidates: dict[str, list[dict]] = {}

    # 基线：z 来自 modules_doc，复现率取 lam1.0_mu1.0 里 baseline 行
    base_mods = pd.read_parquet(DIR / "outputs/modules_doc/fold_0/modules.parquet")
    base_mem = pd.read_parquet(DIR / "outputs/modules_doc/fold_0/members.parquet")
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    ref_rows = pd.read_parquet(DIR / "outputs/lam_refine/fold_0_lam1.0_mu1.0/lam_refine.parquet")
    base_rep = ref_rows[ref_rows.variant == "baseline"].set_index("disease_id")["replication"].to_dict()
    for disease, module_id in largest.items():
        genes = set(list(base_sets.get(module_id, set()))[:25])
        if len(genes) < 5:
            continue
        candidates.setdefault(disease, []).append({
            "source": "baseline", "genes": genes,
            "z": float(base_mods.loc[base_mods.module_id == module_id, "z_spec"].iloc[0]),
            "replication": float(base_rep.get(disease, 0.0)),
        })

    # 精修
    for tag, folder in REFINE_DIRS.items():
        path = DIR / "outputs/lam_refine" / folder
        if not (path / "lam_refine_members.parquet").exists():
            continue
        table = pd.read_parquet(path / "lam_refine.parquet")
        table = table[table.variant == "refined"]
        mem = pd.read_parquet(path / "lam_refine_members.parquet")
        mem["gene_id"] = mem["gene_id"].astype(str)
        sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()
        for row in table.itertuples(index=False):
            genes = sets.get(row.disease_id, set())
            if len(genes) < 5:
                continue
            candidates.setdefault(row.disease_id, []).append({
                "source": tag, "genes": genes, "z": float(row.z), "replication": float(row.replication),
            })

    rows = []
    chosen_members = []
    for disease, items in candidates.items():
        frame = pd.DataFrame([{"source": i["source"], "z": i["z"], "replication": i["replication"]}
                              for i in items])
        # 帕累托：不被 z 和 rep 同时支配
        front = []
        for idx, row in frame.iterrows():
            dominated = ((frame.z >= row.z) & (frame.replication >= row.replication) &
                         ((frame.z > row.z) | (frame.replication > row.replication))).any()
            if not dominated:
                front.append(idx)
        eligible = [i for i in front if frame.loc[i, "replication"] >= 0.7]
        pick = max(eligible, key=lambda i: frame.loc[i, "z"]) if eligible else \
            max(front, key=lambda i: frame.loc[i, "replication"])
        source = frame.loc[pick, "source"]
        rows.append({"disease_id": disease, "source": source,
                     "z": float(frame.loc[pick, "z"]), "replication": float(frame.loc[pick, "replication"]),
                     "front_size": len(front), "n_candidates": len(frame)})
        genes = next(i["genes"] for i in items if i["source"] == source)
        for gene in genes:
            chosen_members.append({"module_id": f"{disease}:pareto", "disease_id": disease, "gene_id": gene})

    selected = pd.DataFrame(rows)
    selected.to_parquet(OUT / "pareto_selected_fold0.parquet", index=False)
    pd.DataFrame(chosen_members).to_parquet(OUT / "pareto_members_fold0.parquet", index=False)
    summary = {"diseases": int(len(selected)),
               "source_counts": selected.source.value_counts().to_dict(),
               "z_median": round(float(selected.z.median()), 3),
               "rep_mean": round(float(selected.replication.mean()), 3),
               "frac_rep_ge_0.7": round(float((selected.replication >= 0.7).mean()), 3),
               "mean_front_size": round(float(selected.front_size.mean()), 1)}
    (OUT / "pareto_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
