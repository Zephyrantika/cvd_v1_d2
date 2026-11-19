"""五折版 Pareto 选模块：每折在固定候选池里按 (z_spec, 复现率) 取前沿，再选「复现率≥0.7 中 z 最大」。

候选池（每折一致）：基线最大模块、λ=0.5/μ=1、λ=1/μ=1，以及 λ=1/μ=1 下规模 15/20/30/40 的精修。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto"


def candidates_for(fold: int) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}
    base_mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/modules.parquet")
    base_mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/members.parquet")
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    ref_rows = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{fold}_lam1.0_mu1.0/lam_refine.parquet")
    base_rep = ref_rows[ref_rows.variant == "baseline"].set_index("disease_id")["replication"].to_dict()
    z_lookup = dict(zip(base_mods["module_id"], base_mods["z_spec"].astype(float)))
    for disease, module_id in largest.items():
        genes = set(list(base_sets.get(module_id, set()))[:25])
        if len(genes) < 5:
            continue
        pools.setdefault(disease, []).append({
            "source": "baseline", "genes": genes, "z": z_lookup.get(module_id, 0.0),
            "replication": float(base_rep.get(disease, 0.0)),
        })

    folders = {
        "lam0.5": f"fold_{fold}_lam0.5_mu1.0",
        "lam1.0": f"fold_{fold}_lam1.0_mu1.0",
        "size15": f"fold_{fold}_lam1.0_mu1.0_size15",
        "size20": f"fold_{fold}_lam1.0_mu1.0_size20",
        "size30": f"fold_{fold}_lam1.0_mu1.0_size30",
        "size40": f"fold_{fold}_lam1.0_mu1.0_size40",
    }
    for tag, folder in folders.items():
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
            pools.setdefault(row.disease_id, []).append({
                "source": tag, "genes": genes, "z": float(row.z), "replication": float(row.replication),
            })
    return pools


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    fold = args.fold
    OUT.mkdir(parents=True, exist_ok=True)

    pools = candidates_for(fold)
    rows, members = [], []
    for disease, items in pools.items():
        frame = pd.DataFrame([{"source": i["source"], "z": i["z"], "replication": i["replication"]}
                              for i in items])
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
        rows.append({"disease_id": disease, "source": source, "z": float(frame.loc[pick, "z"]),
                     "replication": float(frame.loc[pick, "replication"]),
                     "front_size": len(front), "n_candidates": len(frame)})
        genes = next(i["genes"] for i in items if i["source"] == source)
        for gene in genes:
            members.append({"module_id": f"{disease}:pareto", "disease_id": disease, "gene_id": gene})

    selected = pd.DataFrame(rows)
    selected.to_parquet(OUT / f"pareto_selected_fold{fold}.parquet", index=False)
    pd.DataFrame(members).to_parquet(OUT / f"pareto_members_fold{fold}.parquet", index=False)
    summary = {"fold": fold, "diseases": int(len(selected)),
               "source_counts": selected.source.value_counts().to_dict(),
               "z_median": round(float(selected.z.median()), 3),
               "rep_mean": round(float(selected.replication.mean()), 3),
               "frac_rep_ge_0.7": round(float((selected.replication >= 0.7).mean()), 3)}
    (OUT / f"pareto_summary_fold{fold}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
