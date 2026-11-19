"""R1：完整 Pareto 前沿（EMODMI 思路的落地版）。

候选池：fold_N 下所有带成员表的 lam_refine 配置（λ × μ × 规模），加上基线最大模块。
每个候选计算 (z_spec, 复现率, hub, 密度)，逐病求非支配集（前沿），
在「复现率≥0.7 且 hub≤0.5」的点里取 z 最大者作为输出，并报告前沿宽度与面积。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]
OUT = DIR / "outputs/pareto_r1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-rep", type=float, default=0.7)
    parser.add_argument("--max-hub", type=float, default=0.5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ppi = pd.read_parquet(DIR.parents[0] / "phenotype-network-v0/data/processed/string_gene_edges_700.parquet",
                          columns=["gene_a", "gene_b", "weight"])
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    def hub_of(genes: set[str]) -> float:
        wd = {g: sum(w for n, w in adjacency.get(g, {}).items() if n in genes) for g in genes}
        total = sum(wd.values())
        return float(2 * max(wd.values(), default=0) / total) if total else 1.0

    def density_of(genes: set[str]) -> float:
        n = len(genes)
        if n < 2:
            return 0.0
        edges = sum(1 for g in genes for x in adjacency.get(g, {}) if x in genes)
        return edges / (n * (n - 1))

    pools: dict[str, list[dict]] = {}
    base_mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    base_mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    ref_rows = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0/lam_refine.parquet")
    base_rep = ref_rows[ref_rows.variant == "baseline"].set_index("disease_id")["replication"].to_dict()
    for disease, module_id in largest.items():
        genes = set(list(base_sets.get(module_id, set()))[:25])
        if len(genes) < 5:
            continue
        pools.setdefault(disease, []).append({"source": "baseline", "genes": genes,
                                              "replication": float(base_rep.get(disease, 0.0)),
                                              "z": float(base_mods.loc[base_mods.module_id == module_id, "z_spec"].iloc[0])})

    for path in sorted((DIR / "outputs/lam_refine").glob(f"fold_{args.fold}_lam*")):
        mem_path = path / "lam_refine_members.parquet"
        table_path = path / "lam_refine.parquet"
        if not (mem_path.exists() and table_path.exists()):
            continue
        tag = re.sub(rf"^fold_{args.fold}_", "", path.name)
        table = pd.read_parquet(table_path)
        table = table[table.variant == "refined"]
        mem = pd.read_parquet(mem_path)
        mem["gene_id"] = mem["gene_id"].astype(str)
        sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()
        for row in table.itertuples(index=False):
            genes = sets.get(row.disease_id, set())
            if len(genes) < 5:
                continue
            pools.setdefault(row.disease_id, []).append({"source": tag, "genes": genes,
                                                         "replication": float(row.replication),
                                                         "z": float(row.z)})

    rows, members = [], []
    for disease, items in pools.items():
        frame = pd.DataFrame([{"source": i["source"], "z": i["z"], "replication": i["replication"],
                               "hub": hub_of(i["genes"]), "density": density_of(i["genes"])} for i in items])
        front = []
        for idx, row in frame.iterrows():
            dominated = ((frame.z >= row.z) & (frame.replication >= row.replication) &
                         ((frame.z > row.z) | (frame.replication > row.replication))).any()
            if not dominated:
                front.append(idx)
        eligible = [i for i in front
                    if frame.loc[i, "replication"] >= args.min_rep and frame.loc[i, "hub"] <= args.max_hub]
        pool = eligible if eligible else front
        pick = max(pool, key=lambda i: frame.loc[i, "z"])
        src = frame.loc[pick, "source"]
        rows.append({"disease_id": disease, "source": src, "z": float(frame.loc[pick, "z"]),
                     "replication": float(frame.loc[pick, "replication"]),
                     "hub": float(frame.loc[pick, "hub"]), "density": float(frame.loc[pick, "density"]),
                     "front_size": len(front), "n_candidates": len(frame)})
        genes = next(i["genes"] for i in items if i["source"] == src)
        for gene in genes:
            members.append({"module_id": f"{disease}:r1", "disease_id": disease, "gene_id": gene})

    selected = pd.DataFrame(rows)
    selected.to_parquet(OUT / f"selected_fold{args.fold}.parquet", index=False)
    pd.DataFrame(members).to_parquet(OUT / f"members_fold{args.fold}.parquet", index=False)
    summary = {"fold": args.fold, "diseases": int(len(selected)),
               "source_counts": selected.source.value_counts().head(12).to_dict(),
               "front_size_mean": round(float(selected.front_size.mean()), 2),
               "candidates_mean": round(float(selected.n_candidates.mean()), 1),
               "z_median": round(float(selected.z.median()), 3),
               "rep_mean": round(float(selected.replication.mean()), 3),
               "hub_median": round(float(selected.hub.median()), 3),
               "frac_rep_ge_0.7": round(float((selected.replication >= 0.7).mean()), 3)}
    (OUT / f"summary_fold{args.fold}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
