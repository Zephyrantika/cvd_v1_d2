"""R4：图式共识（FastEnsemble 思路）替代层次聚类共识。

读入 87 号保存的原始划分，对每个疾病：
  1. 由划分统计共现矩阵（两基因同处一个划分模块的次数 / 运行数）
  2. 图式共识：以共现≥阈值连边，Leiden 切模块，取最大模块
  3. 与现有层次聚类共识（consensus_members）比较：核心集合 Jaccard、规模、频率分布
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = DIR / f"outputs/fastensemble/fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = pd.read_parquet(DIR / f"outputs/partitions/fold_{args.fold}/partitions.parquet")
    parts["gene_id"] = parts["gene_id"].astype(str)
    runs = parts["run"].nunique()

    existing = pd.read_parquet(DIR / f"outputs/consensus/fold_{args.fold}/consensus_members.parquet")
    existing["gene_id"] = existing["gene_id"].astype(str)
    existing_modules = pd.read_parquet(DIR / f"outputs/consensus/fold_{args.fold}/consensus_modules.parquet")
    largest = existing_modules.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    existing_core = {}
    for module_id, group in existing.groupby("module_id"):
        disease = str(group["disease_id"].iloc[0])
        if largest.get(disease) != module_id:
            continue
        existing_core[disease] = set(group.loc[group["frequency"] >= 0.8, "gene_id"].astype(str))

    rows = []
    for disease, group in parts.groupby("disease_id"):
        counts: dict[tuple[str, str], int] = {}
        genes: set[str] = set()
        for run, block in group.groupby("run"):
            for _, module in block.groupby("module_index"):
                members = module["gene_id"].tolist()
                genes.update(members)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = sorted((members[i], members[j]))
                        counts[(a, b)] = counts.get((a, b), 0) + 1
        if not genes:
            continue
        edges = [(a, b, c / runs) for (a, b), c in counts.items() if c / runs >= args.threshold]
        if not edges:
            continue
        graph = ig.Graph()
        index = {g: i for i, g in enumerate(sorted(genes))}
        graph.add_vertices(len(index))
        graph.add_edges([(index[a], index[b]) for a, b, _ in edges])
        graph.es["weight"] = [w for _, _, w in edges]
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                        weights="weight", resolution_parameter=args.resolution,
                                        seed=42)
        biggest = max(part, key=len)
        members = [sorted(index)[i] for i in biggest]
        # 频率：成员在该病划分里出现的运行比例（以最大模块内共现为近似）
        freq = {g: counts.get((g, g), 0) for g in members}
        core = set(members)
        old = existing_core.get(disease, set())
        rows.append({"disease_id": disease, "graph_size": len(members),
                     "graph_core": len(core), "existing_core": len(old),
                     "jaccard": len(core & old) / len(core | old) if (core | old) else np.nan})

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "compare.parquet", index=False)
    summary = {"fold": args.fold, "threshold": args.threshold, "runs": int(runs),
               "diseases": int(len(table)),
               "jaccard_median": round(float(table.jaccard.median()), 3) if len(table) else None,
               "graph_size_median": round(float(table.graph_size.median()), 1) if len(table) else None,
               "existing_core_median": round(float(table.existing_core.median()), 1) if len(table) else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
