"""文献里挑出的三种替代模块抽取，与 Leiden 基线配对比较。

依据交接包文献：
- Fortunato & Barthelemy 2007（分辨率极限）：模块度优化有分辨率偏差，小模块会被合并；
- Menche 2015：疾病模块常取已知基因在互作组上的连通子图；
- Traag 2019：Leiden 保证连通，但仍优化模块度类目标。

三种替代：
  kcore   取 top-200 诱导子图的最大 k-core，再按 r_d 取前 25
  lcc     取 top-200 诱导子图的最大连通分量，再按 r_d 取前 25
  res     多分辨率 Leiden（0.5/1.0/1.5/2.0），按模块内密度选一个，再取前 25

评价：同疾病、同大小 25 的 z_spec（IC 相似度背景 20），与基线配对。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "alternatives.log", "46_alternatives")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))


    def z_of(disease: str, genes: list[str]) -> float:
        genes = [g for g in genes if g in position]
        if not genes:
            return np.nan
        background = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            background.append((sim, other))
        background = [b for _, b in sorted(background, reverse=True)[:20]]
        own = float(percentiles[matrix.disease_index(disease), [position[g] for g in genes]].mean())
        back = np.array([percentiles[matrix.disease_index(b), [position[g] for g in genes]].mean()
                         for b in background])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    def edges_of(candidate: list[str]):
        index = {g: i for i, g in enumerate(candidate)}
        pairs = []
        for a, b in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str)):
            ia, ib = index.get(a), index.get(b)
            if ia is not None and ib is not None:
                pairs.append((ia, ib))
        return pairs, index

    modules = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    largest = modules.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    baseline_rows = []
    for disease, module_id in largest.items():
        genes = list(members.loc[members.module_id == module_id].sort_values("r_d", ascending=False)["gene_id"])[:SIZE]
        if len(genes) >= 5:
            baseline_rows.append({"disease_id": disease, "z_baseline": z_of(disease, genes)})

    rows = []
    for disease in matrix.disease_ids:
        candidate = list(matrix.top_k(disease, 200))
        pairs, index = edges_of(candidate)
        graph = ig.Graph(n=len(candidate), edges=pairs, directed=False)
        r_d = percentiles[matrix.disease_index(disease)][[position[g] for g in candidate]]

        def top_of(nodes: list[int]) -> list[str]:
            ordered = sorted(nodes, key=lambda i: -r_d[i])[:SIZE]
            return [candidate[i] for i in ordered]

        # k-core：取最大的 k 使 core 至少 5 个节点
        coreness = graph.coreness()
        kcore_nodes = list(range(len(candidate)))
        for level in range(max(coreness), 0, -1):
            nodes = [i for i, value in enumerate(coreness) if value >= level]
            if len(nodes) >= 5:
                kcore_nodes = nodes
                break
        kcore_genes = top_of(kcore_nodes)

        # 最大连通分量
        membership = graph.connected_components().membership
        counts = pd.Series(membership).value_counts()
        lcc_nodes = [i for i, label in enumerate(membership) if label == counts.index[0]]
        lcc_genes = top_of(lcc_nodes)

        # 多分辨率 Leiden，按模块内密度选
        best, best_density = None, -1.0
        for resolution in (0.5, 1.0, 1.5, 2.0):
            partition = graph.community_leiden(objective_function="modularity", resolution=resolution,
                                               weights=None, n_iterations=2)
            for cluster in partition:
                if len(cluster) < 5:
                    continue
                sub = graph.subgraph(cluster)
                possible = len(cluster) * (len(cluster) - 1) / 2
                density = sub.ecount() / possible if possible else 0.0
                if density > best_density:
                    best_density, best = density, cluster
        res_genes = top_of(best if best is not None else list(range(len(candidate))))

        rows.append({"disease_id": disease,
                     "z_kcore": z_of(disease, kcore_genes),
                     "z_lcc": z_of(disease, lcc_genes),
                     "z_res": z_of(disease, res_genes)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows).merge(pd.DataFrame(baseline_rows), on="disease_id", how="inner")
    table.to_parquet(out_dir / "alternatives.parquet", index=False)

    summary = {"fold": args.fold, "diseases": int(len(table))}
    for column in ("z_kcore", "z_lcc", "z_res"):
        paired = table[[column, "z_baseline"]].dropna()
        summary[column] = {
            "median": round(float(paired[column].median()), 3),
            "wins": int((paired[column] > paired["z_baseline"]).sum()),
            "n": int(len(paired)),
            "p": float(wilcoxon(paired[column], paired["z_baseline"]).pvalue) if len(paired) > 5 else None,
        }
    summary["baseline_median"] = round(float(table["z_baseline"].median()), 3)
    summary["seconds"] = round(time.time() - started, 2)
    (out_dir / "alternatives.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())