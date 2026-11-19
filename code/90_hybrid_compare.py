"""R6 混合打分对照（Sci Rep 2026 思路的简化实现）。

该文把「网络拓扑特征」与「扩散（RWR）分数」融合后再做优先化。这里按同样思路实现：
    hybrid = w_rwr · 分位(RWR) + w_deg · 分位(度) + w_cc · 分位(聚类系数)
权重取论文描述的中性方案（等权）与两档偏向扩散的方案，均事先固定，不在测试折上调。
模块流程与原 pipeline 一致：top-200 → Leiden → 最大模块 → 按混合分取前 25。
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
from scipy.stats import hypergeom, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20
SIZE = 25
WEIGHTS = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "diffusion_heavy": (0.6, 0.2, 0.2),
    "topology_heavy": (0.2, 0.4, 0.4),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "hybrid.log", "90_hybrid_compare")
    started = time.time()
    out_dir = paths.out("hybrid", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    genes = list(matrix.gene_ids)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    ppi["gene_a"] = ppi["gene_a"].astype(str)
    ppi["gene_b"] = ppi["gene_b"].astype(str)
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"], ppi["gene_b"], ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    degree = {g: sum(adjacency.get(g, {}).values()) for g in genes}
    deg_rank = pd.Series(degree).rank(pct=True).to_dict()

    graph = ig.Graph()
    index = {g: i for i, g in enumerate(genes)}
    graph.add_vertices(len(genes))
    edges = [(index[a], index[b]) for a in adjacency for b in adjacency[a] if a < b]
    graph.add_edges(edges)
    cc = dict(zip(genes, graph.transitivity_local_undirected(mode="zero")))

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))

    def background(disease: str) -> list[str]:
        scored = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_of(disease: str, values: np.ndarray, genes_set: set[str]) -> float:
        idx = [position[g] for g in genes_set if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(values[idx].mean())
        back = np.array([values[idx].mean() for _ in [0]])  # placeholder
        return np.nan

    mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    def score_z(disease: str, genes_set: set[str]) -> float:
        idx = [position[g] for g in genes_set if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), idx].mean())
        back = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in background(disease)])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    rows = []
    for name, (w_rwr, w_deg, w_cc) in WEIGHTS.items():
        for disease in matrix.disease_ids:
            row = percentiles[matrix.disease_index(disease)]
            hybrid = np.array([w_rwr * row[position[g]] + w_deg * deg_rank.get(g, 0.0) + w_cc * cc.get(g, 0.0)
                               for g in genes])
            order = np.argsort(-hybrid)[:200]
            candidate = tuple(genes[i] for i in order)
            detections = detect(candidate, ppi, seed=stable_seed(90, args.fold, disease, name))
            if not detections:
                continue
            module = set(max(detections, key=len))
            module = set(sorted(module, key=lambda g: -hybrid[position[g]])[:SIZE])
            if len(module) < 5:
                continue
            rows.append({"variant": name, "disease_id": disease, "z_hybrid": score_z(disease, module),
                         "z_baseline": score_z(disease, set(list(base_sets.get(largest.get(disease), set()))[:SIZE])),
                         "size": len(module)})
        log.info("variant_done", variant=name)

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "hybrid.parquet", index=False)
    payload = {"fold": args.fold}
    for name in WEIGHTS:
        sub = table[table.variant == name]
        if not len(sub):
            continue
        payload[name] = {"diseases": int(len(sub)),
                         "z_hybrid": round(float(sub.z_hybrid.median()), 3),
                         "z_baseline": round(float(sub.z_baseline.median()), 3),
                         "wins": int((sub.z_hybrid > sub.z_baseline).sum()),
                         "p": float(wilcoxon(sub.z_hybrid, sub.z_baseline).pvalue)}
    payload["seconds"] = round(time.time() - started, 1)
    (out_dir / "hybrid.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
