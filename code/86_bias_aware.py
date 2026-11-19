"""R5 偏差感知边权：用度条件化重写 STRING 边权，再走同一条 RWR → 模块流程。

公式：w'(u,v) = w(u,v) / (d_u · d_v)^α，d 为加权度。
α=0 时退化为原始 STRING 权（作为对照，用来验证脚本与旧结果一致）。
判据（R5 卡片）：hub ≤ 基线（0.182）且 z ≥ 基线（2.416）同时成立。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}
BACKDROP = 20
SIZE = 25


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "bias_aware.log", "86_bias_aware")
    started = time.time()
    tag = f"a{args.alpha:g}"
    out_dir = paths.out("bias_aware", f"fold_{args.fold}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_node = {g: int(i) for g, i in zip(gene_ids, gene_rows["node_index"])}
    n_nodes = int(node_map["node_index"].max()) + 1

    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr()
                 for p in sorted(paths.v0(f"data/graphs/fold_{args.fold}").glob("A_*.npz"))}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    ppi["gene_a"] = ppi["gene_a"].astype(str)
    ppi["gene_b"] = ppi["gene_b"].astype(str)
    ppi = ppi[ppi["gene_a"].isin(gene_node) & ppi["gene_b"].isin(gene_node)]

    degree: dict[str, float] = {}
    for a, b, w in zip(ppi["gene_a"], ppi["gene_b"], ppi["weight"].astype(float)):
        degree[a] = degree.get(a, 0.0) + float(w)
        degree[b] = degree.get(b, 0.0) + float(w)
    med = float(np.median(list(degree.values()))) or 1.0

    # 直接在原 A_gene_gene 上按度条件化，保持原有的归一化方式
    base_gg = relations["gene_gene"].tocoo()
    node_to_gene = {int(i): g for g, i in gene_node.items()}
    node_factor = {i: (max(degree.get(node_to_gene.get(i, ""), 1e-6), 1e-6) / med) ** args.alpha
                   for i in range(n_nodes)}
    scale = np.array([1.0 / (node_factor.get(int(r), 1.0) * node_factor.get(int(c), 1.0))
                      for r, c in zip(base_gg.row, base_gg.col)])
    gene_gene = csr_matrix((base_gg.data * scale, (base_gg.row, base_gg.col)), shape=base_gg.shape)
    weights = ppi["weight"].to_numpy(dtype=float)
    log.info("network_ready", edges=int(gene_gene.nnz // 2), alpha=args.alpha)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if args.limit:
        test_ids = test_ids[: args.limit]

    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, ic, node_map)
    alpha_rwr = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                                 .read_text(encoding="utf-8"))["selected_alpha"])

    def z_of(disease: str, genes: set[str]) -> float:
        idx = [position[g] for g in genes if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), idx].mean())
        back = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            back.append((sim, other))
        back = [d for _, d in sorted(back, reverse=True)[:BACKDROP]]
        values = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in back])
        return float((own - values.mean()) / (values.std(ddof=1) + 1e-9))

    rel = dict(relations)
    rel["gene_gene"] = gene_gene
    transition = compose_transition_matrix(rel, BUDGET).tocsr()
    result = v0_rwr.run_rwr(transition, seeds, alpha=alpha_rwr)
    raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
    pct = np.zeros_like(raw)
    for i in range(raw.shape[1]):
        order = np.argsort(-raw[:, i])
        ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        pct[:, i] = ranks / len(order)

    gene_pos = {g: i for i, g in enumerate(gene_ids)}

    # 用新网络的分数重算 z（与原 pipeline 一致：z 基于自身分位）
    def z_new(disease: str, genes: set[str]) -> float:
        idx = [gene_pos[g] for g in genes if g in gene_pos]
        if len(idx) < 5:
            return np.nan
        col = test_ids.index(disease)
        own = float(pct[idx, col].mean())
        back = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease) or other not in test_ids:
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            back.append((sim, other))
        back = [d for _, d in sorted(back, reverse=True)[:BACKDROP] if d in test_ids]
        values = np.array([pct[idx, test_ids.index(b)].mean() for b in back])
        return float((own - values.mean()) / (values.std(ddof=1) + 1e-9))

    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"], ppi["gene_b"], weights):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    def hub_of(genes: set[str]) -> float:
        wd = {g: sum(w for n, w in adjacency.get(g, {}).items() if n in genes) for g in genes}
        total = sum(wd.values())
        return float(2 * max(wd.values(), default=0) / total) if total else 1.0

    rows_out, members = [], []
    for column, disease in enumerate(test_ids):
        ranking = np.argsort(-pct[:, column])[:200]
        candidate = tuple(gene_ids[i] for i in ranking)
        detections = detect(candidate, ppi, seed=stable_seed(86, args.fold, disease, "bias"))
        if not detections:
            continue
        genes = set(max(detections, key=len))
        genes = set(sorted(genes, key=lambda g: -pct[gene_pos[g], column])[:SIZE])
        if len(genes) < 5:
            continue
        rows_out.append({"disease_id": disease, "z": z_new(disease, genes), "hub": hub_of(genes),
                         "size": len(genes)})
        for gene in genes:
            members.append({"module_id": f"{disease}:bias", "disease_id": disease, "gene_id": gene})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows_out)
    table.to_parquet(out_dir / "bias_aware.parquet", index=False)
    pd.DataFrame(members).to_parquet(out_dir / "bias_aware_members.parquet", index=False)
    payload = {"fold": args.fold, "alpha": args.alpha, "diseases": int(len(table)),
               "z_median": round(float(table.z.median()), 3),
               "hub_median": round(float(table.hub.median()), 3),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "bias_aware.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
