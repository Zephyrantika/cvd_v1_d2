"""按技术文档口径重算 z_spec 与 ZM2。

§3.2：D_bg(d) = 与 d 表型最相似、不同家族的 20 个疾病，相似度用 HPO IC 语义相似度。
       这里用 IC 加权余弦（共享术语的 IC 之和 / 两侧 IC 模长乘积的平方根）作为语义相似度，
       Resnik 版本留作敏感性。
§4.4 ZM2：对 C_d 诱导子图做 double edge swap（10× 边数），重算 density/conductance，
       经验 p 值 p_struct。

输出按文档 §7 的 schema 组织：modules 表含 z_spec、p_struct、density、conductance、hub、tier。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP_SIZE = 20


def ic_weighted_similarity(sets: dict[str, set[str]], ic: dict[str, float]) -> dict[tuple[str, str], float]:
    """疾病两两的 IC 加权余弦相似度。"""
    norms = {d: float(np.sqrt(sum(ic.get(t, 0.0) for t in terms))) for d, terms in sets.items()}
    diseases = sorted(sets)
    similarities: dict[tuple[str, str], float] = {}
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            shared = sets[a] & sets[b]
            if not shared or norms[a] == 0 or norms[b] == 0:
                # 没有共享术语时用 0，避免除零
                similarities[(a, b)] = 0.0
                continue
            value = sum(ic.get(t, 0.0) for t in shared) / (norms[a] * norms[b])
            similarities[(a, b)] = float(value)
    return similarities


def backdrop(disease: str, fold_diseases: tuple[str, ...], family: dict[str, str],
             sim: dict[tuple[str, str], float]) -> list[str]:
    candidates = []
    for other in fold_diseases:
        if other == disease or family.get(other) == family.get(disease):
            continue
        key = (disease, other) if (disease, other) in sim else (other, disease)
        candidates.append((sim.get(key, 0.0), other))
    candidates.sort(reverse=True)
    return [other for _, other in candidates[:BACKDROP_SIZE]]


def module_topology(members: list[str], adjacency, complement: set[str]) -> tuple[float, float, float]:
    member_set = set(members)
    internal = 0
    weighted_degree = {g: 0.0 for g in member_set}
    for gene in member_set:
        for neighbour, weight in adjacency.get(gene, {}).items():
            if neighbour in member_set:
                internal += 1
                weighted_degree[gene] += weight
    possible = len(member_set) * (len(member_set) - 1) / 2
    density = (internal / 2) / possible if possible else 0.0
    total = sum(weighted_degree.values())
    hub = 2.0 * max(weighted_degree.values(), default=0.0) / total if total else 1.0

    volume_m = sum(weighted_degree.values())
    cut = sum(
        weight
        for gene in member_set
        for neighbour, weight in adjacency.get(gene, {}).items()
        if neighbour in complement
    )
    volume_out = sum(
        sum(w for n, w in adjacency.get(g, {}).items() if n in complement)
        for g in complement
    )
    denominator = min(volume_m, volume_out) if min(volume_m, volume_out) > 0 else 1.0
    conductance = cut / denominator
    return float(density), float(hub), float(conductance)


def rewired_graph(genes: list[str], edges: list[tuple[int, int, float]], swaps: int) -> ig.Graph:
    graph = ig.Graph(n=len(genes), edges=[(a, b) for a, b, _ in edges], directed=False)
    graph.es["weight"] = [w for _, _, w in edges]
    if graph.ecount():
        graph.rewire(n=swaps, mode="simple")
    # 重排后权重与边不再对应，按原权重分布随机赋回去
    if graph.ecount():
        weights = np.random.default_rng(0).permutation(np.array([w for _, _, w in edges]))
        graph.es["weight"] = weights[: graph.ecount()].tolist()
    return graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "doc_compliance.log", "32_doc_compliance")
    started = time.time()
    out_dir = paths.out("compliance")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"))
    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    tiers = pd.read_parquet(paths.out("audit", "disease_tiers.parquet"))[["disease_id", "tier"]]
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "fold", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    sets = {d: set(s.hpo_id) for d, s in phenotypes.groupby("disease_id")}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    module_members = members.groupby("module_id")["gene_id"].apply(list).to_dict()
    module_rows = modules.set_index("module_id")

    gene_position = {g: i for i, g in enumerate(next(iter([load_saved_scores(paths.out("scores", "fold_0", "score_matrix.npz")).gene_ids])))}
    core_sizes = (
        members.assign(is_core=members["consensus_frequency"] >= 0.8)
        .groupby("module_id")["is_core"].sum().to_dict()
    )
    tier_lookup = tiers.set_index("disease_id")["tier"].to_dict()

    results = []
    for fold in [int(x) for x in args.folds.split(",") if x.strip()]:
        matrix = load_saved_scores(paths.out("scores", f"fold_{fold}", "score_matrix.npz"))
        percentiles = matrix.percentiles()
        position = {g: i for i, g in enumerate(matrix.gene_ids)}
        ic = pd.read_parquet(paths.v0(f"data/folds/fold_{fold}/train_hpo_ic.parquet"))
        ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
        similarities = ic_weighted_similarity(sets, ic_lookup)
        fold_diseases = tuple(sorted(matrix.disease_ids))

        disease_modules = (
            module_rows[module_rows["fold"] == fold]
            .assign(disease_id=lambda d: d["disease_id"].astype(str))
            .groupby("disease_id")
        )
        for disease, group in disease_modules:
            candidate = list(matrix.top_k(disease, 200))
            candidate_set = set(candidate)
            index = {g: i for i, g in enumerate(candidate)}
            edges = [
                (index[str(a)], index[str(b)], float(w))
                for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str),
                                   ppi["weight"].astype(float), strict=True)
                if str(a) in index and str(b) in index
            ]
            bg = backdrop(disease, fold_diseases, family, similarities)
            bg_rows = [(d, matrix.disease_index(d)) for d in bg if d in matrix.disease_ids]
            if len(bg_rows) < 5:
                continue

            module_sets = []
            for module_id, row in group.iterrows():
                genes = [g for g in module_members.get(module_id, []) if g in candidate_set]
                if len(genes) < 5:
                    continue
                indices = [index[g] for g in genes]
                density, hub, conductance = module_topology(genes, adjacency, candidate_set - set(genes))
                module_sets.append({
                    "module_id": module_id, "genes": genes, "indices": indices,
                    "density": density, "hub": hub, "conductance": conductance,
                    "stability": float(row["stability_mean_jaccard"]),
                })
            if not module_sets:
                continue

            null_counts = {m["module_id"]: [] for m in module_sets}
            base_edges = [(a, b) for a, b, _ in edges]
            for _ in range(args.draws):
                shuffled = base_edges[:]
                rng.shuffle(shuffled)
                graph = ig.Graph(n=len(candidate), edges=shuffled, directed=False)
                if graph.ecount():
                    graph.rewire(n=10 * graph.ecount(), mode="simple")
                for m in module_sets:
                    sub = graph.subgraph(m["indices"])
                    possible = len(m["indices"]) * (len(m["indices"]) - 1) / 2
                    null_counts[m["module_id"]].append(sub.ecount() / possible if possible else 0.0)

            for m in module_sets:
                null = np.array(null_counts[m["module_id"]])
                p_struct = (1 + int((null >= m["density"]).sum())) / (len(null) + 1)
                gene_rows = [position[g] for g in m["genes"]]
                self_score = float(percentiles[matrix.disease_index(disease), gene_rows].mean())
                background_scores = np.array([
                    percentiles[row_index, gene_rows].mean() for _, row_index in bg_rows
                ])
                z_spec = (self_score - float(background_scores.mean())) / (float(background_scores.std(ddof=1)) + 1e-9)
                results.append({
                    "module_id": m["module_id"],
                    "disease_id": disease,
                    "fold": int(fold),
                    "method": "leiden",
                    "size": len(m["genes"]),
                    "z_spec": float(z_spec),
                    "p_struct": float(p_struct),
                    "density": m["density"],
                    "conductance": m["conductance"],
                    "hub": m["hub"],
                    "stability": m["stability"],
                    "core_size": int(core_sizes.get(m["module_id"], 0)),
                    "core_jaccard": float("nan"),
                    "tier": tier_lookup.get(disease),
                    "reliability_tier": "pending",
                    "n_backdrop": len(bg_rows),
                })
        log.info("fold_done", fold=fold, modules=len(results))
    table = pd.DataFrame(results)
    table.to_parquet(out_dir / "modules_doc_schema.parquet", index=False)
    summary = {
        "modules": int(len(table)),
        "z_spec_median": round(float(table["z_spec"].median()), 3),
        "z_spec_gt_2": round(float((table["z_spec"] > 2).mean()), 3),
        "p_struct_le_0.05": round(float((table["p_struct"] <= 0.05).mean()), 3),
        "density_median": round(float(table["density"].median()), 3),
        "conductance_median": round(float(table["conductance"].median()), 3),
        "draws": args.draws,
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "modules_doc_schema.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    by_tier = table.groupby("tier").agg(n=("module_id", "size"),
                                         z_median=("z_spec", "median"),
                                         p_struct_le_005=("p_struct", lambda s: float((s <= 0.05).mean())),
                                         density=("density", "median")).round(3)
    by_tier.to_csv(out_dir / "modules_doc_schema_by_tier.csv", encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(by_tier.to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
