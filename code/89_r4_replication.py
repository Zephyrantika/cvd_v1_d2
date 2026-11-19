"""R4 评估：图式共识模块的独立扰动复现率（fold 3，与基线最大模块对比）。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(2026)
    log = JsonlLogger(paths.out("qc") / "r4_replication.log", "89_r4_replication")
    started = time.time()
    out_dir = paths.out("fastensemble", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    # 图式共识模块：用 88 号输出的比较表里的 graph 模块（重新计算并落盘成员）
    import igraph as ig
    import leidenalg
    parts = pd.read_parquet(paths.out("partitions", f"fold_{args.fold}", "partitions.parquet"))
    parts["gene_id"] = parts["gene_id"].astype(str)
    runs = parts["run"].nunique()
    graph_sets: dict[str, set[str]] = {}
    for disease, group in parts.groupby("disease_id"):
        counts: dict[tuple[str, str], int] = {}
        genes: set[str] = set()
        for _, block in group.groupby("run"):
            for _, module in block.groupby("module_index"):
                members = module["gene_id"].tolist()
                genes.update(members)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = sorted((members[i], members[j]))
                        counts[(a, b)] = counts.get((a, b), 0) + 1
        edges = [(a, b) for (a, b), c in counts.items() if c / runs >= 0.5]
        if not edges:
            continue
        index = {g: i for i, g in enumerate(sorted(genes))}
        graph = ig.Graph()
        graph.add_vertices(len(index))
        graph.add_edges([(index[a], index[b]) for a, b in edges])
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, seed=42)
        biggest = max(part, key=len)
        graph_sets[disease] = {sorted(index)[i] for i in biggest}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_index = {str(t): int(i) for t, i in zip(node_map.loc[node_map["node_type"] == "phenotype", "node_id"],
                                                node_map.loc[node_map["node_type"] == "phenotype", "node_index"])}
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    rows = []
    for disease, module in graph_sets.items():
        if disease not in largest:
            continue
        base = set(list(base_sets.get(largest[disease], set()))[:25])
        if len(base) < 5 or len(module) < 5:
            continue
        terms = terms_of.get(disease, [])
        partitions = []
        for run in range(args.runs):
            kept = [t for t in terms if rng.random() > 0.10]
            indices = [hpo_index[t] for t in kept if t in hpo_index]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[:200]
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(2026, args.fold, disease, run))
            partitions.append([set(m) for m in detections])
        for name, genes in (("baseline", base), ("graph", module)):
            covered = sum(1 for p in partitions if any(len(genes & m) / len(genes) >= COVER for m in p))
            rows.append({"disease_id": disease, "variant": name, "size": len(genes),
                         "replication": covered / max(len(partitions), 1)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "replication.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant", values="replication").dropna()
    payload = {"fold": args.fold, "runs": args.runs, "diseases": int(len(pivot)),
               "baseline": round(float(pivot.baseline.mean()), 3),
               "graph": round(float(pivot.graph.mean()), 3),
               "wins": int((pivot.graph > pivot.baseline).sum()),
               "losses": int((pivot.graph < pivot.baseline).sum()),
               "p": float(wilcoxon(pivot.graph, pivot.baseline).pvalue),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "replication.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
