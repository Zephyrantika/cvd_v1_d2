"""外部验证二：在 HumanBase 心脏网络上重算复现率，检验优势是否只在 STRING 上成立。

做法：把基因—基因关系换成心脏网络，其余流程与 STRING 版一致；对同一批模块
（基线最大模块、R1 选点模块）各跑 20 次扰动，比较复现率。
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

COVER = 0.6
BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_interacts": 0.0,
             "gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "heart_replication.log", "96_heart_replication")
    started = time.time()
    out_dir = paths.out("heart_validation", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
    detect, _ = load_v0_models(paths.v0_root)

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_index = {g: int(i) for g, i in zip(gene_ids, gene_rows["node_index"])}
    hpo_index = {str(t): int(i) for t, i in zip(node_map.loc[node_map["node_type"] == "phenotype", "node_id"],
                                                node_map.loc[node_map["node_type"] == "phenotype", "node_index"])}
    n_nodes = int(node_map["node_index"].max()) + 1

    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr()
                 for p in sorted(paths.v0(f"data/graphs/fold_{args.fold}").glob("A_*.npz"))}
    heart_path = Path(__file__).resolve().parents[1] / "data/external/networks/humanbase/heart_top.gz"
    heart = pd.read_csv(heart_path, sep="\t", header=None, names=["a", "b", "p"], dtype={"a": str, "b": str})
    heart = heart[heart["a"].isin(gene_index) & heart["b"].isin(gene_index)]
    rows = heart["a"].map(gene_index).to_numpy()
    cols = heart["b"].map(gene_index).to_numpy()
    vals = heart["p"].to_numpy(dtype=float)
    heart_matrix = csr_matrix((np.concatenate([vals, vals]), (np.concatenate([rows, cols]),
                                                            np.concatenate([cols, rows]))), shape=(n_nodes, n_nodes))
    heart_matrix.sum_duplicates()
    rel = dict(relations)
    rel["gene_gene"] = heart_matrix
    budget = {"phenotype": BUDGET["phenotype"],
              "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
                       "gene_participates_in_pathway": 0.25},
              "pathway": BUDGET["pathway"]}
    transition = compose_transition_matrix(rel, budget).tocsr()
    log.info("heart_transition_ready")

    base_mem = pd.read_parquet(paths.out("pareto", f"base_members_fold{args.fold}.parquet"))
    r1_mem = pd.read_parquet(paths.out("pareto_r1", f"members_fold{args.fold}.parquet"))
    sets = {}
    for name, frame in (("baseline", base_mem), ("r1", r1_mem)):
        frame["gene_id"] = frame["gene_id"].astype(str)
        sets[name] = frame.groupby("disease_id")["gene_id"].apply(set).to_dict()

    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    rows_out = []
    for disease in sorted(set(sets["baseline"]) & set(sets["r1"])):
        terms = terms_of.get(disease, [])
        if len(terms) < 5:
            continue
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
                                seed=stable_seed(args.seed, args.fold, disease, run))
            partitions.append([set(m) for m in detections])
        for name in ("baseline", "r1"):
            module = sets[name][disease]
            covered = sum(1 for p in partitions if any(len(module & m) / len(module) >= COVER for m in p))
            rows_out.append({"disease_id": disease, "variant": name,
                             "replication_heart": covered / max(len(partitions), 1)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows_out)
    table.to_parquet(out_dir / "heart_replication.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant", values="replication_heart").dropna()
    payload = {"fold": args.fold, "diseases": int(len(pivot)),
               "baseline": round(float(pivot.baseline.mean()), 3),
               "r1": round(float(pivot.r1.mean()), 3),
               "wins": int((pivot.r1 > pivot.baseline).sum()),
               "losses": int((pivot.r1 < pivot.baseline).sum()),
               "p": float(wilcoxon(pivot.r1, pivot.baseline).pvalue),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "heart_replication.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
