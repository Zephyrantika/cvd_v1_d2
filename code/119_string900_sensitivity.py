"""网络版本敏感性：把 PPI 从 STRING 700 换成 900，看基线模块的 A 级比例是否保持。"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}
COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    out = paths.out("string900", f"fold_{args.fold}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_index = {g: int(i) for g, i in zip(gene_ids, gene_rows["node_index"])}
    n_nodes = int(node_map["node_index"].max()) + 1

    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr()
                 for p in sorted(paths.v0(f"data/graphs/fold_{args.fold}").glob("A_*.npz"))}
    edges900 = pd.read_parquet(paths.v0("data/processed/string_gene_edges_900.parquet"),
                               columns=["gene_a", "gene_b", "weight"])
    rows = edges900["gene_a"].astype(str).map(gene_index)
    cols = edges900["gene_b"].astype(str).map(gene_index)
    keep = rows.notna() & cols.notna()
    rows = rows[keep].to_numpy(dtype=int)
    cols = cols[keep].to_numpy(dtype=int)
    w = edges900["weight"].to_numpy(dtype=float)[keep.to_numpy()]
    ppi700 = csr_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]),
                                                 np.concatenate([cols, rows]))), shape=(n_nodes, n_nodes))
    rel = dict(relations)
    rel["gene_gene"] = ppi700
    transition = compose_transition_matrix(rel, BUDGET).tocsr()
    print(f"[info] 900 edges {ppi700.nnz // 2}", flush=True)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"))
    if "frequency_weight" not in phenotypes.columns:
        phenotypes["frequency_weight"] = 1.0
    phenotypes = phenotypes[["disease_id", "hpo_id", "frequency_weight"]]
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
    result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
    raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
    from scipy.stats import rankdata
    pct = np.vstack([rankdata(raw[:, c]) / raw.shape[0] for c in range(raw.shape[1])]).T

    ppi_edges = pd.read_parquet(paths.v0("data/processed/string_gene_edges_900.parquet"),
                                columns=["gene_a", "gene_b", "weight"])
    members = {}
    for col, disease in enumerate(test_ids):
        order = np.argsort(-pct[:, col])[:200]
        candidate = tuple(gene_ids[i] for i in order)
        detections = detect(candidate, ppi_edges, seed=stable_seed(119, args.fold, disease, 0))
        if detections:
            members[disease] = set(max(detections, key=len))

    hpo_rows = node_map.loc[node_map["node_type"] == "phenotype"]
    hpo_index = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    rows_out, member_rows = [], []
    for disease, module in members.items():
        parts = []
        for run in range(args.runs):
            kept = [t for t in terms_of.get(disease, []) if rng.random() > 0.10]
            indices = [hpo_index[t] for t in kept if t in hpo_index]
            if not indices:
                continue
            sd = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            sd[unique, 0] = 1.0 / len(unique)
            res = v0_rwr.run_rwr(transition, sd, alpha=alpha)
            sc = res.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-sc)[:200]
            det = detect(tuple(gene_ids[i] for i in top), ppi_edges,
                         seed=stable_seed(119, args.fold, disease, run))
            parts.append([set(m) for m in det])
        covered = sum(1 for p in parts if any(len(module & m) / len(module) >= COVER for m in p))
        rows_out.append({"disease_id": disease, "replication": covered / max(len(parts), 1)})
        for gene in module:
            member_rows.append({"module_id": f"{disease}:s900", "disease_id": disease, "gene_id": gene})

    pd.DataFrame(rows_out).to_parquet(out / "selected.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out / "members.parquet", index=False)
    summary = {"fold": args.fold, "diseases": len(members),
               "rep_mean": round(float(pd.DataFrame(rows_out).replication.mean()), 3),
               "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
