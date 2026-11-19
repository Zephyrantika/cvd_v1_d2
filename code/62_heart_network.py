"""换网络实验二：用 HumanBase 心脏组织网络替换 STRING 做基因—基因边，重跑基线与共识。

HumanBase（GIANT 的继任者）心脏网络：6,161 万条边，落在候选集内 806 万条，覆盖 96.7% 候选基因、99.8% 已知基因。
对照：原 STRING 700 网络，同一流程。
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-prob", type=float, default=0.1)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "heart_network.log", "62_heart_network")
    started = time.time()
    out_dir = paths.out("heart_network", f"fold_{args.fold}")
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
    gene_index = {g: int(i) for g, i in zip(gene_ids, gene_rows["node_index"])}
    n_nodes = int(node_map["node_index"].max()) + 1

    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr()
                 for p in sorted(paths.v0(f"data/graphs/fold_{args.fold}").glob("A_*.npz"))}

    heart_path = Path(__file__).resolve().parents[1] / "data/external/networks/humanbase/heart_top.gz"
    heart = pd.read_csv(heart_path, sep="\t", header=None, names=["a", "b", "p"], dtype={"a": str, "b": str})
    heart = heart[heart["p"] >= args.min_prob]
    heart = heart[heart["a"].isin(gene_index) & heart["b"].isin(gene_index)]
    rows = heart["a"].map(gene_index).to_numpy()
    cols = heart["b"].map(gene_index).to_numpy()
    values = heart["p"].to_numpy(dtype=float)
    heart_matrix = csr_matrix((np.concatenate([values, values]),
                               (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                              shape=(n_nodes, n_nodes))
    heart_matrix.sum_duplicates()
    log.info("heart_matrix", edges=int(heart_matrix.nnz // 2))

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if args.limit:
        test_ids = test_ids[: args.limit]

    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    def z_of(disease: str, genes: list[str]) -> float:
        cols_ = [position[g] for g in genes if g in position]
        if len(cols_) < 5:
            return np.nan
        bg = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            bg.append((sim, other))
        bg = [d for _, d in sorted(bg, reverse=True)[:BACKDROP]]
        own = float(percentiles[matrix.disease_index(disease), cols_].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols_].mean() for b in bg])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    rows_out = []
    for label, relation in (("string", relations["gene_gene"]), ("heart", heart_matrix)):
        rel = dict(relations)
        rel["gene_gene"] = relation.tocsr()
        transition = compose_transition_matrix(rel, BUDGET).tocsr()
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        for column, disease in enumerate(test_ids):
            order = np.argsort(-scores[:, column], kind="stable")[:200]
            candidate = tuple(gene_ids[i] for i in order)
            detections = detect(candidate, ppi, seed=stable_seed(42, args.fold, disease, "heart"))
            if not detections:
                continue
            best = max(detections, key=len)
            rows_out.append({"disease_id": disease, "network": label, "size": len(best),
                             "z_spec": z_of(disease, list(best))})
        log.info("network_done", network=label)

    table = pd.DataFrame(rows_out)
    table.to_parquet(out_dir / "heart_network.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="network", values="z_spec").dropna()
    payload = {"fold": args.fold, "diseases": int(len(pivot)),
               "z_string": round(float(pivot.string.median()), 3),
               "z_heart": round(float(pivot.heart.median()), 3),
               "heart_better": int((pivot.heart > pivot.string).sum()),
               "p": float(wilcoxon(pivot.heart, pivot.string).pvalue) if len(pivot) > 5 else None,
               "seconds": round(time.time() - started, 2)}
    (out_dir / "heart_network.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                                encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
