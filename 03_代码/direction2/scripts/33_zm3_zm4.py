"""文档 §4.4 的 ZM3 与 ZM4。

ZM3 真实 vs 随机网络：把 STRING 700 的基因—基因关系做度数保持随机化（double edge swap），
    保持同样的边数，重建转移矩阵，重跑 RWR → top-200 → Leiden，比较 z_spec 与 GO 一致性。
    如果随机网络上的模块和真实网络一样好，说明模块主要学的是度数。

ZM4 重叠零模型：同一折内的疾病对，观察值是两组模块的最大 Jaccard；零分布用同大小、
    按度数分层的随机基因集重算，得到 z_overlap 与 p_overlap。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency, gene_degrees  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BUDGET = {
    "phenotype": {"phenotype_is_a_phenotype": 0.40, "phenotype_associated_with_gene": 0.60},
    "gene": {"gene_associated_with_phenotype": 0.15, "gene_interacts_with_gene": 0.60,
             "gene_participates_in_pathway": 0.25},
    "pathway": {"pathway_contains_gene": 0.70, "pathway_part_of_pathway": 0.30},
}
EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
BACKDROP_SIZE = 20


def ic_similarity(sets: dict[str, set[str]], ic: dict[str, float]) -> dict[tuple[str, str], float]:
    norms = {d: float(np.sqrt(sum(ic.get(t, 0.0) for t in terms))) for d, terms in sets.items()}
    out: dict[tuple[str, str], float] = {}
    diseases = sorted(sets)
    for i, a in enumerate(diseases):
        for b in diseases[i + 1:]:
            shared = sets[a] & sets[b]
            out[(a, b)] = 0.0 if not shared or not norms[a] or not norms[b] else float(
                sum(ic.get(t, 0.0) for t in shared) / (norms[a] * norms[b]))
    return out


def backdrop(disease: str, fold_diseases: tuple[str, ...], family: dict[str, str],
             sim: dict[tuple[str, str], float]) -> list[str]:
    scored = []
    for other in fold_diseases:
        if other == disease or family.get(other) == family.get(disease):
            continue
        key = (disease, other) if (disease, other) in sim else (other, disease)
        scored.append((sim.get(key, 0.0), other))
    scored.sort(reverse=True)
    return [o for _, o in scored[:BACKDROP_SIZE]]


def best_go(genes: set[str], term_genes: dict[str, set[str]], gene_terms: dict[str, list[str]], universe: int):
    if not genes:
        return float("nan")
    terms: set[str] = set()
    for gene in genes:
        terms.update(gene_terms.get(gene, ()))
    if not terms:
        return float("nan")
    ordered = sorted(terms)
    ks = np.array([len(term_genes[t]) for t in ordered])
    xs = np.array([len(term_genes[t] & genes) for t in ordered])
    p = hypergeom.sf(xs - 1, universe, ks, len(genes))
    return float(multipletests(p, method="fdr_bh")[1].min())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "zm3_zm4.log", "33_zm3_zm4")
    started = time.time()
    out_dir = paths.out("nulls", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    from phenotype_network_v0.graph.graph_builder import compose_transition_matrix

    detect, _ = load_v0_models(paths.v0_root)
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)
    degrees = gene_degrees(adjacency)

    graph_dir = paths.v0(f"data/graphs/fold_{args.fold}")
    relations = {p.stem.removeprefix("A_"): load_npz(p).tocsr() for p in sorted(graph_dir.glob("A_*.npz"))}
    node_map = pd.read_parquet(graph_dir / "node_map.parquet")
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_indices = gene_rows["node_index"].to_numpy(dtype=int)
    gene_ids = gene_rows["node_id"].astype(str).tolist()

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    sets = {d: set(s.hpo_id) for d, s in phenotypes.groupby("disease_id")}
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "fold", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))
    fold_of = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["fold"].astype(int)))
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    test_ids = [
        line.strip()
        for line in paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])

    annotations = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                                  columns=["gene_id", "go_id", "evidence_code"])
    annotations["gene_id"] = annotations["gene_id"].astype(str)
    annotations = annotations.loc[~annotations["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(annotations["gene_id"], annotations["go_id"], strict=True):
        term_genes.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes = {t: g for t, g in term_genes.items() if MIN_TERM <= len(g) <= MAX_TERM}
    gene_terms = {g: [t for t in ts if t in term_genes] for g, ts in gene_terms.items()}
    universe = len(matrix.gene_ids)

    # ---------- ZM3：随机化网络的模块 ----------
    coo = relations["gene_gene"].tocoo()
    upper = coo.row < coo.col
    base_edges = list(zip(coo.row[upper].tolist(), coo.col[upper].tolist()))
    weights = coo.data[upper]
    graph = ig.Graph(n=relations["gene_gene"].shape[0], edges=base_edges, directed=False)
    graph.rewire(n=10 * max(graph.ecount(), 1), mode="simple")
    new_edges = graph.get_edgelist()
    from scipy.sparse import csr_matrix
    randomized = csr_matrix(
        (np.concatenate([weights[: len(new_edges)], weights[: len(new_edges)]]),
         (np.array([a for a, _ in new_edges] + [b for _, b in new_edges]),
          np.array([b for _, b in new_edges] + [a for a, _ in new_edges]))),
        shape=relations["gene_gene"].shape,
    )
    relations_random = dict(relations)
    relations_random["gene_gene"] = randomized
    similarities = ic_similarity(sets, ic_lookup)
    log.info("network_randomized", edges=len(new_edges))

    def modules_and_scores(rel: dict):
        transition = compose_transition_matrix(rel, BUDGET).tocsr()
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        pct = result.scores[gene_indices, :]
        rows = []
        for column, disease in enumerate(test_ids):
            order = np.argsort(-pct[:, column], kind="stable")[:200]
            genes = tuple(gene_ids[i] for i in order)
            detections = detect(genes, ppi, seed=stable_seed(args.seed, args.fold, disease, "base"))
            bg = backdrop(disease, tuple(test_ids), family, similarities)
            bg_rows = [matrix.disease_index(d) for d in bg if d in matrix.disease_ids]
            for members in detections:
                idx = [position[g] for g in members if g in position]
                if len(idx) < 5:
                    continue
                self_score = float(percentiles[matrix.disease_index(disease), idx].mean())
                background = np.array([percentiles[r, idx].mean() for r in bg_rows])
                z = (self_score - background.mean()) / (background.std(ddof=1) + 1e-9)
                rows.append({"disease_id": disease, "size": len(idx), "z_spec": float(z),
                             "go_fdr": best_go({gene_ids[i] for i in idx}, term_genes, gene_terms, universe)})
        return rows

    real_rows = modules_and_scores(relations)
    random_rows = modules_and_scores(relations_random)
    real = pd.DataFrame(real_rows)
    rnd = pd.DataFrame(random_rows)
    real["network"] = "real"
    rnd["network"] = "random"
    both = pd.concat([real, rnd], ignore_index=True)
    both.to_parquet(out_dir / "zm3_random_network.parquet", index=False)
    zm3 = both.groupby("network").agg(modules=("size", "size"), size_median=("size", "median"),
                                      z_median=("z_spec", "median"),
                                      go_median=("go_fdr", "median"),
                                      go_le_005=("go_fdr", lambda s: float((s <= 0.05).mean()))).round(4)
    zm3.to_csv(out_dir / "zm3_summary.csv", encoding="utf-8-sig")
    log.info("zm3_done", real=len(real), random=len(rnd))

    # ---------- ZM4：重叠零模型 ----------
    base_path = paths.out("baselines", f"fold_{args.fold}", "module_members.parquet")
    if not base_path.exists():
        base_path = paths.out("modules_doc", f"fold_{args.fold}", "members.parquet")
    base_members = pd.read_parquet(base_path)
    base_members["gene_id"] = base_members["gene_id"].astype(str)
    if "disease_id" not in base_members.columns:
        _mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"),
                                columns=["module_id", "disease_id"])
        _mods["disease_id"] = _mods["disease_id"].astype(str)
        base_members = base_members.merge(_mods, on="module_id", how="left")
    fold_set = {d for d in base_members["disease_id"].astype(str).unique() if fold_of.get(d) == args.fold}
    base_members = base_members[base_members["disease_id"].astype(str).isin(fold_set)]
    by_disease = base_members.groupby("disease_id")["gene_id"].apply(lambda s: set(map(str, s))).to_dict()
    diseases = sorted(by_disease)
    strata = pd.qcut([degrees.get(g, 0.0) for g in matrix.gene_ids], q=10, labels=False, duplicates="drop")

    gene_stratum = {g: int(strata[i]) for i, g in enumerate(matrix.gene_ids)}
    candidate_cache: dict[str, list[str]] = {}

    def sample_set(disease: str, size: int) -> set[str]:
        """按文档：在疾病自己的候选集 C_d 内抽同大小、按度数分层的随机模块。"""
        if disease not in candidate_cache:
            candidate_cache[disease] = list(matrix.top_k(disease, 200))
        pool = candidate_cache[disease]
        chosen: list[str] = []
        for label in sorted({gene_stratum[g] for g in pool}):
            subset = [g for g in pool if gene_stratum[g] == label]
            take = max(1, int(round(size * len(subset) / len(pool))))
            chosen.extend(rng.choice(np.array(subset), size=min(take, len(subset)), replace=False).tolist())
        return set(chosen[:size])

    pairs = []
    for _ in range(args.pairs):
        a, b = rng.choice(np.array(diseases), size=2, replace=False)
        if fold_of.get(a) != fold_of.get(b):
            continue
        set_a, set_b = by_disease[a], by_disease[b]
        observed = len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0
        null = []
        for _ in range(args.draws):
            sa = sample_set(a, len(set_a))
            sb = sample_set(b, len(set_b))
            union = sa | sb
            null.append(len(sa & sb) / len(union) if union else 0.0)
        null = np.array(null)
        pairs.append({"disease_a": a, "disease_b": b, "jaccard": observed,
                      "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                      "z_overlap": float((observed - null.mean()) / (null.std(ddof=1) + 1e-9)),
                      "p_overlap": (1 + int((null >= observed).sum())) / (len(null) + 1)})
    zm4 = pd.DataFrame(pairs)
    zm4.to_parquet(out_dir / "zm4_overlap.parquet", index=False)
    log.info("zm4_done", pairs=len(zm4))

    payload = {
        "fold": args.fold,
        "zm3": zm3.reset_index().to_dict(orient="records"),
        "zm4": {"pairs": int(len(zm4)),
                "frac_p_le_0.05": round(float((zm4["p_overlap"] <= 0.05).mean()), 4) if len(zm4) else None,
                "z_median": round(float(zm4["z_overlap"].median()), 3) if len(zm4) else None},
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "zm3_zm4_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
