"""给精修模块做 A/B/C 分级：用随机表型零模型（ZM1）计算 p_spec。

做法：对每个疾病取与其表型数相同的随机 HPO 术语集，跑 RWR，算精修模块在该次随机查询下的 z；
重复 100 次得到零分布，p = (1 + #{null z >= observed z}) / (B + 1)，再按疾病做 BH。
其余三项（稳定性、hub、GO）用已有结果。分级规则与 35_grading.py 一致。
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

DRAWS = 100
BACKDROP = 20


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--members", default=None)
    parser.add_argument("--tag", default="lam1.0_mu1.0")
    parser.add_argument("--rep-source", default="lam_refine")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "refined_grading.log", "73_refined_grading")
    started = time.time()
    out_dir = paths.out("refined_grading", f"fold_{args.fold}_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    hpo_rows = node_map.loc[node_map["node_type"].eq("phenotype")]
    hpo_index = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    all_terms = list(hpo_index)
    n_nodes = int(node_map["node_index"].max()) + 1

    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])

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

    if args.members:
        refined = pd.read_parquet(args.members)
    else:
        refined = pd.read_parquet(paths.out("lam_refine", f"fold_{args.fold}_lam1.0_mu1.0",
                                            "lam_refine_members.parquet"))
    refined["gene_id"] = refined["gene_id"].astype(str)
    modules = refined.groupby("disease_id")["gene_id"].apply(set).to_dict()
    diseases = [d for d in matrix.disease_ids if d in modules]
    log.info("diseases", n=len(diseases))

    backgrounds = {}
    for disease in diseases:
        scored = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        backgrounds[disease] = [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_from(pct: np.ndarray, disease: str, genes: set[str]) -> float:
        cols = [position[g] for g in genes if g in position]
        if len(cols) < 5:
            return np.nan
        row = matrix.disease_index(disease) if disease in matrix.disease_ids else None
        own = pct[row, cols].mean() if row is not None else np.nan
        back = []
        for other in backgrounds[disease]:
            other_row = matrix.disease_index(other)
            back.append(float(pct[other_row, cols].mean()))
        back = np.array(back)
        if back.size < 2 or back.std(ddof=1) == 0:
            return np.nan
        return float((own - back.mean()) / back.std(ddof=1))

    observed = {d: z_from(percentiles, d, modules[d]) for d in diseases}

    null_matrix = np.full((len(diseases), args.draws), np.nan)
    for draw in range(args.draws):
        seeds = np.zeros((n_nodes, len(diseases)))
        for column, disease in enumerate(diseases):
            size = len(terms_of.get(disease, set()))
            sample = rng.choice(all_terms, size=min(size, len(all_terms)), replace=False)
            indices = [hpo_index[t] for t in sample if t in hpo_index]
            if not indices:
                continue
            unique = np.unique(indices)
            seeds[unique, column] = 1.0 / len(unique)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        pct = np.zeros_like(raw)
        from scipy.stats import rankdata
        for column in range(raw.shape[1]):
            pct[:, column] = rankdata(raw[:, column]) / raw.shape[0]
        for i, disease in enumerate(diseases):
            row = matrix.disease_index(disease)
            cols = [position[g] for g in modules[disease] if g in position]
            if not cols:
                continue
            own = pct[cols, i].mean()
            back = [pct[cols, diseases.index(o)].mean() for o in backgrounds[disease] if o in diseases]
            if len(back) >= 2 and np.std(back, ddof=1) > 0:
                null_matrix[i, draw] = (own - np.mean(back)) / np.std(back, ddof=1)
        if draw % 10 == 0:
            log.info("draw_done", draw=draw)

    rows = []
    for i, disease in enumerate(diseases):
        nulls = null_matrix[i][~np.isnan(null_matrix[i])]
        obs = observed[disease]
        p = (1 + int((nulls >= obs).sum())) / (len(nulls) + 1) if len(nulls) and not np.isnan(obs) else np.nan
        rows.append({"disease_id": disease, "z_spec": obs, "p_spec": p, "null_mean": float(np.mean(nulls))})
    table = pd.DataFrame(rows)
    q = multipletests(table["p_spec"].fillna(1.0), method="fdr_bh")[1]
    table["q_spec"] = q

    # 稳定性（独立扰动复现率）来自 lam_refine
    if args.rep_source == "lam_refine":
        lam = pd.read_parquet(paths.out("lam_refine", f"fold_{args.fold}_lam1.0_mu1.0", "lam_refine.parquet"))
        lam = lam[lam.variant == "refined"][["disease_id", "replication"]]
        table = table.merge(lam, on="disease_id", how="left")
    else:
        rep = pd.read_parquet(args.rep_source)[["disease_id", "replication"]]
        table = table.merge(rep, on="disease_id", how="left")

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    def hub_of(genes: set[str]) -> float:
        weighted = {}
        for gene in genes:
            total = sum(weight for other, weight in adjacency.get(gene, {}).items() if other in genes)
            weighted[gene] = total
        overall = sum(weighted.values())
        return float(2 * max(weighted.values(), default=0) / overall) if overall else 1.0

    table["hub"] = [hub_of(modules[d]) for d in table.disease_id]

    go = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                         columns=["gene_id", "go_id", "evidence_code"])
    go["gene_id"] = go["gene_id"].astype(str)
    go = go[~go["evidence_code"].astype(str).isin({"IEA"})]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(go["gene_id"], go["go_id"].astype(str)):
        term_genes.setdefault(term, set()).add(gene)
        gene_terms.setdefault(gene, []).append(term)
    term_genes = {t: g for t, g in term_genes.items() if 10 <= len(g) <= 500}
    gene_terms = {g: [t for t in ts if t in term_genes] for g, ts in gene_terms.items()}
    universe_size = 16992

    def go_fdr(genes: set[str]) -> float:
        target = set(genes)
        terms = set()
        for gene in target:
            terms.update(gene_terms.get(gene, []))
        if not terms:
            return np.nan
        ordered = sorted(terms)
        k = np.array([len(term_genes[t]) for t in ordered])
        x = np.array([len(term_genes[t] & target) for t in ordered])
        return float(multipletests(hypergeom.sf(x - 1, universe_size, k, len(target)), method="fdr_bh")[1].min())

    table["go_fdr"] = [go_fdr(modules[d]) for d in table.disease_id]

    def grade(row) -> str:
        if (row.p_spec <= 0.01 and row.replication >= 0.7 and row.hub <= 0.5 and row.go_fdr <= 0.05):
            return "A"
        if (row.p_spec <= 0.05 and row.replication >= 0.6 and row.hub <= 0.5):
            return "B"
        return "C"

    table["grade"] = [grade(row) for row in table.itertuples(index=False)]
    table.to_parquet(out_dir / "refined_graded.parquet", index=False)
    counts = table["grade"].value_counts().to_dict()
    payload = {"fold": args.fold, "draws": args.draws, "diseases": int(len(table)),
               "counts": counts, "frac_A": round(counts.get("A", 0) / len(table), 4),
               "frac_B": round(counts.get("B", 0) / len(table), 4),
               "p_spec_median": round(float(table.p_spec.median()), 4),
               "q_spec_le_005": round(float((table.q_spec <= 0.05).mean()), 4),
               "hub_median": round(float(table.hub.median()), 3),
               "go_median": float(table.go_fdr.median()),
               "rep_mean": round(float(table.replication.mean()), 3),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "refined_grading.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
