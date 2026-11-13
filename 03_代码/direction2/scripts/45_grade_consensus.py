"""给 100 次扰动共识模块做分级，与 V0 模块在同一批疾病上比较。

共识模块来自 34 号脚本（fold 0，15 个疾病）。这里补它缺的三样：
p_spec（100 次随机表型查询）、hub/density（从 PPI 与 C_d 算）、GO FDR；
稳定性用共识的平均共现（mean_cooccurrence）作代理。
然后按 §4.5 的 A/B/C 条件评级，与同疾病的 V0 模块对比。
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
from scipy.sparse import load_npz
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
DRAWS = 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "grade_consensus.log", "45_grade_consensus")
    started = time.time()
    out_dir = paths.out("modules_doc", f"fold_{args.fold}", "consensus")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    members = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    meta = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet"))

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_by_term = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    ic_values = np.array(list(ic_by_term.values()))
    edges = np.quantile(ic_values, np.linspace(0, 1, 11))
    bins: dict[int, list[str]] = {}
    for term, value in ic_by_term.items():
        index = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, 9))
        bins.setdefault(index, []).append(term)

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                        node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8"))["selected_alpha"])
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]

    go = pd.read_parquet(paths.v0("data/processed/go_gene_annotations.parquet"),
                         columns=["gene_id", "go_id", "evidence_code"])
    go["gene_id"] = go["gene_id"].astype(str)
    go = go.loc[~go["evidence_code"].astype(str).isin(EXCLUDE_EVIDENCE)]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(go["gene_id"], go["go_id"], strict=True):
        term_genes.setdefault(str(term), set()).add(gene)
        gene_terms.setdefault(gene, []).append(str(term))
    term_genes = {t: g for t, g in term_genes.items() if MIN_TERM <= len(g) <= MAX_TERM}
    gene_terms = {g: [t for t in ts if t in term_genes] for g, ts in gene_terms.items()}
    universe = len(matrix.gene_ids)

    def go_fdr(genes: list[str]) -> float:
        target = set(genes)
        terms: set[str] = set()
        for gene in target:
            terms.update(gene_terms.get(gene, ()))
        if not terms:
            return float("nan")
        ordered = sorted(terms)
        ks = np.array([len(term_genes[t]) for t in ordered])
        xs = np.array([len(term_genes[t] & target) for t in ordered])
        return float(multipletests(hypergeom.sf(xs - 1, universe, ks, len(target)), method="fdr_bh")[1].min())

    def topology(genes: list[str], candidate: set[str]):
        member = set(genes)
        internal = 0
        weighted = {g: 0.0 for g in member}
        for gene in member:
            for neighbour, weight in adjacency.get(gene, {}).items():
                if neighbour in member:
                    internal += 1
                    weighted[gene] += weight
        possible = len(member) * (len(member) - 1) / 2
        density = (internal / 2) / possible if possible else 0.0
        total = sum(weighted.values())
        hub = 2.0 * max(weighted.values(), default=0.0) / total if total else 1.0
        return float(density), float(hub)

    # p_spec：每个疾病 100 次随机表型查询
    rows = []
    for disease, group in members.groupby("disease_id"):
        terms = list(terms_of.get(str(disease), set()))
        if len(terms) < 3:
            continue
        strata = [int(np.clip(np.searchsorted(edges, ic_by_term.get(t, 0.0), side="right") - 1, 0, 9))
                  for t in terms]
        queries = []
        for _ in range(DRAWS):
            sampled = {str(rng.choice(bins.get(b) or list(ic_by_term))) for b in strata}
            queries.append(sampled)
        seeds = np.zeros((transition.shape[0], DRAWS))
        for column, query in enumerate(queries):
            indices = [hpo_node[t] for t in query if t in hpo_node]
            if not indices:
                indices = [int(rng.choice(list(hpo_node.values())))]
            unique = np.unique(indices)
            seeds[unique, column] = 1.0 / len(unique)
        sums = seeds.sum(axis=0)
        sums[sums <= 0] = 1.0
        seeds = seeds / sums
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        random_scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        gene_ids = gene_rows["node_id"].astype(str).tolist()
        gene_pos = {g: i for i, g in enumerate(gene_ids)}
        random_pct = np.empty_like(random_scores)
        for column in range(random_scores.shape[1]):
            order = np.argsort(-random_scores[:, column], kind="stable")
            rank = np.empty(len(order))
            rank[order] = np.arange(1, len(order) + 1)
            random_pct[:, column] = 1.0 - (rank - 1) / (len(order) - 1)

        candidate = set(matrix.top_k(str(disease), 200))
        for module_id, module_group in group.groupby("module_id"):
            genes = [g for g in module_group["gene_id"].astype(str) if g in position]
            if len(genes) < 5:
                continue
            observed = float(percentiles[matrix.disease_index(str(disease)),
                                         [position[g] for g in genes]].mean())
            idx = [gene_pos[g] for g in genes if g in gene_pos]
            null = random_pct[idx, :].mean(axis=0)
            p_spec = (1 + int((null >= observed).sum())) / (DRAWS + 1)
            density, hub = topology(genes, candidate)
            rows.append({"module_id": module_id, "disease_id": str(disease), "size": len(genes),
                         "p_spec": float(p_spec), "density": density, "hub": hub,
                         "go_fdr": go_fdr(genes)})
        log.info("disease_done", disease=str(disease))

    table = pd.DataFrame(rows).merge(meta[["module_id", "mean_cooccurrence", "core_size"]],
                                     on="module_id", how="left")
    table["q_spec"] = multipletests(table["p_spec"], method="fdr_bh")[1]
    table["stability"] = table["mean_cooccurrence"]
    table["p_struct"] = 0.0  # ZM2 饱和，前面已确认

    def grade(row):
        if (row.p_spec <= 0.01 and row.stability >= 0.7 and row.hub <= 0.5 and row.go_fdr <= 0.05):
            return "A"
        if (row.p_spec <= 0.05 and row.stability >= 0.6 and row.hub <= 0.5):
            return "B"
        return "C"

    table["reliability_tier"] = table.apply(grade, axis=1)
    table.to_parquet(out_dir / "consensus_graded.parquet", index=False)

    v0 = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    v0 = v0[v0.disease_id.astype(str).isin(set(table.disease_id))]
    summary = {
        "fold": args.fold,
        "consensus_modules": int(len(table)),
        "consensus_frac_A": round(float((table.reliability_tier == "A").mean()), 3),
        "consensus_z_median": round(float(table.get("z_spec", pd.Series([np.nan])).median()), 3) if "z_spec" in table else None,
        "consensus_stability_median": round(float(table.stability.median()), 3),
        "v0_modules": int(len(v0)),
        "v0_frac_A": round(float((v0.reliability_tier == "A").mean()), 3),
        "v0_stability_median": round(float(v0.stability.median()), 3),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "consensus_grading.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
