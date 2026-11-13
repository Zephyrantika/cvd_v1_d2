"""文档 §4.5 可靠性分级 + §7 输出 schema。

输入：32 号脚本产出的 z_spec/p_struct/density/conductance/hub/stability（fold 0）。
补两样：p_spec（ZM1 随机表型，按术语数与 IC 分层匹配）、GO FDR；然后按 §4.5 分级并做折内 BH。

输出（按 §7 schema）：
    modules.parquet / members.parquet / nulls.parquet / shared.parquet
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
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

EXCLUDE_EVIDENCE = {"IEA"}
MIN_TERM, MAX_TERM = 10, 500
BACKDROP = 20
RANDOM_DRAWS = 0  # 运行时由 --draws 覆盖


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--limit", action="store_true", help="只用共识模块覆盖的疾病做试点")
    args = parser.parse_args()

    global RANDOM_DRAWS
    RANDOM_DRAWS = args.draws
    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "grading.log", "35_grading")
    started = time.time()
    out_dir = paths.out("modules_doc", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    schema_path = paths.out("compliance", f"modules_doc_schema_fold{args.fold}.parquet")
    if not schema_path.exists():
        schema_path = paths.out("compliance", "modules_doc_schema.parquet")
    modules = pd.read_parquet(schema_path)
    modules = modules[modules["fold"] == args.fold].copy()
    if args.limit:
        consensus_path = paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet")
        if consensus_path.exists():
            keep = set(pd.read_parquet(consensus_path)["disease_id"].astype(str))
            modules = modules[modules["disease_id"].astype(str).isin(keep)].copy()
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"),
                              columns=["module_id", "gene_id", "consensus_frequency"])
    members["gene_id"] = members["gene_id"].astype(str)
    member_lists = members.groupby("module_id")["gene_id"].apply(list).to_dict()

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    disease_terms = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: set(s)).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_by_term = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    ic_values = np.array(list(ic_by_term.values()))
    ic_edges = np.quantile(ic_values, np.linspace(0, 1, 11))
    term_by_bin: dict[int, list[str]] = {}
    for term, value in ic_by_term.items():
        bin_index = int(np.clip(np.searchsorted(ic_edges, value, side="right") - 1, 0, 9))
        term_by_bin.setdefault(bin_index, []).append(term)

    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])
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
        p = hypergeom.sf(xs - 1, universe, ks, len(target))
        return float(multipletests(p, method="fdr_bh")[1].min())

    module_rows = modules.set_index("module_id")
    p_spec_rows = []
    for disease, group in modules.groupby("disease_id"):
        terms = list(disease_terms.get(disease, set()))
        if len(terms) < 3:
            continue
        bins = [int(np.clip(np.searchsorted(ic_edges, ic_by_term.get(t, 0.0), side="right") - 1, 0, 9)) for t in terms]
        random_queries = []
        for _ in range(RANDOM_DRAWS):
            sampled_terms = set()
            for bin_index in bins:
                pool = term_by_bin.get(bin_index) or list(ic_by_term)
                sampled_terms.add(str(rng.choice(pool)))
            random_queries.append(sampled_terms)
        hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                            node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
        seeds = np.zeros((transition.shape[0], RANDOM_DRAWS))
        for column, terms in enumerate(random_queries):
            indices = [hpo_node[t] for t in terms if t in hpo_node]
            if not indices:
                indices = [int(rng.choice(list(hpo_node.values())))]
            unique = np.unique(indices)
            seeds[unique, column] = 1.0 / len(unique)
        column_sums = seeds.sum(axis=0)
        column_sums[column_sums <= 0] = 1.0
        seeds = seeds / column_sums
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        random_scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        gene_ids = gene_rows["node_id"].astype(str).tolist()
        random_percentiles = np.empty_like(random_scores)
        for column in range(random_scores.shape[1]):
            order = np.argsort(-random_scores[:, column], kind="stable")
            rank = np.empty(len(order))
            rank[order] = np.arange(1, len(order) + 1)
            random_percentiles[:, column] = 1.0 - (rank - 1) / (len(order) - 1)
        gene_pos = {g: i for i, g in enumerate(gene_ids)}

        for module_id in group["module_id"]:
            genes = [g for g in member_lists.get(module_id, []) if g in gene_pos]
            if not genes:
                continue
            idx = [gene_pos[g] for g in genes]
            observed = float(percentiles[matrix.disease_index(disease), [position[g] for g in genes]].mean())
            null = random_percentiles[idx, :].mean(axis=0)
            p_spec = (1 + int((null >= observed).sum())) / (RANDOM_DRAWS + 1)
            p_spec_rows.append({"module_id": module_id, "p_spec": float(p_spec)})
        log.info("p_spec_done", disease=disease, modules=len(p_spec_rows))

    p_spec_frame = pd.DataFrame(p_spec_rows)
    modules = modules.merge(p_spec_frame, on="module_id", how="left")
    modules["p_spec"] = modules["p_spec"].fillna(1.0)
    modules["q_spec"] = multipletests(modules["p_spec"], method="fdr_bh")[1]

    modules["go_fdr"] = [go_fdr(member_lists.get(m, [])) for m in modules["module_id"]]
    modules["q_struct"] = multipletests(modules["p_struct"], method="fdr_bh")[1]

    def grade(row) -> str:
        base = (row["p_spec"] <= 0.01 and row["p_struct"] <= 0.05 and row["stability"] >= 0.7
                and row["hub"] <= 0.5 and row["go_fdr"] <= 0.05)
        if base:
            return "A"
        if (row["p_spec"] <= 0.05 and row["p_struct"] <= 0.05 and row["stability"] >= 0.6
                and row["hub"] <= 0.5):
            return "B"
        return "C"

    modules["reliability_tier"] = modules.apply(grade, axis=1)
    modules["method"] = "leiden_consensus"

    modules_out = modules[[
        "module_id", "disease_id", "fold", "method", "size", "z_spec", "p_spec", "q_spec",
        "density", "conductance", "hub", "stability", "core_size", "core_jaccard", "tier",
        "reliability_tier",
    ]].copy()
    modules_out.to_parquet(out_dir / "modules.parquet", index=False)

    member_rows = []
    for module_id in modules_out["module_id"]:
        for gene in member_lists.get(module_id, []):
            if gene not in position:
                continue
            role = "core" if members.loc[(members.module_id == module_id) & (members.gene_id == gene),
                                         "consensus_frequency"].iloc[0] >= 0.8 else "periphery"
            disease = module_rows.loc[module_id, "disease_id"]
            member_rows.append({"module_id": module_id, "gene_id": gene, "role": role,
                                "frequency": float(members.loc[(members.module_id == module_id)
                                                               & (members.gene_id == gene), "consensus_frequency"].iloc[0]),
                                "r_d": float(percentiles[matrix.disease_index(disease), position[gene]])})
    pd.DataFrame(member_rows).to_parquet(out_dir / "members.parquet", index=False)

    nulls = pd.DataFrame([
        {"module_id": row.module_id, "null_type": "ZM1_random_phenotype", "observed": row.z_spec,
         "null_mean": float("nan"), "null_sd": float("nan"), "p": row.p_spec, "q": row.q_spec}
        for row in modules_out.itertuples()
    ] + [
        {"module_id": row.module_id, "null_type": "ZM2_degree_preserving", "observed": row.density,
         "null_mean": float("nan"), "null_sd": float("nan"), "p": float("nan"), "q": float("nan")}
        for row in modules_out.itertuples()
    ])
    nulls.to_parquet(out_dir / "nulls.parquet", index=False)

    shared_path = paths.out("nulls", f"fold_{args.fold}", "zm4_overlap.parquet")
    if shared_path.exists():
        shared = pd.read_parquet(shared_path).rename(columns={"disease_a": "disease_a", "disease_b": "disease_b"})
        shared.to_parquet(out_dir / "shared.parquet", index=False)

    summary = {
        "fold": args.fold,
        "modules": int(len(modules_out)),
        "reliability_counts": modules_out["reliability_tier"].value_counts().to_dict(),
        "frac_A": round(float((modules_out["reliability_tier"] == "A").mean()), 4),
        "frac_B": round(float((modules_out["reliability_tier"] == "B").mean()), 4),
        "p_spec_median": round(float(modules_out["p_spec"].median()), 4),
        "q_spec_le_005": round(float((modules_out["q_spec"] <= 0.05).mean()), 4),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "grading_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
