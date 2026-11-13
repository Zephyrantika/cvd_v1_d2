"""M4 原型：疾病特异性 z_spec + 结构零模型 + 随机表型查询零模型（fold 0 小样本）。

三个量回答三个不同问题，不能混：
    specificity  模块对当前疾病的信号，是否高于它对同折其它疾病的信号（设计文档 03 的 z_spec）；
    structure    模块内部密度，是否高于度数匹配的同大小随机子集；
    query null   当前疾病的表型查询，是否比一组匹配的随机表型查询传播得更集中（查询层面的对照）。

输出分三张表，不合并成"可靠性等级"——等级要等全量零模型和 GO 一致性有了再定。
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.evaluation.specificity import (  # noqa: E402
    background_by_similarity,
    background_diseases,
    phenotype_sets,
    specificity_for_module,
)
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.phenotype_null import (  # noqa: E402
    ic_bins,
    rwr_percentiles,
    sample_matched_query,
    term_neighborhoods,
)
from phenotype_modules_v0.nulls.structure_null import (  # noqa: E402
    build_adjacency,
    structure_null,
)
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--query-diseases", type=int, default=5)
    parser.add_argument("--query-draws", type=int, default=20)
    parser.add_argument("--structure-draws", type=int, default=200)
    parser.add_argument("--calibration-draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "nulls.log", "03_nulls")
    started = time.time()
    out_dir = paths.out("nulls", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    members_by_module = members.groupby("module_id")["gene_id"].apply(lambda s: list(map(str, s))).to_dict()

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    gene_position = {gene: i for i, gene in enumerate(matrix.gene_ids)}
    candidate_genes = {d: list(matrix.top_k(d, args.top_k)) for d in matrix.disease_ids}

    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "fold", "disease_family_id"])
    family_by_disease = dict(zip(folds["disease_id"].astype(str), folds["disease_family_id"].astype(str)))
    fold_diseases = tuple(sorted(matrix.disease_ids))

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    pheno_sets = phenotype_sets(phenotypes)

    # ---- 1. 疾病特异性 z_spec（两套背景） ----
    spec_started = time.time()
    spec_rows = []
    for row in modules.itertuples(index=False):
        disease = str(row.disease_id)
        module_id = str(row.module_id)
        genes = members_by_module[module_id]
        bg_family = background_diseases(disease, fold_diseases, family_by_disease)
        bg_sim = background_by_similarity(disease, fold_diseases, pheno_sets)
        by_family = specificity_for_module(
            matrix, percentiles, gene_position, module_id, disease, genes, bg_family, family_by_disease,
        )
        by_sim = specificity_for_module(
            matrix, percentiles, gene_position, module_id, disease, genes, bg_sim, family_by_disease,
        )
        spec_rows.append({
            "module_id": module_id,
            "disease_id": disease,
            "module_size": by_family.module_size,
            "score_self": by_family.score_self,
            "z_family": by_family.z_spec,
            "n_family": by_family.n_background,
            "p_family": max(1.0 / (by_family.n_background + 1), 1.0 - by_family.percentile_in_background),
            "z_sim": by_sim.z_spec,
            "n_sim": by_sim.n_background,
            "p_sim": max(1.0 / (by_sim.n_background + 1), 1.0 - by_sim.percentile_in_background),
        })
    specificity = pd.DataFrame(spec_rows)
    specificity.to_parquet(out_dir / "specificity.parquet", index=False)
    spec_seconds = round(time.time() - spec_started, 2)
    log.info("specificity_done", modules=len(specificity), seconds=spec_seconds,
             median_z_family=round(float(specificity["z_family"].median()), 3),
             median_z_sim=round(float(specificity["z_sim"].median()), 3),
             frac_p_sim_le_005=round(float((specificity["p_sim"] <= 0.05).mean()), 4))

    # ---- 2. 结构零模型 ----
    structure_started = time.time()
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)
    structure = structure_null(
        modules, members_by_module, candidate_genes, adjacency,
        draws=args.structure_draws, seed=args.seed,
    )
    structure.to_parquet(out_dir / "structure_null.parquet", index=False)
    structure_seconds = round(time.time() - structure_started, 2)
    log.info("structure_null_done", modules=len(structure), seconds=structure_seconds,
             frac_p_density_le_005=round(float((structure["p_density"] <= 0.05).mean()), 4))

    # ---- 3. 随机表型查询零模型（查询层面，不是特异性） ----
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    rwr_qc = json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )
    alpha = float(rwr_qc["selected_alpha"])

    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    edges = pd.read_parquet(paths.v0("data/processed/hpo_edges.parquet"),
                            columns=["child_hpo_id", "parent_hpo_id"])
    ic = hpo_ic.set_index("hpo_id")["ic"].astype(float)
    ic.index = ic.index.astype(str)
    graph_terms = set(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str))
    ic = ic.loc[[t for t in ic.index if t in graph_terms]]
    term_bin, pool = ic_bins(ic, n_bins=5)
    neighborhood = term_neighborhoods(edges, hops=2)
    log.info("query_null_setup", terms=len(ic), alpha=alpha, hops=2)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    selected = sorted(modules["disease_id"].astype(str).unique())[: args.query_diseases]
    rng = np.random.default_rng(args.seed + 1)
    query_rows: list[dict] = []

    for disease in selected:
        disease_terms = (
            phenotypes.loc[phenotypes["disease_id"].astype(str) == disease]
            .groupby("hpo_id", as_index=False)["frequency_weight"].max()
        )
        disease_terms = disease_terms.loc[disease_terms["hpo_id"].astype(str).isin(term_bin)]
        if len(disease_terms) < 3:
            continue
        terms = [str(t) for t in disease_terms["hpo_id"]]
        weights = sorted(disease_terms["frequency_weight"].astype(float).tolist(), reverse=True)
        records = []
        for draw in range(args.query_draws):
            sampled = sample_matched_query(terms, term_bin, pool, neighborhood, rng)
            query_id = f"{disease}#null{draw:03d}"
            records.extend(
                {"disease_id": query_id, "hpo_id": term, "frequency_weight": weight}
                for term, weight in zip(sampled, weights, strict=True)
            )
        fake = pd.DataFrame(records)
        seeds, _ = v0_rwr.build_seed_matrix(tuple(fake["disease_id"].unique()), fake, hpo_ic, node_map)
        rwr = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        null_percentiles = rwr_percentiles(rwr.scores, gene_rows)

        row_real = percentiles[matrix.disease_index(disease)]
        for module_id in modules.loc[modules["disease_id"].astype(str) == disease, "module_id"].astype(str):
            genes = [g for g in members_by_module[module_id] if g in gene_position]
            if not genes:
                continue
            rows = [gene_position[g] for g in genes]
            observed = float(np.mean(row_real[rows]))
            null_values = null_percentiles[rows, :].mean(axis=0)
            query_rows.append({
                "disease_id": disease,
                "module_id": module_id,
                "module_size": len(rows),
                "observed_mean_percentile": observed,
                "null_mean": float(null_values.mean()),
                "null_sd": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
                "z_query": float((observed - null_values.mean()) / (null_values.std(ddof=1) + 1e-9)),
                "p_query": (1 + int((null_values >= observed).sum())) / (len(null_values) + 1),
                "draws": int(null_percentiles.shape[1]),
            })
        log.info("query_null_disease_done", disease=disease, draws=args.query_draws)

    query = pd.DataFrame(query_rows)
    query.to_parquet(out_dir / "query_null.parquet", index=False)

    # ---- 4. 模块层面校准：用随机查询造模块，走同一套 z_spec ----
    # 目的：判断真实模块的 z_spec 是否只是"从 d 的 top 基因里挑出来的"这一选择效应。
    from phenotype_modules_v0.discovery.baseline import load_v0_models
    from phenotype_modules_v0.seeds import derive_seed

    detect, _ = load_v0_models(paths.v0_root)
    rng_calib = np.random.default_rng(args.seed + 2)
    calib_rows: list[dict] = []

    for disease in selected:
        disease_terms = (
            phenotypes.loc[phenotypes["disease_id"].astype(str) == disease]
            .groupby("hpo_id", as_index=False)["frequency_weight"].max()
        )
        disease_terms = disease_terms.loc[disease_terms["hpo_id"].astype(str).isin(term_bin)]
        if len(disease_terms) < 3:
            continue
        terms = [str(t) for t in disease_terms["hpo_id"]]
        weights = sorted(disease_terms["frequency_weight"].astype(float).tolist(), reverse=True)
        records = []
        for draw in range(args.calibration_draws):
            sampled = sample_matched_query(terms, term_bin, pool, neighborhood, rng_calib)
            query_id = f"{disease}#calib{draw:03d}"
            records.extend(
                {"disease_id": query_id, "hpo_id": term, "frequency_weight": weight}
                for term, weight in zip(sampled, weights, strict=True)
            )
        fake = pd.DataFrame(records)
        seeds, _ = v0_rwr.build_seed_matrix(tuple(fake["disease_id"].unique()), fake, hpo_ic, node_map)
        rwr = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        null_percentiles = rwr_percentiles(rwr.scores, gene_rows)

        bg = background_by_similarity(disease, fold_diseases, pheno_sets)
        bg_rows = np.array([matrix.disease_index(d) for d in bg if d in matrix.disease_ids])

        for column in range(null_percentiles.shape[1]):
            order = np.argsort(-null_percentiles[:, column], kind="stable")[: args.top_k]
            genes = [matrix.gene_ids[i] for i in order]
            modules_null = detect(
                genes, ppi,
                seed=derive_seed(args.seed, fold=args.fold, disease=f"{disease}#{column}",
                                 purpose="calibration", run=column) % (2**31),
            )
            for members_null in modules_null:
                rows = [gene_position[g] for g in members_null if g in gene_position]
                if len(rows) < 2:
                    continue
                score_self = float(null_percentiles[rows, column].mean())
                background_scores = percentiles[bg_rows][:, rows].mean(axis=1)
                calib_rows.append({
                    "disease_id": disease,
                    "draw": column,
                    "module_size": len(rows),
                    "z_calib": float((score_self - background_scores.mean())
                                     / (background_scores.std(ddof=1) + 1e-9)),
                })
        log.info("calibration_disease_done", disease=disease, draws=args.calibration_draws)

    calibration = pd.DataFrame(calib_rows)
    calibration.to_parquet(out_dir / "calibration_random_query.parquet", index=False)

    summary = {
        "fold": args.fold,
        "top_k": args.top_k,
        "seed": args.seed,
        "specificity": {
            "modules": int(len(specificity)),
            "background_rule": "family-excluded vs phenotype-Jaccard<0.5",
            "n_background_family_median": int(specificity["n_family"].median()),
            "n_background_sim_median": int(specificity["n_sim"].median()),
            "z_family_median": round(float(specificity["z_family"].median()), 3),
            "z_sim_median": round(float(specificity["z_sim"].median()), 3),
            "z_sim_q25": round(float(specificity["z_sim"].quantile(0.25)), 3),
            "z_sim_q75": round(float(specificity["z_sim"].quantile(0.75)), 3),
            "frac_sim_z_gt_2": round(float((specificity["z_sim"] > 2).mean()), 4),
            "frac_sim_p_le_0.05": round(float((specificity["p_sim"] <= 0.05).mean()), 4),
            "seconds": spec_seconds,
        },
        "structure": {
            "modules": int(len(structure)),
            "draws": args.structure_draws,
            "frac_p_density_le_0.05": round(float((structure["p_density"] <= 0.05).mean()), 4),
            "seconds": structure_seconds,
        },
        "calibration": {
            "draws_per_disease": args.calibration_draws,
            "null_modules": int(len(calibration)),
            "z_median": round(float(calibration["z_calib"].median()), 3) if len(calibration) else None,
            "frac_z_gt_2": round(float((calibration["z_calib"] > 2).mean()), 4) if len(calibration) else None,
            "mannwhitney_p_vs_real": (
                round(float(__import__("scipy.stats", fromlist=["mannwhitneyu"]).mannwhitneyu(
                    specificity["z_sim"], calibration["z_calib"], alternative="greater").pvalue), 6)
                if len(calibration) else None
            ),
        },
        "query_null": {
            "diseases": int(query["disease_id"].nunique()) if len(query) else 0,
            "modules": int(len(query)),
            "draws_per_disease": args.query_draws,
            "alpha": alpha,
            "frac_p_le_0.05": round(float((query["p_query"] <= 0.05).mean()), 4) if len(query) else None,
        },
        "total_seconds": round(time.time() - started, 2),
    }
    (out_dir / "nulls_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
