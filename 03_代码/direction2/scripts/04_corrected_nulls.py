"""M4 修正版：逐基因超额分位 + 同构随机模块零模型。

与 03 的区别：
    1. 特异性不再用原始分位差，改成逐基因超额分位，热度被减掉；
    2. 结构对照不再对"随机基因子集"，而是对"随机表型查询造出来的模块"，同算法同池子；
    3. 每个真实模块的 p 值来自同疾病上下文的随机模块分布，不再用随机查询下的真实模块。
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
import yaml
from scipy.sparse import load_npz
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models  # noqa: E402
from phenotype_modules_v0.evaluation.excess import (  # noqa: E402
    empirical_p,
    excess_z,
    gene_baseline,
    module_excess_by_disease,
)
from phenotype_modules_v0.evaluation.specificity import (  # noqa: E402
    background_by_similarity,
    phenotype_sets,
)
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.phenotype_null import (  # noqa: E402
    ic_bins,
    rwr_percentiles,
    sample_matched_query,
    term_neighborhoods,
)
from phenotype_modules_v0.nulls.structure_null import density_and_hub, build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", default="configs/nulls.yaml")
    parser.add_argument("--diseases", type=int, default=10)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = paths.package_root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    jaccard = float(cfg["background"]["jaccard_threshold"])
    top_k = int(cfg["metric"]["top_k"])
    size_tol = int(cfg["nulls"]["structure_size_tolerance"])

    log = JsonlLogger(paths.out("qc") / "nulls_corrected.log", "04_corrected_nulls")
    started = time.time()
    out_dir = paths.out("nulls_corrected", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("baselines", f"fold_{args.fold}", "module_members.parquet"))
    members_by_module = members.groupby("module_id")["gene_id"].apply(lambda s: list(map(str, s))).to_dict()

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    baseline = gene_baseline(percentiles)
    gene_position = {gene: i for i, gene in enumerate(matrix.gene_ids)}
    fold_diseases = tuple(sorted(matrix.disease_ids))

    phenotypes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id", "frequency_weight"],
    )
    pheno_sets = phenotype_sets(phenotypes)

    # ---- 真实模块的 z_excess ----
    real_rows = []
    background_rows_cache: dict[str, np.ndarray] = {}
    for row in modules.itertuples(index=False):
        disease = str(row.disease_id)
        module_id = str(row.module_id)
        genes = [g for g in members_by_module[module_id] if g in gene_position]
        indices = [gene_position[g] for g in genes]
        bg = background_by_similarity(disease, fold_diseases, pheno_sets, jaccard)
        bg_rows = np.array([matrix.disease_index(d) for d in bg if d in matrix.disease_ids])
        background_rows_cache[disease] = bg_rows
        score_self, mean_bg, sd_bg, z = excess_z(
            percentiles, baseline, indices, matrix.disease_index(disease), bg_rows,
        )
        real_rows.append({
            "module_id": module_id,
            "disease_id": disease,
            "module_size": len(indices),
            "score_self": score_self,
            "background_mean": mean_bg,
            "background_sd": sd_bg,
            "z_excess": z,
            "n_background": len(bg_rows),
        })
    real = pd.DataFrame(real_rows)

    # ---- 随机表型查询 → 同构随机模块 ----
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_node_ids = node_map.loc[node_map["node_type"].eq("gene"), "node_id"].astype(str).tolist()
    gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    row_of_gene = {gene: i for i, gene in enumerate(gene_node_ids)}
    align = np.array([row_of_gene[g] for g in matrix.gene_ids])  # matrix 基因顺序 -> 图内行号

    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    edges = pd.read_parquet(paths.v0("data/processed/hpo_edges.parquet"),
                            columns=["child_hpo_id", "parent_hpo_id"])
    ic = hpo_ic.set_index("hpo_id")["ic"].astype(float)
    ic.index = ic.index.astype(str)
    graph_terms = set(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str))
    ic = ic.loc[[t for t in ic.index if t in graph_terms]]
    term_bin, pool = ic_bins(ic, n_bins=5)
    neighborhood = term_neighborhoods(edges, hops=2)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    selected = sorted(modules["disease_id"].astype(str).unique())[: args.diseases]
    rng = np.random.default_rng(args.seed + 3)
    random_rows = []
    random_members = []

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
        for draw in range(args.draws):
            sampled = sample_matched_query(terms, term_bin, pool, neighborhood, rng)
            query_id = f"{disease}#rand{draw:03d}"
            records.extend(
                {"disease_id": query_id, "hpo_id": term, "frequency_weight": weight}
                for term, weight in zip(sampled, weights, strict=True)
            )
        fake = pd.DataFrame(records)
        seeds, _ = v0_rwr.build_seed_matrix(tuple(fake["disease_id"].unique()), fake, hpo_ic, node_map)
        rwr = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        pct_graph = rwr_percentiles(rwr.scores, gene_rows)      # 图内基因顺序 × draws
        pct = pct_graph[align, :]                               # 对齐到 matrix 基因顺序

        bg_rows = background_rows_cache.get(
            disease,
            np.array([i for i in range(len(fold_diseases)) if fold_diseases[i] != disease]),
        )

        for column in range(pct.shape[1]):
            order = np.argsort(-pct[:, column], kind="stable")[:top_k]
            genes = [matrix.gene_ids[i] for i in order]
            for index, module_genes in enumerate(detect(
                genes, ppi, seed=int((args.seed + column + 7919 * len(random_rows)) % (2**31 - 1)),
            )):
                indices = [gene_position[g] for g in module_genes if g in gene_position]
                if len(indices) < 5:
                    continue
                cur_pct = pct[:, column]
                excess = cur_pct[indices] - baseline[indices]
                score_self = float(excess.mean()) if len(indices) else float("nan")
                bg_scores = []
                for bg_row in bg_rows:
                    bg_scores.append(float((percentiles[bg_row, indices] - baseline[indices]).mean()))
                bg_scores = np.array(bg_scores)
                sd = float(bg_scores.std(ddof=1)) if len(bg_scores) > 1 else 0.0
                z = (score_self - float(bg_scores.mean())) / (sd + 1e-9)
                density, _ = density_and_hub(list(module_genes), adjacency)
                rid = f"{disease}#rand{column:03d}:M{index + 1}"
                random_rows.append({
                    "random_module_id": rid,
                    "disease_id": disease,
                    "draw": column,
                    "module_size": len(indices),
                    "density": density,
                    "z_excess": z,
                })
                random_members.extend(
                    {"random_module_id": rid, "gene_id": g} for g in module_genes
                )
        log.info("random_modules_done", disease=disease, draws=args.draws,
                 modules=sum(1 for r in random_rows if r["disease_id"] == disease))

    random_modules = pd.DataFrame(random_rows)
    random_modules.to_parquet(out_dir / "random_modules.parquet", index=False)
    pd.DataFrame(random_members).to_parquet(out_dir / "random_module_members.parquet", index=False)

    # ---- 比较与 p 值 ----
    real["p_from_random_modules"] = np.nan
    real["n_random_context"] = 0
    for index, row in real.iterrows():
        pool_z = random_modules.loc[random_modules["disease_id"] == row["disease_id"], "z_excess"]
        if len(pool_z) < 5:
            pool_z = random_modules["z_excess"]
        real.loc[index, "p_from_random_modules"] = empirical_p(float(row["z_excess"]), pool_z.to_numpy())
        real.loc[index, "n_random_context"] = int(len(pool_z))
    real.to_parquet(out_dir / "specificity_excess.parquet", index=False)

    # 结构：同构对照，按大小匹配
    density_rows = []
    for row in modules.itertuples(index=False):
        module_id, disease = str(row.module_id), str(row.disease_id)
        observed = float(row.internal_density)
        matched = random_modules.loc[
            (random_modules["disease_id"] == disease)
            & (random_modules["module_size"].sub(int(row.module_size)).abs() <= size_tol)
        ]
        if len(matched) < 5:
            matched = random_modules.loc[
                random_modules["module_size"].sub(int(row.module_size)).abs() <= size_tol
            ]
        density_rows.append({
            "module_id": module_id,
            "disease_id": disease,
            "module_size": int(row.module_size),
            "observed_density": observed,
            "random_density_mean": float(matched["density"].mean()) if len(matched) else float("nan"),
            "p_density_vs_random_modules": empirical_p(observed, matched["density"].to_numpy()),
            "n_matched": int(len(matched)),
        })
    density_table = pd.DataFrame(density_rows)
    density_table.to_parquet(out_dir / "structure_isomorphic.parquet", index=False)

    real_z = real["z_excess"].to_numpy()
    random_z = random_modules["z_excess"].to_numpy()
    p_mw = float(mannwhitneyu(real_z, random_z, alternative="greater").pvalue) if len(random_z) else float("nan")

    summary = {
        "fold": args.fold,
        "config": {"background": cfg["background"], "metric": cfg["metric"], "nulls": cfg["nulls"]},
        "real": {
            "modules": int(len(real)),
            "z_median": round(float(np.median(real_z)), 3),
            "z_q25": round(float(np.quantile(real_z, 0.25)), 3),
            "z_q75": round(float(np.quantile(real_z, 0.75)), 3),
            "frac_p_le_0.05": round(float((real["p_from_random_modules"] <= 0.05).mean()), 4),
        },
        "random": {
            "modules": int(len(random_modules)),
            "diseases": int(random_modules["disease_id"].nunique()) if len(random_modules) else 0,
            "draws": args.draws,
            "z_median": round(float(np.median(random_z)), 3) if len(random_z) else None,
        },
        "comparison": {
            "mannwhitney_p_real_greater": round(p_mw, 4),
            "median_diff": round(float(np.median(real_z) - np.median(random_z)), 3) if len(random_z) else None,
        },
        "structure": {
            "frac_p_le_0.05_vs_random_modules": round(
                float((density_table["p_density_vs_random_modules"] <= 0.05).mean()), 4
            ),
            "observed_density_median": round(float(density_table["observed_density"].median()), 3),
            "random_density_median": round(float(random_modules["density"].median()), 3) if len(random_modules) else None,
        },
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
