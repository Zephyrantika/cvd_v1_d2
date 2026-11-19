"""标签参与传播：表型种子 + 已知基因种子，重新生成模块并计算 z 与随机表型零模型的 p。

种子构成：表型术语总权重 1.0，已知基因总权重 w（默认 0.5）。
零模型：表型随机替换为同数量随机术语，基因种子保持不变——检验「表型那一路」是否还带来特异性。
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
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gene-weight", type=float, default=0.5)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "gene_seeded.log", "76_gene_seeded")
    started = time.time()
    out_dir = paths.out("gene_seeded", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_node = {g: int(i) for g, i in zip(gene_ids, gene_rows["node_index"])}
    hpo_rows = node_map.loc[node_map["node_type"].eq("phenotype")]
    hpo_node = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    all_terms = list(hpo_node)
    n_nodes = int(node_map["node_index"].max()) + 1

    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()

    chain = Path(__file__).resolve().parents[1] / "data/chain"
    labels = pd.read_parquet(chain / "fact_disease_gene.parquet")
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["gene_id"] = labels["gene_id"].astype(str)
    curated = labels[labels["source"].isin(["original", "clinvar", "clingen", "disgenet"])]
    genes_of = curated.groupby("disease_id")["gene_id"].apply(set).to_dict()

    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))
    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    diseases = [d for d in test_ids if d in terms_of and d in genes_of]

    def background(disease: str) -> list[str]:
        scored = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def build_seeds(term_sets: dict[str, list[str]]) -> np.ndarray:
        seeds = np.zeros((n_nodes, len(diseases)))
        for column, disease in enumerate(diseases):
            terms = [hpo_node[t] for t in term_sets[disease] if t in hpo_node]
            genes = [gene_node[g] for g in genes_of.get(disease, set()) if g in gene_node]
            if terms:
                unique = np.unique(terms)
                seeds[unique, column] += 1.0 / len(unique)
            if genes:
                unique = np.unique(genes)
                seeds[unique, column] += args.gene_weight / len(unique)
            total = seeds[:, column].sum()
            if total > 0:
                seeds[:, column] /= total
        return seeds

    def percentiles(raw: np.ndarray) -> np.ndarray:
        out = np.zeros_like(raw)
        for column in range(raw.shape[1]):
            out[:, column] = rankdata(raw[:, column]) / raw.shape[0]
        return out

    def z_of(pct: np.ndarray, disease: str, genes: set[str]) -> float:
        idx = diseases.index(disease)
        cols = [position[g] for g in genes if g in position]
        if not cols:
            return np.nan
        own = float(pct[cols, idx].mean())
        back = [float(pct[cols, diseases.index(o)].mean()) for o in background(disease) if o in diseases]
        if len(back) < 2 or np.std(back, ddof=1) == 0:
            return np.nan
        return float((own - np.mean(back)) / np.std(back, ddof=1))

    # 观测
    seeds = build_seeds({d: sorted(terms_of[d]) for d in diseases})
    result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
    raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
    pct = percentiles(raw)
    modules: dict[str, set[str]] = {}
    for column, disease in enumerate(diseases):
        ranking = np.argsort(-pct[:, column])[:200]
        candidate = tuple(gene_ids[i] for i in ranking)
        detections = detect(candidate, ppi, seed=stable_seed(5, args.fold, disease, "geneseed"))
        if not detections:
            continue
        modules[disease] = set(max(detections, key=len))
    log.info("modules", n=len(modules))
    pd.DataFrame([{"module_id": f"{d}:geneseed", "disease_id": d, "gene_id": g}
                  for d, genes in modules.items() for g in genes]).to_parquet(
        out_dir / "geneseeded_members.parquet", index=False)

    observed = {d: z_of(pct, d, m) for d, m in modules.items()}

    # 零模型：随机表型 + 固定基因种子
    nulls = {d: [] for d in modules}
    for draw in range(args.draws):
        random_terms = {}
        for disease in diseases:
            size = len(terms_of[disease])
            random_terms[disease] = rng.choice(all_terms, size=min(size, len(all_terms)), replace=False).tolist()
        seeds = build_seeds(random_terms)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        pct = percentiles(raw)
        for disease, genes in modules.items():
            value = z_of(pct, disease, genes)
            if not np.isnan(value):
                nulls[disease].append(value)
        if draw % 5 == 0:
            log.info("null_draw", draw=draw)

    rows = []
    for disease, genes in modules.items():
        values = np.array(nulls[disease])
        obs = observed[disease]
        p = (1 + int((values >= obs).sum())) / (len(values) + 1) if len(values) and not np.isnan(obs) else np.nan
        rows.append({"disease_id": disease, "z_observed": obs, "z_null_mean": float(values.mean()) if len(values) else np.nan,
                     "p_spec": p, "size": len(genes)})
    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "geneseeded_z.parquet", index=False)
    payload = {"fold": args.fold, "gene_weight": args.gene_weight, "draws": args.draws,
               "diseases": int(len(table)),
               "z_median": round(float(table.z_observed.median()), 3),
               "z_null_median": round(float(table.z_null_mean.median()), 3),
               "frac_p_le_001": round(float((table.p_spec <= 0.01).mean()), 3),
               "frac_p_le_005": round(float((table.p_spec <= 0.05).mean()), 3),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "geneseeded_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
