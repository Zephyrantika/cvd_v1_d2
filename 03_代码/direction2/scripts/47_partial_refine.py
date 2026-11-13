"""改进方案：用部分表型查询做模块精修，在完整查询上评价。

思路（对应文献里「扰动一致性」与「稳健模块」两条线）：
1. 把疾病的 HPO 术语按 70/30 拆成 query-train 与 query-eval；
2. 只用 query-train 重跑 RWR，得到 train 分数，取 top-200、Leiden；
3. 对模块做贪心精修：在候选集内加/减基因，使 train 分数下的 z_spec + 密度最大化；
4. 用完整表型查询的原始分数评价 z_spec（query-eval 与其背景都没参与构建）。

基线：用完整分数跑 Leiden，取最大模块前 25，同样用完整分数评价。
如果精修版在完整评价上超过基线，说明「部分查询 + 精修」是有效的模块发现方法。
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25
STEPS = 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "partial_refine.log", "47_partial_refine")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

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

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                        node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
    from scipy.sparse import load_npz
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_pos = {g: i for i, g in enumerate(gene_ids)}

    def z_full(disease: str, genes: list[str]) -> float:
        genes = [g for g in genes if g in position]
        if not genes:
            return np.nan
        cols = [position[g] for g in genes]
        background = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            background.append((sim, other))
        background = [b for _, b in sorted(background, reverse=True)[:20]]
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in background])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    # 基线：完整分数 + Leiden 最大模块前 25
    modules = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    largest = modules.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    baseline = {}
    for disease, module_id in largest.items():
        genes = list(members.loc[members.module_id == module_id].sort_values("r_d", ascending=False)["gene_id"])[:SIZE]
        if len(genes) >= 5:
            baseline[disease] = z_full(disease, genes)

    rows = []
    for disease in matrix.disease_ids:
        terms = sorted(terms_of.get(disease, set()))
        if len(terms) < 5 or disease not in baseline:
            continue
        order = rng.permutation(len(terms))
        cut = max(1, int(len(terms) * 0.7))
        train_terms = [terms[i] for i in order[:cut]]
        indices = [hpo_node[t] for t in train_terms if t in hpo_node]
        if not indices:
            continue
        seeds = np.zeros((transition.shape[0], 1))
        unique = np.unique(indices)
        seeds[unique, 0] = 1.0 / len(unique)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        train_scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
        order_genes = np.argsort(-train_scores, kind="stable")[:200]
        candidate = [gene_ids[i] for i in order_genes]
        train_rank = np.argsort(-train_scores, kind="stable")
        train_pct = np.empty(len(train_scores))
        train_pct[train_rank] = 1.0 - np.arange(len(train_scores)) / (len(train_scores) - 1)

        def proxy(genes: list[str]) -> float:
            values = [train_pct[gene_pos[g]] for g in genes if g in gene_pos]
            if not values:
                return -1.0
            member = set(genes)
            internal = sum(1 for g in member for n in adjacency.get(g, {}) if n in member)
            possible = len(member) * (len(member) - 1) / 2
            density = (internal / 2) / possible if possible else 0.0
            return float(np.mean(values)) + 0.1 * density

        detections = detect(tuple(candidate), ppi, seed=stable_seed(args.seed, args.fold, disease, "partial"))
        if not detections:
            continue
        module = set(max(detections, key=len))
        if len(module) > SIZE:
            ranked = sorted(module, key=lambda g: -train_scores[gene_pos[g]])
            module = set(ranked[:SIZE])

        # 贪心精修：在候选集内做加/减
        for _ in range(STEPS):
            current = proxy(list(module))
            best_gain, best_move = 0.0, None
            neighbours = set()
            for gene in module:
                neighbours.update(adjacency.get(gene, {}).keys())
            for gene in neighbours - module:
                gain = proxy(list(module | {gene})) - current
                if gain > best_gain:
                    best_gain, best_move = gain, ("add", gene)
            for gene in module:
                if len(module) <= 5:
                    break
                gain = proxy(list(module - {gene})) - current
                if gain > best_gain:
                    best_gain, best_move = gain, ("remove", gene)
            if best_move is None:
                break
            module = module | {best_move[1]} if best_move[0] == "add" else module - {best_move[1]}

        rows.append({"disease_id": disease, "z_refined": z_full(disease, list(module)),
                     "z_baseline": baseline[disease]})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "partial_refine.parquet", index=False)
    paired = table.dropna()
    summary = {
        "fold": args.fold,
        "diseases": int(len(paired)),
        "z_refined_median": round(float(paired.z_refined.median()), 3),
        "z_baseline_median": round(float(paired.z_baseline.median()), 3),
        "wins": int((paired.z_refined > paired.z_baseline).sum()),
        "p": float(wilcoxon(paired.z_refined, paired.z_baseline).pvalue),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "partial_refine.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
