"""交叉拟合的 z_spec：用一半表型建模块，另一半表型评价。

对每个疾病把 HPO 术语按 IC 排序后交替分成 A/B 两份：
  建模块：只用 A 的种子跑 RWR → top-200 → Leiden → 最大模块（可在 A 上精修）
  评价  ：用 B 的种子重跑 RWR，算模块在 B 分位上的 z_spec（背景仍是 20 个异家族疾病）
这样模块没见过评价用的术语，z_spec 从「样本内拟合」变成「样本外验证」。
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
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402

BACKDROP = 20
SIZE = 25
STEPS = 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "crossfit.log", "65_crossfit")
    started = time.time()
    out_dir = paths.out("crossfit", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_pos = {g: i for i, g in enumerate(gene_ids)}
    hpo_rows = node_map.loc[node_map["node_type"].eq("phenotype")]
    hpo_index = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    n_nodes = int(node_map["node_index"].max()) + 1

    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    no_freq = "frequency_weight" not in phenotypes.columns
    if no_freq:
        phenotypes["frequency_weight"] = 1.0
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(lambda s: sorted(set(s))).to_dict()
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

    halves: dict[str, list[list[str]]] = {}
    for disease in test_ids:
        terms = sorted(terms_of.get(disease, []), key=lambda t: -ic_lookup.get(t, 0.0))
        if len(terms) < 4:
            continue
        halves[disease] = [terms[0::2], terms[1::2]]
    test_ids = [d for d in test_ids if d in halves]

    def seeds_from(part: int):
        seeds = np.zeros((n_nodes, len(test_ids)))
        for column, disease in enumerate(test_ids):
            indices = [hpo_index[t] for t in halves[disease][part] if t in hpo_index]
            if not indices:
                continue
            unique = np.unique(indices)
            seeds[unique, column] = 1.0 / len(unique)
        return seeds

    def percentiles_of(scores: np.ndarray) -> np.ndarray:
        out = np.zeros_like(scores, dtype=float)
        for column in range(scores.shape[1]):
            out[:, column] = rankdata(scores[:, column]) / scores.shape[0]
        return out

    scores = {}
    for part, name in ((0, "A"), (1, "B")):
        result = v0_rwr.run_rwr(transition, seeds_from(part), alpha=alpha)
        scores[name] = percentiles_of(result.scores[gene_rows["node_index"].to_numpy(dtype=int), :])
        log.info("rwr_done", half=name)

    backgrounds = {}
    for disease in test_ids:
        scored = []
        for other in test_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = set(terms_of.get(disease, [])) & set(terms_of.get(other, []))
            if not shared:
                scored.append((0.0, other))
                continue
            sim = sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, [])))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, []))) + 1e-9)
            scored.append((sim, other))
        pool = [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]
        backgrounds[disease] = pool

    def z_on(half: str, disease: str, genes: set[str]) -> float:
        cols = [gene_pos[g] for g in genes if g in gene_pos]
        if len(cols) < 5:
            return np.nan
        column = test_ids.index(disease)
        own = float(scores[half][cols, column].mean())
        back = []
        for other in backgrounds[disease]:
            other_col = test_ids.index(other)
            back.append(float(scores[half][cols, other_col].mean()))
        back = np.array(back)
        if back.size < 2 or back.std(ddof=1) == 0:
            return np.nan
        return float((own - back.mean()) / back.std(ddof=1))

    rows = []
    for column, disease in enumerate(test_ids):
        ranking = np.argsort(-scores["A"][:, column])[:200]
        candidate = [gene_ids[i] for i in ranking]
        detections = detect(tuple(candidate), ppi, seed=stable_seed(7, args.fold, disease, "crossfit"))
        if not detections:
            continue
        module = set(max(detections, key=len))
        module = set(sorted(module, key=lambda g: -scores["A"][gene_pos[g], column])[:SIZE])
        if len(module) < 5:
            continue
        z_a_start = z_on("A", disease, module)

        current = set(module)
        best = z_a_start
        for _ in range(args.steps):
            neighbours = set()
            for gene in current:
                neighbours.update(adjacency.get(gene, {}).keys())
            pool = (set(candidate) & neighbours) - current
            if not pool:
                break
            weakest = sorted(current, key=lambda g: scores["A"][gene_pos[g], column])[:5]
            best_move, best_value = None, best
            for out_gene in weakest:
                for in_gene in sorted(pool, key=lambda g: -scores["A"][gene_pos[g], column])[:10]:
                    trial = (current - {out_gene}) | {in_gene}
                    value = z_on("A", disease, trial)
                    if value > best_value:
                        best_move, best_value = (out_gene, in_gene), value
            if best_move is None:
                break
            current = (current - {best_move[0]}) | {best_move[1]}
            best = best_value

        rows.append({"disease_id": disease,
                     "z_A_in_sample_base": z_a_start,
                     "z_A_in_sample_refined": best,
                     "z_B_heldout_base": z_on("B", disease, module),
                     "z_B_heldout_refined": z_on("B", disease, current),
                     "size": len(current)})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "crossfit.parquet", index=False)
    payload = {"fold": args.fold, "diseases": int(len(table)), "steps": args.steps,
               "in_sample_base": round(float(table.z_A_in_sample_base.median()), 3),
               "in_sample_refined": round(float(table.z_A_in_sample_refined.median()), 3),
               "heldout_base": round(float(table.z_B_heldout_base.median()), 3),
               "heldout_refined": round(float(table.z_B_heldout_refined.median()), 3),
               "heldout_wins": int((table.z_B_heldout_refined > table.z_B_heldout_base).sum()),
               "heldout_p": float(wilcoxon(table.z_B_heldout_refined, table.z_B_heldout_base).pvalue),
               "seconds": round(time.time() - started, 2)}
    (out_dir / "crossfit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
