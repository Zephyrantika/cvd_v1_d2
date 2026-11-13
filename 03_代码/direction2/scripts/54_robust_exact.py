"""ROBUST 式枚举（精确子问题版）：每次用扰动后的 prize 精确求一个 PCST/MWCS，再做多样性筛选。

pcst_fast 在本机装不上（网络停滞），所以用 53 号脚本的精确 ILP 当子问题求解器：
PCST 与最大权重连通子图在固定根、线性 prize/cost 下等价，因此每次枚举是精确解，
差别只在「多样性枚举」这一层是自定的（扰动 prize、换根、要求与已选解 Jaccard < 0.85）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _exact_ilp import solve_mwcs  # noqa: E402

SIZE = 25
BACKDROP = 20
RUNS = 12
DIVERSITY = 0.85


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "robust_exact.log", "54_robust_exact")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

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

    fair = pd.read_parquet(paths.out("methods", f"fold_{args.fold}", "alternatives.parquet"))
    base_lookup = dict(zip(fair["disease_id"], fair["z_baseline"]))

    def z_of(disease: str, genes: list[str]) -> float:
        cols = [position[g] for g in genes if g in position]
        if not cols:
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
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in bg])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    rows = []
    diseases = sorted(matrix.disease_ids)
    if args.limit:
        diseases = diseases[: args.limit]
    for disease in diseases:
        candidate = list(matrix.top_k(disease, 200))
        index = {g: i for i, g in enumerate(candidate)}
        edges = []
        for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str),
                           ppi["weight"].astype(float), strict=True):
            ia, ib = index.get(a), index.get(b)
            if ia is not None and ib is not None and ia < ib:
                edges.append((ia, ib, float(w)))
        base_prize = percentiles[matrix.disease_index(disease)][[position[g] for g in candidate]]

        solutions: list[set[int]] = []
        for _ in range(RUNS):
            jittered = base_prize * rng.lognormal(0.0, 0.25, size=len(candidate))
            jittered = jittered / (jittered.max() + 1e-12)
            top = np.argsort(-jittered)[: max(5, len(candidate) // 10)]
            root = int(rng.choice(top))
            selected = set(solve_mwcs(candidate, edges, jittered, SIZE, root))
            if not selected:
                continue
            if all(len(selected & s) / len(selected | s) < DIVERSITY for s in solutions):
                solutions.append(selected)

        frequency: dict[int, int] = {}
        for solution in solutions:
            for node in solution:
                frequency[node] = frequency.get(node, 0) + 1
        ranked = [node for node, _ in sorted(frequency.items(), key=lambda kv: -kv[1])][:SIZE]
        genes = [candidate[i] for i in ranked]
        rows.append({"disease_id": disease, "z_robust_exact": z_of(disease, genes),
                     "size": len(genes), "solutions": len(solutions),
                     "z_baseline": base_lookup.get(disease)})
        log.info("disease_done", disease=disease, solutions=len(solutions))

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "robust_exact.parquet", index=False)
    from scipy.stats import wilcoxon
    paired = table.dropna(subset=["z_robust_exact", "z_baseline"])
    payload = {"fold": args.fold, "diseases": int(len(paired)),
               "z_median": round(float(paired.z_robust_exact.median()), 3),
               "z_baseline_median": round(float(paired.z_baseline.median()), 3),
               "wins": int((paired.z_robust_exact > paired.z_baseline).sum()),
               "p": float(wilcoxon(paired.z_robust_exact, paired.z_baseline).pvalue) if len(paired) > 5 else None,
               "solutions_mean": round(float(table["solutions"].mean()), 1),
               "seconds": round(time.time() - started, 2)}
    (out_dir / "robust_exact.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
