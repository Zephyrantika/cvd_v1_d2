"""精确版最大权重连通子图（Heinz / DOMINO 的共同数学形式），用 scipy 的 HiGHS 求解。

之前的 Heinz 是贪心替代；这里写成标准 ILP：
    变量 x_v（节点选否）、y_e（边选否）、f_uv（从根出发的流量）
    目标 max Σ p_v x_v − Σ c_e y_e
    约束 流量守恒保证连通、y 与 x 一致、Σ x ≤ 25
规模：候选集 200 个节点、约 1–2 千条边，HiGHS 可秒级求解。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25
BACKDROP = 20


def solve_mwcs(nodes: list[str], edges: list[tuple[int, int, float]], prizes: np.ndarray,
               max_size: int, root: int) -> list[int]:
    n = len(nodes)
    m = len(edges)
    # 变量顺序：x(0..n-1), y(n..n+m-1), f(n+m..n+m+m-1)（每条边一个有向流量）
    n_var = n + m + m
    cost = np.zeros(n_var)
    cost[:n] = -prizes           # 最大化 prize → 最小化负值
    for index, (_, _, weight) in enumerate(edges):
        cost[n + index] = -np.log(max(weight, 1e-6) / 0.9)  # 边成本（越小越好）

    rows, cols, values, lower, upper = [], [], [], [], []
    row = 0

    # y_e ≤ x_u, y_e ≤ x_v
    for index, (u, v, _) in enumerate(edges):
        rows += [row, row]; cols += [n + index, u]; values += [1.0, -1.0]; lower.append(-np.inf); upper.append(0.0); row += 1
        rows += [row, row]; cols += [n + index, v]; values += [1.0, -1.0]; lower.append(-np.inf); upper.append(0.0); row += 1

    # 流量守恒：Σ_in f − Σ_out f = x_v（v≠root）；root 处 = −(Σx)
    incidence: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for index, (u, v, _) in enumerate(edges):
        incidence[u].append((index, -1.0))   # 流出
        incidence[v].append((index, 1.0))    # 流入
    for node in range(n):
        if node == root:
            continue
        for index, sign in incidence[node]:
            rows.append(row); cols.append(n + m + index); values.append(sign)
        rows.append(row); cols.append(node); values.append(-1.0)
        lower.append(0.0); upper.append(0.0); row += 1

    # f_uv ≤ max_size * y_e
    for index in range(m):
        rows += [row, row]; cols += [n + m + index, n + index]; values += [1.0, -float(max_size)]
        lower.append(-np.inf); upper.append(0.0); row += 1

    # Σ x ≤ max_size
    for node in range(n):
        rows.append(row); cols.append(node); values.append(1.0)
    lower.append(-np.inf); upper.append(float(max_size)); row += 1

    matrix = csr_matrix((values, (rows, cols)), shape=(row, n_var))
    constraint = LinearConstraint(matrix, np.array(lower), np.array(upper))
    integrality = np.zeros(n_var)
    integrality[: n + m] = 1          # x, y 为 0/1
    bounds = Bounds(np.zeros(n_var), np.concatenate([np.ones(n + m), np.full(m, float(max_size))]))
    result = milp(cost, constraints=[constraint], integrality=integrality, bounds=bounds,
                  options={"time_limit": 20, "mip_rel_gap": 0.01})
    if result.x is None:
        return []
    selected = [i for i in range(n) if result.x[i] > 0.5]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "exact_mwcs.log", "53_exact_mwcs")
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

    baseline = pd.read_parquet(paths.out("methods", f"fold_{args.fold}", "e6_size_matched.parquet"))
    base_lookup = dict(zip(baseline["disease_id"], baseline["z_leiden"]))

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
        prizes = percentiles[matrix.disease_index(disease)][[position[g] for g in candidate]]
        root = int(np.argmax(prizes))
        selected = solve_mwcs(candidate, edges, prizes, SIZE, root)
        genes = [candidate[i] for i in selected]
        rows.append({"disease_id": disease, "z_exact": z_of(disease, genes), "size": len(genes),
                     "z_baseline": base_lookup.get(disease)})
        log.info("disease_done", disease=disease, size=len(genes))

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "exact_mwcs.parquet", index=False)
    from scipy.stats import wilcoxon
    summary = {
        "fold": args.fold, "diseases": int(len(table)),
        "size_mean": round(float(table["size"].mean()), 1),
        "z_exact_median": round(float(table.z_exact.median()), 3),
        "z_baseline_median": round(float(table.z_baseline.median()), 3),
        "exact_wins": int((table.z_exact > table.z_baseline).sum()),
        "p": float(wilcoxon(table.z_exact, table.z_baseline).pvalue) if len(table) > 5 else None,
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "exact_mwcs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
