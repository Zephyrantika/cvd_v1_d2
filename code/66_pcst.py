"""用 PCST 精确求模块：prize 取「本病分位 − 背景疾病同基因均分位」，连接基因当 Steiner。

与 64 号（MWCS）的区别：不要求所有终端直接连通，允许低分节点作为连接代价，规模只数终端。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pcst_ilp import solve_pcst  # noqa: E402

SIZE = 25
BACKDROP = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "pcst.log", "66_pcst")
    started = time.time()
    out_dir = paths.out("pcst", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    degree: dict[str, float] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
        degree[a] = degree.get(a, 0.0) + float(w)
        degree[b] = degree.get(b, 0.0) + float(w)
    degree_total = sum(degree.values()) or 1.0
    degree_share = {g: v / degree_total for g, v in degree.items()}

    cons_mem = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"))
    cons_mem["gene_id"] = cons_mem["gene_id"].astype(str)
    cons_mods = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet"))
    cons_largest = cons_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    frequency = {(r.module_id, str(r.gene_id)): float(r.frequency) for r in cons_mem.itertuples(index=False)}

    base_mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    base_mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()

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

    def background(disease: str) -> list[str]:
        scored = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_of(disease: str, genes: set[str]) -> float:
        cols = [position[g] for g in genes if g in position]
        if len(cols) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in background(disease)])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    rows, member_rows = [], []
    diseases = sorted(matrix.disease_ids)
    if args.limit:
        diseases = diseases[: args.limit]
    for disease in diseases:
        base_id = largest.get(disease)
        if not base_id:
            continue
        candidate = list(matrix.top_k(disease, 200))
        index = {g: i for i, g in enumerate(candidate)}
        edges = []
        for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float), strict=True):
            ia, ib = index.get(a), index.get(b)
            if ia is not None and ib is not None and ia < ib:
                edges.append((ia, ib, float(w)))
        cols_idx = [position[g] for g in candidate]
        own = percentiles[matrix.disease_index(disease)][cols_idx]
        back_matrix = np.vstack([percentiles[matrix.disease_index(b)][cols_idx] for b in background(disease)])
        back_mean = back_matrix.mean(axis=0)
        back_sd = back_matrix.std(axis=0, ddof=1) + 1e-6
        base_prizes = (own - back_mean) / back_sd     # 每个基因相对背景的 z
        freq_term = np.array([frequency.get((cons_largest.get(disease), g), 0.0) for g in candidate])
        hub_term = np.array([degree_share.get(g, 0.0) for g in candidate])
        prizes = base_prizes + args.lam * freq_term - args.mu * hub_term
        root = int(np.argmax(prizes))
        selected = solve_pcst(candidate, edges, prizes, SIZE, root)
        genes = {candidate[i] for i in selected} | {candidate[root]}
        if len(genes) < 5:
            continue
        rows.append({"disease_id": disease, "z_pcst": z_of(disease, genes),
                     "z_baseline": z_of(disease, set(list(base_sets.get(base_id, set()))[:SIZE])),
                     "size": len(genes), "terminals": len(selected) + 1})
        for gene in genes:
            member_rows.append({"module_id": f"{disease}:pcst", "disease_id": disease, "gene_id": gene})
        log.info("disease_done", disease=disease, size=len(genes))

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "pcst.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out_dir / "pcst_members.parquet", index=False)
    payload = {"fold": args.fold, "diseases": int(len(table)),
               "z_baseline": round(float(table.z_baseline.median()), 3),
               "z_pcst": round(float(table.z_pcst.median()), 3),
               "wins": int((table.z_pcst > table.z_baseline).sum()),
               "p": float(wilcoxon(table.z_pcst, table.z_baseline).pvalue) if len(table) > 5 else None,
               "size_mean": round(float(table["size"].mean()), 1),
               "seconds": round(time.time() - started, 2)}
    (out_dir / "pcst.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
