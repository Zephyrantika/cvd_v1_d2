"""按 p_spec 选模块（修正版）：对每个候选模块，用同一批随机表型零模型算它自己的 p，
再在「复现率≥0.7」的候选里取 p 最小（并列取 z 大）。

与 79_pareto_fold.py 的区别：不用原始 z 跨基因集比较，改成每个模块相对自身零分布的位置，
与 A 级判据（p_spec≤0.01）同一量纲。
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
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20

CANDIDATE_FOLDERS = {
    "lam0.5": "fold_{f}_lam0.5_mu1.0",
    "lam1.0": "fold_{f}_lam1.0_mu1.0",
    "size15": "fold_{f}_lam1.0_mu1.0_size15",
    "size20": "fold_{f}_lam1.0_mu1.0_size20",
    "size30": "fold_{f}_lam1.0_mu1.0_size30",
    "size40": "fold_{f}_lam1.0_mu1.0_size40",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "select_by_p.log", "81_select_by_p")
    started = time.time()
    out_dir = paths.out("pareto", f"byp_fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles().T   # 转成 (基因 × 疾病)，与零模型的矩阵方向一致
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    hpo_rows = node_map.loc[node_map["node_type"].eq("phenotype")]
    hpo_node = {str(t): int(i) for t, i in zip(hpo_rows["node_id"], hpo_rows["node_index"])}
    all_terms = list(hpo_node)
    n_nodes = int(node_map["node_index"].max()) + 1
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))

    # ---- 候选 ----
    pools: dict[str, list[dict]] = {}
    base_mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    base_mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    ref_rows = pd.read_parquet(paths.out("lam_refine", f"fold_{args.fold}_lam1.0_mu1.0", "lam_refine.parquet"))
    base_rep = ref_rows[ref_rows.variant == "baseline"].set_index("disease_id")["replication"].to_dict()
    for disease, module_id in largest.items():
        genes = set(list(base_sets.get(module_id, set()))[:25])
        if len(genes) < 5:
            continue
        pools.setdefault(disease, []).append({"source": "baseline", "genes": genes,
                                              "replication": float(base_rep.get(disease, 0.0))})
    for tag, pattern in CANDIDATE_FOLDERS.items():
        folder = paths.out("lam_refine", pattern.format(f=args.fold))
        if not (folder / "lam_refine_members.parquet").exists():
            continue
        table = pd.read_parquet(folder / "lam_refine.parquet")
        table = table[table.variant == "refined"]
        mem = pd.read_parquet(folder / "lam_refine_members.parquet")
        mem["gene_id"] = mem["gene_id"].astype(str)
        sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()
        for row in table.itertuples(index=False):
            genes = sets.get(row.disease_id, set())
            if len(genes) < 5:
                continue
            pools.setdefault(row.disease_id, []).append({"source": tag, "genes": genes,
                                                         "replication": float(row.replication)})

    diseases = [d for d in matrix.disease_ids if d in pools]
    column = {d: i for i, d in enumerate(diseases)}

    def backdrop(disease: str) -> list[int]:
        scored = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [column[d] for _, d in sorted(scored, reverse=True)[:BACKDROP] if d in column]

    backdrops = {d: backdrop(d) for d in diseases}
    candidates = [(d, c) for d in diseases for c in pools[d]]
    genes_idx = {id(c): [position[g] for g in c["genes"] if g in position] for _, c in candidates}
    log.info("candidates", n=len(candidates))

    def z_from(pct: np.ndarray, disease: str, cols: list[int]) -> float:
        if len(cols) < 5:
            return np.nan
        own_col = column[disease]
        own = float(pct[cols, own_col].mean())
        back = [float(pct[cols, b].mean()) for b in backdrops[disease]]
        if len(back) < 2 or np.std(back, ddof=1) == 0:
            return np.nan
        return float((own - np.mean(back)) / np.std(back, ddof=1))

    observed = np.array([z_from(percentiles, d, genes_idx[id(c)]) for d, c in candidates])
    nulls = np.full((len(candidates), args.draws), np.nan)
    for draw in range(args.draws):
        seeds = np.zeros((n_nodes, len(diseases)))
        for d in diseases:
            size = len(terms_of.get(d, set()))
            sample = rng.choice(all_terms, size=min(size, len(all_terms)), replace=False)
            indices = [hpo_node[t] for t in sample if t in hpo_node]
            if not indices:
                continue
            unique = np.unique(indices)
            seeds[unique, column[d]] = 1.0 / len(unique)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        pct = np.zeros_like(raw)
        for i in range(raw.shape[1]):
            pct[:, i] = rankdata(raw[:, i]) / raw.shape[0]
        for k, (d, c) in enumerate(candidates):
            nulls[k, draw] = z_from(pct, d, genes_idx[id(c)])
        if draw % 10 == 0:
            log.info("draw_done", draw=draw)

    rows = []
    for k, (d, c) in enumerate(candidates):
        values = nulls[k][~np.isnan(nulls[k])]
        p = (1 + int((values >= observed[k]).sum())) / (len(values) + 1) if len(values) else np.nan
        rows.append({"disease_id": d, "source": c["source"], "z": observed[k], "p_spec": p,
                     "replication": c["replication"],
                     "null_mean": float(np.mean(values)) if len(values) else np.nan})
    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "candidates_byp.parquet", index=False)

    selected, members = [], []
    for disease, group in frame.groupby("disease_id"):
        eligible = group[group.replication >= 0.7]
        pool = eligible if len(eligible) else group
        # p 越小越好；并列时 z 越大越好
        best = pool.sort_values(["p_spec", "z"], ascending=[True, False]).iloc[0]
        selected.append({"disease_id": disease, "source": best.source, "z": float(best.z),
                         "p_spec": float(best.p_spec), "replication": float(best.replication)})
        genes = next(c["genes"] for d2, c in candidates if d2 == disease and c["source"] == best.source)
        for gene in genes:
            members.append({"module_id": f"{disease}:byp", "disease_id": disease, "gene_id": gene})
    chosen = pd.DataFrame(selected)
    chosen.to_parquet(out_dir / "selected_byp.parquet", index=False)
    pd.DataFrame(members).to_parquet(out_dir / "selected_byp_members.parquet", index=False)

    summary = {"fold": args.fold, "diseases": int(len(chosen)), "draws": args.draws,
               "source_counts": chosen.source.value_counts().to_dict(),
               "frac_p_le_001": round(float((chosen.p_spec <= 0.01).mean()), 3),
               "frac_rep_ge_0.7": round(float((chosen.replication >= 0.7).mean()), 3),
               "z_median": round(float(chosen.z.median()), 3),
               "p_median": round(float(chosen.p_spec.median()), 4),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
