"""按 p 选模块（批量版）：每次 RWR 同时跑 10 个随机表型抽样（91 病 × 10 抽样 = 910 列），
把 100 次抽样压成 10 次调用，避免单进程长时间运行导致的卡死。

输出与 81 相同：outputs/pareto/byp_fold{N}/selected_byp*.parquet
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
BATCH = 10

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
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "select_by_p_batched.log", "83_select_by_p_batched")
    started = time.time()
    out_dir = paths.out("pareto", f"byp_fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles().T     # (基因 × 疾病)
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
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
    backdrops = {}
    for disease in diseases:
        scored = []
        for other in diseases:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        backdrops[disease] = [column[d] for _, d in sorted(scored, reverse=True)[:BACKDROP] if d in column]

    candidates = [(d, c) for d in diseases for c in pools[d]]
    genes_idx = [np.array([position[g] for g in c["genes"] if g in position], dtype=int) for _, c in candidates]
    own_col = np.array([column[d] for d, _ in candidates])
    back_idx = [backdrops[d] for d, _ in candidates]
    nd = len(diseases)
    log.info("candidates", n=len(candidates), diseases=nd)

    def z_vec(pct: np.ndarray, k: int, col: int) -> float:
        idx = genes_idx[k]
        if len(idx) < 5:
            return np.nan
        own = float(pct[idx, col].mean())
        back = [float(pct[idx, b].mean()) for b in back_idx[k]]
        if len(back) < 2 or np.std(back, ddof=1) == 0:
            return np.nan
        return float((own - np.mean(back)) / np.std(back, ddof=1))

    observed = np.array([z_vec(percentiles, k, own_col[k]) for k in range(len(candidates))])
    nulls = np.full((len(candidates), args.draws), np.nan)

    draw = 0
    while draw < args.draws:
        count = min(args.batch, args.draws - draw)
        seeds = np.zeros((n_nodes, nd * count))
        for b in range(count):
            for d in diseases:
                size = len(terms_of.get(d, set()))
                sample = rng.choice(all_terms, size=min(size, len(all_terms)), replace=False)
                indices = [hpo_node[t] for t in sample if t in hpo_node]
                if not indices:
                    continue
                unique = np.unique(indices)
                seeds[unique, b * nd + column[d]] = 1.0 / len(unique)
        result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
        raw = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
        for b in range(count):
            pct = np.zeros_like(raw[:, :nd])
            for i in range(nd):
                col = b * nd + i
                pct[:, i] = rankdata(raw[:, col]) / raw.shape[0]
            for k in range(len(candidates)):
                nulls[k, draw + b] = z_vec(pct, k, own_col[k])
        draw += count
        log.info("batch_done", draw=draw)
        if draw % 20 == 0:
            print(f"[info] draw {draw}/{args.draws}", flush=True)

    rows = []
    for k, (d, c) in enumerate(candidates):
        values = nulls[k][~np.isnan(nulls[k])]
        p = (1 + int((values >= observed[k]).sum())) / (len(values) + 1) if len(values) else np.nan
        rows.append({"disease_id": d, "source": c["source"], "z": float(observed[k]), "p_spec": p,
                     "replication": c["replication"]})
    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "candidates_byp.parquet", index=False)

    selected, members = [], []
    for disease, group in frame.groupby("disease_id"):
        eligible = group[group.replication >= 0.7]
        pool = eligible if len(eligible) else group
        best = pool.sort_values(["p_spec", "z"], ascending=[True, False]).iloc[0]
        selected.append({"disease_id": disease, "source": best.source, "z": float(best.z),
                         "p_spec": float(best.p_spec), "replication": float(best.replication)})
        genes = next(c["genes"] for d2, c in candidates if d2 == disease and c["source"] == best.source)
        for gene in genes:
            members.append({"module_id": f"{disease}:byp", "disease_id": disease, "gene_id": gene})
    chosen = pd.DataFrame(selected)
    chosen.to_parquet(out_dir / "selected_byp.parquet", index=False)
    pd.DataFrame(members).to_parquet(out_dir / "selected_byp_members.parquet", index=False)

    summary = {"fold": args.fold, "diseases": int(len(chosen)), "draws": args.draws, "batch": args.batch,
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
