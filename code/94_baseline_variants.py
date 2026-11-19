"""两个公平性对照：

1) 公平基线：基线最大模块在 15/20/25/30/40 五个规模里按同一规则选点（复现率≥0.7 取 z 最大），
   看它能否也达到 R1 的水平——检验 R1 的优势是否只是"多了选点自由度"。
2) 不截断基线：用完整最大模块（不截到 25）重算，看基线 A 级是否显著上升。

一次扰动划分（20 次）复用给所有变体，复现率按变体模块分别计算。
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
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

COVER = 0.6
BACKDROP = 20
SIZES = [15, 20, 25, 30, 40, None]  # None = 完整模块


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "baseline_variants.log", "94_baseline_variants")
    started = time.time()
    out_dir = paths.out("fair_baseline", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    msets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_index = {str(t): int(i) for t, i in zip(node_map.loc[node_map["node_type"] == "phenotype", "node_id"],
                                                node_map.loc[node_map["node_type"] == "phenotype", "node_index"])}
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
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
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

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
        idx = [position[g] for g in genes if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), idx].mean())
        back = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in background(disease)])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    # 一次扰动划分
    partitions: dict[str, list[list[set[str]]]] = {}
    diseases = [d for d in matrix.disease_ids if d in largest]
    for disease in diseases:
        terms = terms_of.get(disease, [])
        parts = []
        for run in range(args.runs):
            kept = [t for t in terms if rng.random() > 0.10]
            indices = [hpo_index[t] for t in kept if t in hpo_index]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[:200]
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(args.seed, args.fold, disease, run))
            parts.append([set(m) for m in detections])
        partitions[disease] = parts
        log.info("partitions", disease=disease)

    def replication(disease: str, module: set[str]) -> float:
        parts = partitions.get(disease, [])
        if not parts:
            return np.nan
        covered = sum(1 for p in parts if any(len(module & m) / len(module) >= COVER for m in p))
        return covered / len(parts)

    rows, members = [], []
    for disease in diseases:
        full = set(msets.get(largest[disease], set()))
        if len(full) < 5:
            continue
        for size in SIZES:
            module = set(list(full)[: size]) if size else full
            if len(module) < 5:
                continue
            rows.append({"disease_id": disease, "size": size if size else len(module),
                         "variant": f"size{size}" if size else "full",
                         "z": z_of(disease, module), "replication": replication(disease, module)})
        # 保存完整模块成员
        for gene in full:
            members.append({"module_id": f"{disease}:full", "disease_id": disease, "gene_id": gene})
        for size in [15, 20, 25, 30, 40]:
            for gene in set(list(full)[:size]):
                members.append({"module_id": f"{disease}:size{size}", "disease_id": disease, "gene_id": gene})

    table = pd.DataFrame(rows).dropna()
    table.to_parquet(out_dir / "variants.parquet", index=False)
    pd.DataFrame(members).to_parquet(out_dir / "variant_members.parquet", index=False)

    # 公平基线：rep>=0.7 里取 z 最大
    picked = []
    for disease, group in table[table.variant != "full"].groupby("disease_id"):
        eligible = group[group.replication >= 0.7]
        pool = eligible if len(eligible) else group
        best = pool.sort_values("z", ascending=False).iloc[0]
        picked.append({"disease_id": disease, "variant": best.variant, "size": int(best["size"]),
                       "z": float(best.z), "replication": float(best.replication)})
    fair = pd.DataFrame(picked)
    fair.to_parquet(out_dir / "fair_selected.parquet", index=False)
    fair_members = []
    for row in fair.itertuples(index=False):
        for gene in set(list(msets.get(largest[row.disease_id], set()))[: row.size]):
            fair_members.append({"module_id": f"{row.disease_id}:fair", "disease_id": row.disease_id,
                                 "gene_id": gene})
    pd.DataFrame(fair_members).to_parquet(out_dir / "fair_members.parquet", index=False)

    full_table = table[table.variant == "full"][["disease_id", "z", "replication"]]
    full_table.to_parquet(out_dir / "full_selected.parquet", index=False)

    summary = {"fold": args.fold, "diseases": int(len(fair)),
               "fair_source": fair.variant.value_counts().to_dict(),
               "fair_z_median": round(float(fair.z.median()), 3),
               "fair_rep_mean": round(float(fair.replication.mean()), 3),
               "full_z_median": round(float(full_table.z.median()), 3),
               "full_rep_mean": round(float(full_table.replication.mean()), 3),
               "seconds": round(time.time() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
