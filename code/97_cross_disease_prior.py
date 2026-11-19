"""方案三：用「其他疾病的标签」做先验，重排候选基因。

对疾病 d：
  prior(g) = 除 d 及其同家族疾病之外，有多少个心血管疾病把 g 标为已知基因
  先验分位 = prior 在全部候选基因上的百分位
  hybrid = 分位(RWR) + γ · 先验分位

同家族排除是关键：否则会拿相似疾病的标签去预测目标疾病，构成泄漏。
评价：已知基因在 top-200 的比例、模块召回率；同时报告 z_spec 的代价。
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
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20
SIZE = 25
CURATED = {"original", "clinvar", "clingen", "disgenet"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gammas", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--labels", default="curated", choices=["curated", "all"])
    parser.add_argument("--module-rule", default="largest", choices=["largest", "prior"])
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    started = time.time()
    out_dir = paths.out("prior", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    genes = list(matrix.gene_ids)
    position = {g: i for i, g in enumerate(genes)}

    labels = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/chain/fact_disease_gene.parquet")
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["gene_id"] = labels["gene_id"].astype(str)
    if args.labels == "curated":
        labels = labels[labels["source"].isin(CURATED)]
    labels = labels.drop_duplicates(["disease_id", "gene_id"])
    label_sets = labels.groupby("disease_id")["gene_id"].apply(set).to_dict()

    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))

    base_mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    base_mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()

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

    def z_of(disease: str, genes_set: set[str]) -> float:
        idx = [position[g] for g in genes_set if g in position]
        if len(idx) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), idx].mean())
        back = np.array([percentiles[matrix.disease_index(b), idx].mean() for b in background(disease)])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    def prior_vector(disease: str) -> np.ndarray:
        counts = np.zeros(len(genes))
        for other, gs in label_sets.items():
            if other == disease or family.get(other) == family.get(disease):
                continue
            for g in gs:
                i = position.get(g)
                if i is not None:
                    counts[i] += 1.0
        return rankdata(counts) / len(counts)

    rows = []
    for gamma in [float(x) for x in args.gammas.split(",")]:
        for disease in matrix.disease_ids:
            row = percentiles[matrix.disease_index(disease)]
            if gamma > 0:
                hybrid = row + gamma * prior_vector(disease)
            else:
                hybrid = row
            order = np.argsort(-hybrid)[:200]
            candidate = tuple(genes[i] for i in order)
            detections = detect(candidate, ppi, seed=stable_seed(97, args.fold, disease, "prior"))
            if detections and args.module_rule == "prior":
                prior_vec = prior_vector(disease)
                module = max(detections, key=lambda m: sum(prior_vec[position[g]] for g in m if g in position))
            else:
                module = max(detections, key=len) if detections else set()
            module = set(module)
            module = set(sorted(module, key=lambda g: -hybrid[position[g]])[:SIZE]) if module else set()
            known = label_sets.get(disease, set())
            known = {g for g in known if g in position}
            top200 = set(candidate)
            union = set().union(*detections) if detections else set()
            rows.append({
                "gamma": gamma, "disease_id": disease,
                "union_hits": len(known & union),
                "n_modules": len(detections),
                "known": len(known),
                "top200_hits": len(known & top200),
                "module_hits": len(known & module),
                "z_spec": z_of(disease, module) if module else np.nan,
            })
        print(f"[info] gamma {gamma} done", flush=True)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "prior_recall.parquet", index=False)
    summary = {}
    for gamma, group in table.groupby("gamma"):
        base = group[group.disease_id.isin(largest)]
        summary[str(gamma)] = {
            "diseases": int(len(group)),
            "known_total": int(group.known.sum()),
            "top200_rate": round(float(group.top200_hits.sum() / max(group.known.sum(), 1)), 4),
            "module_rate": round(float(group.module_hits.sum() / max(group.known.sum(), 1)), 4),
            "union_rate": round(float(group.union_hits.sum() / max(group.known.sum(), 1)), 4),
            "modules_mean": round(float(group.n_modules.mean()), 2),
            "z_median": round(float(group.z_spec.median()), 3),
        }
    payload = {"fold": args.fold, "labels": args.labels, "summary": summary,
               "seconds": round(time.time() - started, 1)}
    (out_dir / "prior_recall.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
