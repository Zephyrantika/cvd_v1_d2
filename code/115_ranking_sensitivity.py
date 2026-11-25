"""排序质量敏感性：模拟排序层改进后，模块层能拿到什么。

对每个疾病，把已知基因按比例 h 强行放进 top-200（替换排名最低的非已知基因），
重跑 Leiden 与模块截取，看模块召回与 z 如何变化。
h=0 即现状；h=1 表示排序层能把全部已知基因送进候选集（上界）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    parser.add_argument("--hs", default="0,0.3,0.5,0.7,1.0")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    out = paths.out("ranking_sensitivity", f"fold_{args.fold}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    n_genes = percentiles.shape[1]

    labels = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/chain/fact_disease_gene.parquet")
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["gene_id"] = labels["gene_id"].astype(str)
    labels = labels[labels["source"].isin(CURATED)].drop_duplicates()
    known_of = labels.groupby("disease_id")["gene_id"].apply(set).to_dict()

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

    rows = []
    for h in [float(x) for x in args.hs.split(",")]:
        for disease in matrix.disease_ids:
            di = matrix.disease_index(disease)
            order = np.argsort(-percentiles[di, :])
            known = [g for g in known_of.get(disease, set()) if g in position]
            if not known:
                continue
            k = int(round(h * len(known)))
            forced = known[:k]
            forced_idx = set(position[g] for g in forced)
            candidate_idx = list(forced_idx) + [i for i in order if i not in forced_idx][: 200 - len(forced_idx)]
            candidate = tuple(matrix.gene_ids[i] for i in candidate_idx)
            detections = detect(candidate, ppi, seed=stable_seed(115, args.fold, disease, int(h * 100)))
            if not detections:
                continue
            module = max(detections, key=len)
            module = set(sorted(module, key=lambda g: -percentiles[di, position[g]])[:SIZE])
            hits = len(set(known) & module)
            union = set().union(*[set(m) for m in detections])
            rows.append({"h": h, "disease_id": disease, "known": len(known), "hits": hits,
                         "union_hits": len(set(known) & union),
                         "module_size": len(module), "z": z_of(disease, module)})
        print(f"[info] h={h} done", flush=True)

    table = pd.DataFrame(rows)
    table.to_parquet(out / "sensitivity.parquet", index=False)
    summary = {}
    for h, group in table.groupby("h"):
        summary[str(h)] = {"diseases": int(len(group)),
                           "module_recall": round(float(group.hits.sum() / max(group.known.sum(), 1)), 4),
                           "union_recall": round(float(group.union_hits.sum() / max(group.known.sum(), 1)), 4),
                           "z_median": round(float(group.z.median()), 3)}
    payload = {"fold": args.fold, "summary": summary}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
