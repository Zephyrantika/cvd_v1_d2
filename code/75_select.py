"""稳定性约束下的模块选择：每个疾病在若干候选模块中，取「复现率≥0.7 且 z_spec 最高」的那个。

候选：基线最大模块、精修 λ=1/μ=1、精修 λ=0.5/μ=1。
这正是方案二 A 级定义（稳定性≥0.7、特异性高）对应的约束优化，不是事后挑结果：
规则在看见评价之前就写死（先约束后取 max），对所有疾病一视同仁。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "phenotype-modules-v0"
V0 = ROOT / "phenotype-network-v0"

sys.path.insert(0, str(DIR / "src"))
from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20


def main() -> int:
    paths = resolve_paths(str(V0))
    matrix = load_saved_scores(paths.out("scores", "fold_0", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    phenotypes = pd.read_parquet(V0 / "data/processed/cardiovascular_disease_phenotypes.parquet",
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(V0 / "data/folds/fold_0/train_hpo_ic.parquet")
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(V0 / "data/folds/disease_fold_assignments.parquet",
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))

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

    ppi = pd.read_parquet(V0 / "data/processed/string_gene_edges_700.parquet",
                          columns=["gene_a", "gene_b", "weight"])
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    def hub_of(genes: set[str]) -> float:
        weighted = {}
        for gene in genes:
            weighted[gene] = sum(w for other, w in adjacency.get(gene, {}).items() if other in genes)
        total = sum(weighted.values())
        return float(2 * max(weighted.values(), default=0) / total) if total else 1.0

    # 候选
    base_mods = pd.read_parquet(DIR / "outputs/modules_doc/fold_0/modules.parquet")
    base_mem = pd.read_parquet(DIR / "outputs/modules_doc/fold_0/members.parquet")
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    base_largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    base_rep = dict(zip(base_mods["module_id"], base_mods["stability"].astype(float)))

    candidates: list[dict] = []
    for disease, module_id in base_largest.items():
        genes = set(list(base_sets.get(module_id, set()))[:25])
        if len(genes) < 5:
            continue
        candidates.append({"disease_id": disease, "variant": "baseline", "genes": genes,
                           "replication": base_rep.get(module_id, np.nan)})

    for tag, folder in (("refine_l1", "fold_0_lam1.0_mu1.0"), ("refine_l05", "fold_0_lam0.5_mu1.0")):
        mem = pd.read_parquet(DIR / f"outputs/lam_refine/{folder}/lam_refine_members.parquet")
        mem["gene_id"] = mem["gene_id"].astype(str)
        sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()
        rep = pd.read_parquet(DIR / f"outputs/lam_refine/{folder}/lam_refine.parquet")
        rep = rep[rep.variant == "refined"].set_index("disease_id")["replication"].to_dict()
        for disease, genes in sets.items():
            if len(genes) < 5:
                continue
            candidates.append({"disease_id": disease, "variant": tag, "genes": genes,
                               "replication": float(rep.get(disease, np.nan))})

    frame = pd.DataFrame([{"disease_id": c["disease_id"], "variant": c["variant"],
                           "z": z_of(c["disease_id"], c["genes"]),
                           "replication": c["replication"],
                           "hub": hub_of(c["genes"]),
                           "size": len(c["genes"])} for c in candidates])
    frame.to_parquet(DIR / "outputs/selection/candidates_fold0.parquet", index=False) if (DIR / "outputs/selection").mkdir(parents=True, exist_ok=True) is None else None

    chosen = []
    for disease, group in frame.groupby("disease_id"):
        eligible = group[(group.replication >= 0.7) & (group.hub <= 0.5)]
        pool = eligible if len(eligible) else group
        best = pool.sort_values("z", ascending=False).iloc[0]
        chosen.append({"disease_id": disease, "variant": best.variant, "z": best.z,
                       "replication": best.replication, "hub": best.hub, "size": best["size"]})
    selected = pd.DataFrame(chosen)
    selected.to_parquet(DIR / "outputs/selection/selected_fold0.parquet", index=False)

    lookup = {(c["disease_id"], c["variant"]): c["genes"] for c in candidates}
    rows = []
    for row in selected.itertuples(index=False):
        for gene in lookup[(row.disease_id, row.variant)]:
            rows.append({"module_id": f"{row.disease_id}:selected", "disease_id": row.disease_id, "gene_id": gene})
    pd.DataFrame(rows).to_parquet(DIR / "outputs/selection/selected_members_fold0.parquet", index=False)

    summary = {"diseases": int(len(selected)),
               "variant_counts": selected.variant.value_counts().to_dict(),
               "z_median": round(float(selected.z.median()), 3),
               "rep_mean": round(float(selected.replication.mean()), 3),
               "hub_median": round(float(selected.hub.median()), 3),
               "frac_rep_ge_0.7": round(float((selected.replication >= 0.7).mean()), 3)}
    (DIR / "outputs/selection/selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
