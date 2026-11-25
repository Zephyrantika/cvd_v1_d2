"""模块聚类 vs 疾病家族：用模块基因重叠给疾病聚类，与 MONDO 家族比对（ARI/NMI）。

完全不依赖标签与分数，是一个独立的语义效度检验。
同时给出随机模块的对照（打乱基因后重算）。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pathlib import Path

DIR = Path(__file__).resolve().parents[1]
V0 = DIR.parent / "phenotype-network-v0"


def jaccard_matrix(sets: list[set[str]]) -> np.ndarray:
    n = len(sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            union = len(sets[i] | sets[j])
            value = len(sets[i] & sets[j]) / union if union else 0.0
            mat[i, j] = mat[j, i] = value
    return mat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    members = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    members["gene_id"] = members["gene_id"].astype(str)
    members["disease_id"] = members["disease_id"].astype(str)
    sets_of = members.groupby("disease_id")["gene_id"].apply(set).to_dict()

    fam = pd.read_parquet(V0 / "data/folds/disease_fold_assignments.parquet",
                          columns=["disease_id", "disease_family_id"])
    fam["disease_id"] = fam["disease_id"].astype(str)
    fam["disease_family_id"] = fam["disease_family_id"].astype(str)
    family = dict(zip(fam["disease_id"], fam["disease_family_id"]))

    diseases = [d for d in sets_of if d in family]
    sets = [sets_of[d] for d in diseases]
    labels = [family[d] for d in diseases]
    n_families = len(set(labels))

    mat = jaccard_matrix(sets)
    sim = 1.0 - mat
    np.fill_diagonal(sim, 0.0)
    model = AgglomerativeClustering(n_clusters=n_families, metric="precomputed", linkage="average")
    pred = model.fit_predict(sim)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)

    # 随机对照：打乱每病的成员，重算一次
    universe = sorted(set().union(*sets))
    shuffled = [set(rng.choice(universe, size=len(sets_of[d]), replace=False).tolist()) for d in diseases]
    mat_rand = jaccard_matrix(shuffled)
    sim_rand = 1.0 - mat_rand
    np.fill_diagonal(sim_rand, 0.0)
    pred_rand = AgglomerativeClustering(n_clusters=n_families, metric="precomputed",
                                        linkage="average").fit_predict(sim_rand)
    ari_rand = adjusted_rand_score(labels, pred_rand)

    out = DIR / f"outputs/module_clustering/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"disease_id": diseases, "family": labels, "cluster": pred}).to_parquet(
        out / "assignments.parquet", index=False)
    payload = {"fold": args.fold, "diseases": len(diseases), "families": n_families,
               "ari": round(float(ari), 4), "nmi": round(float(nmi), 4),
               "ari_random_control": round(float(ari_rand), 4)}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
