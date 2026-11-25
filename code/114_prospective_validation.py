"""前瞻性验证：用 2024 年后新进入 ClinGen 的基因—疾病对，检验现有模块能否召回。

模块构建完全不用任何标签，所以新标签天然是「未见过」的样本。
对比 2023 年底之前与 2024 年之后的 top-200 命中率与模块并集召回。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1]
V0 = DIR.parent / "phenotype-network-v0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cutoff", default="2024-01-01")
    args = parser.parse_args()

    clingen = pd.read_csv(DIR / "data/external/clingen/gene_validity.csv", skiprows=4, low_memory=False)
    clingen = clingen[~clingen["GENE SYMBOL"].astype(str).str.startswith("+")]
    clingen["date"] = pd.to_datetime(clingen["CLASSIFICATION DATE"], errors="coerce", utc=True)
    clingen = clingen.dropna(subset=["date"])
    clingen["disease_id"] = clingen["DISEASE ID (MONDO)"].astype(str).str.strip()
    clingen["symbol"] = clingen["GENE SYMBOL"].astype(str).str.strip()

    dim_gene = pd.read_parquet(DIR / "data/chain/dim_gene.parquet")
    symbol_to_gene = {}
    for gid, sym in zip(dim_gene["gene_id"].astype(str), dim_gene["symbol"].astype(str)):
        symbol_to_gene.setdefault(sym.upper(), gid)

    dim_disease = pd.read_parquet(DIR / "data/chain/dim_disease.parquet")
    our_diseases = set(dim_disease["disease_id"].astype(str))

    clingen["gene_id"] = clingen["symbol"].str.upper().map(symbol_to_gene)
    pairs = clingen.dropna(subset=["gene_id"])[["disease_id", "gene_id", "date"]].drop_duplicates()
    pairs = pairs[pairs["disease_id"].isin(our_diseases)]
    print(f"[info] ClinGen 映射到本项目：{len(pairs)} 对，疾病 {pairs.disease_id.nunique()} 个")

    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    pairs["era"] = np.where(pairs["date"] >= cutoff, "new_2024plus", "old_le2023")
    print(pairs.groupby("era").size().to_dict())

    sys.path.insert(0, str(DIR / "src"))
    from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

    matrix = load_saved_scores(DIR / f"outputs/scores/fold_{args.fold}/score_matrix.npz")
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    n_genes = percentiles.shape[1]

    members = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    members["gene_id"] = members["gene_id"].astype(str)
    members["disease_id"] = members["disease_id"].astype(str)
    modules = members.groupby("disease_id")["gene_id"].apply(set).to_dict()

    rows = []
    for row in pairs.itertuples(index=False):
        if row.disease_id not in matrix.disease_ids or row.gene_id not in position:
            continue
        di = matrix.disease_index(row.disease_id)
        pct = float(percentiles[di, position[row.gene_id]])
        threshold = np.quantile(percentiles[di, :], 1 - 200 / n_genes)
        rows.append({"disease_id": row.disease_id, "gene_id": row.gene_id, "era": row.era,
                     "top200": bool(pct >= threshold),
                     "in_module": bool(row.gene_id in modules.get(row.disease_id, set()))})
    table = pd.DataFrame(rows)
    out = DIR / f"outputs/prospective/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "prospective.parquet", index=False)

    summary = {"fold": args.fold, "cutoff": args.cutoff,
               "pairs_total": int(len(pairs)), "pairs_scored": int(len(table))}
    for era, group in table.groupby("era"):
        summary[era] = {"pairs": int(len(group)), "diseases": int(group.disease_id.nunique()),
                        "top200_rate": round(float(group.top200.mean()), 4),
                        "module_rate": round(float(group.in_module.mean()), 4)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
