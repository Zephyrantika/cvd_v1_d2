"""在新标签集（1,761 对）上比较基线模块与精修模块的已知基因覆盖。

指标：模块内命中已知基因的比例（module recall），按 T1/T2/T3 分层，五折合并。
已知基因来自数据链 fact_disease_gene（六个来源去重），不是旧的 568 对。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "phenotype-modules-v0"


def main() -> int:
    chain = DIR / "data/chain"
    labels = pd.read_parquet(chain / "fact_disease_gene.parquet")
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["gene_id"] = labels["gene_id"].astype(str)
    labels = labels[["disease_id", "gene_id"]].drop_duplicates()
    genes_by_disease = labels.groupby("disease_id")["gene_id"].apply(set).to_dict()
    n_by_disease = {d: len(g) for d, g in genes_by_disease.items()}

    rows = []
    for fold in range(5):
        mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/modules.parquet")
        mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{fold}/members.parquet")
        mem["gene_id"] = mem["gene_id"].astype(str)
        largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
        base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
        refined = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{fold}_lam1.0_mu1.0/lam_refine_members.parquet")
        refined["gene_id"] = refined["gene_id"].astype(str)
        ref_sets = refined.groupby("disease_id")["gene_id"].apply(set).to_dict()

        for disease, module_id in largest.items():
            known = genes_by_disease.get(disease)
            if not known or disease not in ref_sets:
                continue
            base = base_sets.get(module_id, set())
            ref = ref_sets[disease]
            if not base or not ref:
                continue
            rows.append({
                "fold": fold, "disease_id": disease, "n_known": len(known),
                "base_hits": len(base & known), "ref_hits": len(ref & known),
                "base_recall": len(base & known) / len(known),
                "ref_recall": len(ref & known) / len(known),
            })

    table = pd.DataFrame(rows)
    table["tier"] = np.where(table.n_known >= 5, "T1",
                             np.where(table.n_known >= 3, "T2", "T3"))
    out = DIR / "outputs/label_eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "module_recall.parquet", index=False)

    summary = {}
    for tier in ["T1", "T2", "T3", "all"]:
        sub = table if tier == "all" else table[table.tier == tier]
        if not len(sub):
            continue
        summary[tier] = {
            "diseases": int(len(sub)),
            "known_genes": int(sub.n_known.sum()),
            "base_hits": int(sub.base_hits.sum()),
            "ref_hits": int(sub.ref_hits.sum()),
            "base_recall": round(float(sub.base_hits.sum() / sub.n_known.sum()), 3),
            "ref_recall": round(float(sub.ref_hits.sum() / sub.n_known.sum()), 3),
            "diseases_improved": int((sub.ref_hits > sub.base_hits).sum()),
            "diseases_worse": int((sub.ref_hits < sub.base_hits).sum()),
            "p": float(wilcoxon(sub.ref_recall, sub.base_recall).pvalue),
        }
    (out / "module_recall.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
