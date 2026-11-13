"""常见心血管性状的可行性检查：从 Open Targets 取强证据靶点，映射到我们的候选集与 PPI。

目的：判断"换成 7 个常见心血管性状 + GWAS 级证据"这条路能不能走。
输出每个性状：强证据靶点数、能映射到 NCBI 的、落在 16,992 候选集里的、在 STRING 700 上有边的。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\ComputerScience_file\VERTICVS_RES_PVBLICVS_STRVCTVS\ARBEIT_G3S1\SENIOR_MODEL_STRUCTURE")
V0 = BASE / "phenotype-network-v0"
API = "https://api.platform.opentargets.org/api/v4/graphql"
QUERY = """
query($id:String!,$size:Int!){
  disease(efoId:$id){
    name
    associatedTargets(page:{index:0,size:$size}){count rows{score target{approvedSymbol}}}
  }
}
"""
TRAITS = {
    "coronary_artery_disease": "MONDO_0005010",
    "myocardial_infarction": "MONDO_0005068",
    "heart_failure": "MONDO_0005252",
    "atrial_fibrillation": "MONDO_0004981",
    "hypertension": "MONDO_0005044",
    "ldl_cholesterol": "EFO_0004611",
    "stroke": "MONDO_0005294",
}


def fetch(ot_id: str) -> dict | None:
    body = json.dumps({"query": QUERY, "variables": {"id": ot_id, "size": 500}}).encode()
    request = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())["data"]["disease"]


def main() -> int:
    symbol_map = pd.read_parquet(V0 / "data/processed/external_gene_symbol_mapping.parquet")
    symbol_map["gene_id"] = symbol_map["gene_id"].astype(str)
    clean = symbol_map.loc[~symbol_map["ambiguous_symbol"].astype(bool)]
    symbol_to_gene = dict(zip(clean["gene_symbol"].astype(str), clean["gene_id"]))

    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    ppi = pd.read_parquet(V0 / "data/processed/string_gene_edges_700.parquet", columns=["gene_a", "gene_b"])
    ppi_genes = set(ppi["gene_a"].astype(str)) | set(ppi["gene_b"].astype(str))

    out_dir = BASE / "phenotype-modules-v0/outputs/external"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for trait, ot_id in TRAITS.items():
        try:
            disease = fetch(ot_id)
        except Exception as error:
            rows.append({"trait": trait, "ot_id": ot_id, "error": str(error)})
            continue
        if not disease:
            rows.append({"trait": trait, "ot_id": ot_id, "error": "not_found"})
            continue
        records = disease["associatedTargets"]["rows"]
        for threshold in (0.3, 0.5):
            selected = [r for r in records if r["score"] >= threshold]
            symbols = [r["target"]["approvedSymbol"] for r in selected]
            genes = {symbol_to_gene[s] for s in symbols if s in symbol_to_gene}
            rows.append({
                "trait": trait,
                "ot_id": ot_id,
                "ot_name": disease["name"],
                "threshold": threshold,
                "n_targets": len(selected),
                "n_mapped": len(genes),
                "n_in_universe": len(genes & universe),
                "n_in_ppi": len(genes & ppi_genes),
                "n_all_three": len(genes & universe & ppi_genes),
            })
        time.sleep(0.3)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "common_cvd_feasibility.parquet", index=False)
    ok = table.dropna(subset=["threshold"])
    summary = {
        "traits": len(TRAITS),
        "rows": ok.to_dict(orient="records"),
        "by_threshold": {
            str(t): {
                "median_n_targets": int(ok.loc[ok.threshold == t, "n_targets"].median()),
                "median_n_all_three": int(ok.loc[ok.threshold == t, "n_all_three"].median()),
                "max_n_all_three": int(ok.loc[ok.threshold == t, "n_all_three"].max()),
            } for t in (0.3, 0.5)
        },
    }
    (out_dir / "common_cvd_feasibility.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(summary["by_threshold"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
