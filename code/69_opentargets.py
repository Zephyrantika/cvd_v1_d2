"""从 Open Targets 拉取 452 个心血管疾病的关联靶点，扩充疾病—基因标签。

Open Targets 的疾病 ID 用下划线格式（MONDO_0007947），与本地冒号格式不同，先转换。
每个疾病取关联靶点（按 score 排序，最多 200 个），符号经 HGNC 映射回 Entrez。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
EXT = ROOT / "phenotype-modules-v0" / "data" / "external"
OUT = EXT / "opentargets"
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
QUERY = """
query($id:String!,$size:Int!){ disease(efoId:$id){ id name
  associatedTargets(page:{index:0,size:$size}){ count rows { target { approvedSymbol } score } } } }
"""


def gql(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    chain = ROOT / "phenotype-modules-v0" / "data" / "chain"
    diseases = pd.read_parquet(chain / "dim_disease.parquet")["disease_id"].astype(str).tolist()

    hgnc = pd.read_csv(EXT / "networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    symbol_to_entrez: dict[str, str] = {}
    for symbol, entrez in zip(hgnc["symbol"].astype(str).str.upper(), pd.to_numeric(hgnc["entrez_id"], errors="coerce")):
        if pd.notna(entrez):
            symbol_to_entrez.setdefault(symbol, str(int(entrez)))

    rows = []
    started = time.time()
    for index, disease in enumerate(diseases):
        ot_id = disease.replace("MONDO:", "MONDO_")
        try:
            payload = gql({"query": QUERY, "variables": {"id": ot_id, "size": 200}})
            record = payload.get("data", {}).get("disease")
        except Exception as error:  # noqa: BLE001
            print(f"[warn] {disease}: {error}", file=sys.stderr)
            continue
        if not record:
            continue
        for hit in record["associatedTargets"]["rows"]:
            symbol = str(hit["target"].get("approvedSymbol", "")).upper()
            gene = symbol_to_entrez.get(symbol)
            if gene and gene in universe:
                rows.append({"disease_id": disease, "gene_id": gene, "score": float(hit["score"]),
                             "source": "opentargets"})
        if index % 50 == 0:
            print(f"[info] {index}/{len(diseases)} {disease} total={len(rows)}", flush=True)

    table = pd.DataFrame(rows).drop_duplicates(["disease_id", "gene_id"])
    table.to_parquet(OUT / "ot_gene_disease.parquet", index=False)

    old = pd.read_parquet(chain / "fact_disease_gene.parquet")
    old_pairs = set(zip(old["disease_id"].astype(str), old["gene_id"].astype(str)))
    new_pairs = set(zip(table["disease_id"], table["gene_id"]))
    summary = {
        "diseases_queried": len(diseases),
        "diseases_with_hits": int(table["disease_id"].nunique()),
        "pairs": int(len(new_pairs)),
        "new_pairs": int(len(new_pairs - old_pairs)),
        "genes": int(table["gene_id"].nunique()),
        "median_score": round(float(table["score"].median()), 3),
        "pairs_score_ge_0.5": int((table["score"] >= 0.5).sum()),
        "new_pairs_score_ge_0.5": int(len({p for p, s in zip(zip(table["disease_id"], table["gene_id"]), table["score"]) if s >= 0.5} - old_pairs)),
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "ot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
