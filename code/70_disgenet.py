"""用 DisGeNET API 拉取 452 个心血管疾病的基因关联，补进疾病—基因标签层。

学术账号只能访问 curated 来源（CLINGEN、CLINVAR、GENCC、ORPHANET、UNIPROT 等），
正好适合当高质量标签。疾病 ID 先由 MONDO 的 UMLS xref 转成 UMLS_Cxxxxxxx。
API key 从环境变量 DGN_KEY 读取，不写进代码。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
DIR = ROOT / "phenotype-modules-v0"
OUT = DIR / "data/external/disgenet"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://api.disgenet.com/api/v1/gda/summary"


def mondo_to_umls() -> dict[str, str]:
    mapping: dict[str, str] = {}
    current = None
    with (V0 / "data/raw/mondo/mondo.obo").open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = None
            elif line.startswith("id: MONDO:"):
                current = line.split("id: ", 1)[1].strip()
            elif line.startswith("xref: UMLS:") and current:
                umls = line.split("xref: UMLS:", 1)[1].split()[0].strip()
                mapping.setdefault(current, f"UMLS_{umls}")
    return mapping


def fetch(disease: str, key: str, page: int = 0) -> dict:
    url = f"{BASE}?disease={disease}&page_number={page}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    key = os.environ.get("DGN_KEY", "")
    if not key:
        print("missing DGN_KEY", file=sys.stderr)
        return 2
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    chain = DIR / "data/chain"
    diseases = pd.read_parquet(chain / "dim_disease.parquet")["disease_id"].astype(str).tolist()
    umls = mondo_to_umls()

    rows = []
    started = time.time()
    for index, disease in enumerate(diseases):
        identifier = umls.get(disease)
        if not identifier:
            continue
        page = 0
        while True:
            try:
                payload = fetch(identifier, key, page)
            except Exception as error:  # noqa: BLE001
                print(f"[warn] {disease} {identifier}: {error}", file=sys.stderr)
                break
            records = payload.get("payload") or []
            for record in records:
                gene = record.get("geneNcbiID")
                if gene is None:
                    continue
                gene = str(int(gene))
                if gene not in universe:
                    continue
                rows.append({"disease_id": disease, "gene_id": gene,
                             "score": float(record.get("score") or 0.0),
                             "ei": float(record.get("ei") or 0.0),
                             "dsi": float(record.get("dsi") or 0.0),
                             "dpi": float(record.get("dpi") or 0.0),
                             "source": str(record.get("source") or ""),
                             "source_id": str(record.get("sourceID") or "")})
            paging = payload.get("paging") or {}
            total = int(paging.get("totalElements") or 0)
            page += 1
            if page * 100 >= total or not records:
                break
            time.sleep(0.2)
        if index % 25 == 0:
            print(f"[info] {index}/{len(diseases)} pairs={len(rows)}", flush=True)
        time.sleep(0.15)

    table = pd.DataFrame(rows).drop_duplicates(["disease_id", "gene_id"])
    table.to_parquet(OUT / "disgenet_gene_disease.parquet", index=False)

    old = pd.read_parquet(chain / "fact_disease_gene.parquet")
    old_pairs = set(zip(old["disease_id"].astype(str), old["gene_id"].astype(str)))
    new_pairs = set(zip(table["disease_id"], table["gene_id"]))
    summary = {"diseases": len(diseases), "mapped_to_umls": int(sum(1 for d in diseases if d in umls)),
               "diseases_with_hits": int(table["disease_id"].nunique()),
               "pairs": int(len(new_pairs)), "new_pairs": int(len(new_pairs - old_pairs)),
               "genes": int(table["gene_id"].nunique()),
               "sources": table["source"].value_counts().head(10).to_dict(),
               "median_score": round(float(table["score"].median()), 3),
               "seconds": round(time.time() - started, 1)}
    (OUT / "disgenet_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
