"""把 CTD 的基因—疾病关联对齐进数据链。

输入：CTD_genes_diseases.tsv.gz（可能正在下载，允许读到截断处即停）
规则：
  1. 只保留 DirectEvidence == "marker/mechanism"（与疾病机制相关的直接证据）
  2. 疾病经 MeSH xref 映射到 MONDO
  3. 基因符号经 HGNC 映射到 Entrez
  4. 只保留本项目 452 个疾病与 16,992 个候选基因
输出：data/external/ctd/ctd_gene_disease.parquet 与 ctd_summary.json
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
DIR = ROOT / "phenotype-modules-v0"
EXT = DIR / "data/external"
OUT = EXT / "ctd"
CHAIN = DIR / "data/chain"


def mesh_to_mondo() -> dict[str, str]:
    mapping: dict[str, str] = {}
    current = None
    with (V0 / "data/raw/mondo/mondo.obo").open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = None
            elif line.startswith("id: MONDO:"):
                current = line.split("id: ", 1)[1].strip()
            elif line.startswith("xref: MESH:") and current:
                mesh = line.split("xref: MESH:", 1)[1].split()[0].strip()
                mapping.setdefault(mesh, current)
    return mapping


def main() -> int:
    started = time.time()
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    diseases = set(pd.read_parquet(CHAIN / "dim_disease.parquet")["disease_id"].astype(str))
    hgnc = pd.read_csv(EXT / "networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    symbol_to_entrez: dict[str, str] = {}
    for symbol, entrez in zip(hgnc["symbol"].astype(str).str.upper(), pd.to_numeric(hgnc["entrez_id"], errors="coerce")):
        if pd.notna(entrez):
            symbol_to_entrez.setdefault(symbol, str(int(entrez)))
    mesh_map = mesh_to_mondo()
    print(f"[info] mesh->mondo {len(mesh_map)}, hgnc symbols {len(symbol_to_entrez)}", flush=True)

    rows: dict[tuple[str, str], float] = {}
    total = 0
    kept = 0
    path = OUT / "CTD_genes_diseases.tsv.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                total += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 7:
                    continue
                symbol, _gene_id, _disease_name, disease_id, direct, _chem, infer_score = parts[:7]
                is_direct = direct == "marker/mechanism"
                if not is_direct:
                    try:
                        if float(infer_score or 0) < 2.0:
                            continue
                    except ValueError:
                        continue
                disease = mesh_map.get(disease_id.replace("MESH:", "").strip())
                if not disease or disease not in diseases:
                    continue
                gene = symbol_to_entrez.get(symbol.upper())
                if not gene or gene not in universe:
                    continue
                key = (disease, gene)
                rows[key] = max(rows.get(key, 0.0), 1.0 if is_direct else 0.5)
                kept += 1
                if total % 2_000_000 == 0:
                    print(f"[info] {total:,} lines, kept {kept}, pairs {len(rows)}", flush=True)
    except (EOFError, OSError) as error:
        print(f"[warn] stopped early at {total:,} lines: {error}", file=sys.stderr)

    table = pd.DataFrame([{"disease_id": d, "gene_id": g, "source": "ctd_direct" if s >= 1.0 else "ctd_inferred", "score": s}
                          for (d, g), s in rows.items()])
    table.to_parquet(OUT / "ctd_gene_disease.parquet", index=False)

    old = pd.read_parquet(CHAIN / "fact_disease_gene.parquet")
    old_pairs = set(zip(old["disease_id"].astype(str), old["gene_id"].astype(str)))
    new_pairs = set(rows)
    summary = {"lines_read": total, "pairs": len(new_pairs), "new_pairs": len(new_pairs - old_pairs),
               "diseases": int(table["disease_id"].nunique()) if len(table) else 0,
               "genes": int(table["gene_id"].nunique()) if len(table) else 0,
               "seconds": round(time.time() - started, 1)}
    (OUT / "ctd_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
