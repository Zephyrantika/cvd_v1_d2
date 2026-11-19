"""补数据链（二）：解析 HPA 组织特异表达与 WikiPathways 通路，并统一到 Entrez 基因空间。

输出：
  data/external/hpa/hpa_gene_tissue.parquet      基因—组织（含心脏相关标记）
  data/external/wikipathways/wp_gene_pathway.parquet
  data/external/data_chain_summary.json          追加覆盖率
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
EXT = ROOT / "phenotype-modules-v0" / "data" / "external"


def main() -> int:
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    known = pd.read_parquet(V0 / "data/processed/cardiovascular_disease_genes.parquet", columns=["gene_id"])
    known = set(known["gene_id"].astype(str)) & universe

    hgnc = pd.read_csv(EXT / "networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    sym2entrez = {}
    ens2entrez = {}
    for symbol, ens, entrez in zip(hgnc["symbol"].astype(str), hgnc["ensembl_gene_id"].astype(str),
                                   pd.to_numeric(hgnc["entrez_id"], errors="coerce")):
        if pd.isna(entrez):
            continue
        sym2entrez.setdefault(symbol.upper(), str(int(entrez)))
        ens2entrez.setdefault(ens, str(int(entrez)))

    # HPA
    with zipfile.ZipFile(EXT / "hpa/proteinatlas.tsv.zip") as archive:
        with archive.open("proteinatlas.tsv") as handle:
            hpa = pd.read_csv(handle, sep="\t", low_memory=False)
    keep = [c for c in ["Gene", "Ensembl", "RNA tissue specificity", "RNA tissue distribution",
                        "RNA tissue specific nTPM", "Disease involvement"] if c in hpa.columns]
    hpa = hpa[keep].copy()
    hpa["gene_id"] = hpa["Ensembl"].astype(str).map(ens2entrez)
    hpa = hpa.dropna(subset=["gene_id"])
    hpa["heart_flag"] = hpa["RNA tissue specific nTPM"].astype(str).str.contains("heart", case=False, na=False)
    hpa[["gene_id", "RNA tissue specificity", "RNA tissue distribution", "heart_flag"]].to_parquet(
        EXT / "hpa/hpa_gene_tissue.parquet", index=False)

    # WikiPathways
    wp_rows = []
    with (EXT / "wikipathways/wikipathways_human.gmt").open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            for gene in parts[2:]:
                if gene.isdigit():
                    wp_rows.append({"pathway": pathway, "gene_id": gene})
    wp = pd.DataFrame(wp_rows).drop_duplicates()
    wp = wp[wp["gene_id"].isin(universe)]
    wp.to_parquet(EXT / "wikipathways/wp_gene_pathway.parquet", index=False)

    summary_path = EXT / "data_chain_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "hpa_genes_in_universe": int(hpa["gene_id"].isin(universe).sum()),
        "hpa_heart_flagged": int(hpa["heart_flag"].sum()),
        "hpa_heart_in_universe": int((hpa["gene_id"].isin(universe) & hpa["heart_flag"]).sum()),
        "hpa_covers_known_genes": int(len(known & set(hpa["gene_id"]))),
        "wikipathways_pathways": int(wp["pathway"].nunique()),
        "wikipathways_genes_in_universe": int(wp["gene_id"].nunique()),
        "wikipathways_covers_known_genes": int(len(known & set(wp["gene_id"]))),
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["hpa_genes_in_universe", "hpa_heart_flagged", "hpa_heart_in_universe",
                                              "hpa_covers_known_genes", "wikipathways_pathways",
                                              "wikipathways_genes_in_universe", "wikipathways_covers_known_genes"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
