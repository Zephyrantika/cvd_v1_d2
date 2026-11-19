"""补数据链：把 HPO、ClinVar、KEGG、HuRI 四条外部来源对齐到本项目的疾病与基因空间。

输出：
  data/external/labels/hpo_gene_disease.parquet      HPO genes_to_disease（OMIM 疾病）
  data/external/labels/clinvar_gene_disease.parquet ClinVar gene-condition（MONDO/OMIM）
  data/external/kegg/kegg_gene_pathway.parquet      KEGG 通路—基因
  data/external/networks/huri_edges_universe.parquet HuRI 边（限定在候选基因集内）
  data/external/data_chain_summary.json             覆盖率汇总
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
EXT = ROOT / "phenotype-modules-v0" / "data" / "external"
(EXT / "labels").mkdir(parents=True, exist_ok=True)


def parse_mondo_xrefs() -> dict[str, str]:
    """MONDO obo → {OMIM id: MONDO id}。"""
    mapping: dict[str, str] = {}
    current = None
    path = V0 / "data/raw/mondo/mondo.obo"
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = None
            elif line.startswith("id: MONDO:"):
                current = line.split("id: ", 1)[1].strip()
            elif line.startswith("xref: OMIM:") and current:
                omim = line.split("xref: OMIM:", 1)[1].split()[0].strip()
                mapping.setdefault(omim, current)
    return mapping


def main() -> int:
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    known = pd.read_parquet(V0 / "data/processed/cardiovascular_disease_genes.parquet", columns=["disease_id", "gene_id"])
    known["disease_id"] = known["disease_id"].astype(str)
    known["gene_id"] = known["gene_id"].astype(str)
    mapped = pd.read_parquet(V0 / "data/interim/cardiovascular_disease_mapping.parquet")
    diseases = set(known["disease_id"])
    print("our diseases", len(diseases), "known pairs", len(known), "known genes", known.gene_id.nunique())
    print("mapping columns", mapped.columns.tolist())

    omim_to_mondo = parse_mondo_xrefs()
    print("mondo omim xrefs", len(omim_to_mondo))

    # HPO genes_to_disease
    hpo = pd.read_csv(V0 / "data/raw/hpo/genes_to_disease.txt", sep="\t", comment="#", low_memory=False)
    hpo["gene_id"] = hpo["ncbi_gene_id"].astype(str).str.replace("NCBIGene:", "", regex=False)
    hpo["omim"] = hpo["disease_id"].astype(str).str.replace("OMIM:", "", regex=False)
    hpo["disease_mono"] = hpo["omim"].map(omim_to_mondo)
    hpo_ours = hpo[hpo["disease_mono"].isin(diseases) & hpo["gene_id"].isin(universe)]
    hpo_pairs = hpo_ours[["disease_mono", "gene_id"]].drop_duplicates().rename(columns={"disease_mono": "disease_id"})
    hpo_pairs.to_parquet(EXT / "labels/hpo_gene_disease.parquet", index=False)

    # ClinVar gene-condition
    clin = pd.read_csv(EXT / "clinvar/gene_condition_source_id.tsv", sep="\t", low_memory=False, dtype=str)
    clin.columns = [c.lstrip("#") for c in clin.columns]
    clin["gene_id"] = clin["GeneID"].astype(str)
    clin["omim"] = clin["DiseaseMIM"].astype(str).str.split(".").str[0]
    clin["disease_mono"] = clin["omim"].map(omim_to_mondo)
    clin["concept"] = clin["ConceptID"].astype(str).str.replace("C", "MONDO:", regex=False)
    clin_ours = clin[(clin["disease_mono"].isin(diseases) | clin["concept"].isin(diseases)) & clin["gene_id"].isin(universe)]
    clin_pairs = pd.DataFrame({
        "disease_id": clin_ours["disease_mono"].where(clin_ours["disease_mono"].isin(diseases), clin_ours["concept"]),
        "gene_id": clin_ours["gene_id"],
    }).dropna().drop_duplicates()
    clin_pairs.to_parquet(EXT / "labels/clinvar_gene_disease.parquet", index=False)

    # KEGG
    kegg = pd.read_csv(EXT / "kegg/hsa_pathway_links.tsv", sep="\t", header=None, names=["pathway", "gene"])
    kegg["pathway"] = kegg["pathway"].str.replace("path:", "", regex=False)
    kegg["gene_id"] = kegg["gene"].str.replace("hsa:", "", regex=False)
    kegg = kegg[kegg["gene_id"].isin(universe)][["pathway", "gene_id"]].drop_duplicates()
    kegg.to_parquet(EXT / "kegg/kegg_gene_pathway.parquet", index=False)

    # HuRI
    hgnc = pd.read_csv(EXT / "networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    e2g = dict(zip(hgnc["ensembl_gene_id"].astype(str), pd.to_numeric(hgnc["entrez_id"], errors="coerce")))
    huri = pd.read_csv(EXT / "networks/HuRI.tsv", sep="\t", header=None, names=["a", "b"])
    huri["ga"] = huri["a"].map(e2g).astype("Int64").astype(str)
    huri["gb"] = huri["b"].map(e2g).astype("Int64").astype(str)
    huri_u = huri[huri["ga"].isin(universe) & huri["gb"].isin(universe)][["ga", "gb"]].rename(
        columns={"ga": "gene_a", "gb": "gene_b"}).drop_duplicates()
    huri_u.to_parquet(EXT / "networks/huri_edges_universe.parquet", index=False)

    # gene -> phenotype (HPO)
    g2p = pd.read_csv(EXT / "phenotypes/genes_to_phenotype.txt", sep="\t", comment="#", low_memory=False)
    g2p["gene_id"] = g2p["ncbi_gene_id"].astype(str)
    g2p["hpo_id"] = g2p["hpo_id"].astype(str)
    g2p_u = g2p[g2p["gene_id"].isin(universe)][["gene_id", "hpo_id"]].drop_duplicates()
    g2p_u.to_parquet(EXT / "phenotypes/gene_phenotype.parquet", index=False)

    known_pairs = set(zip(known["disease_id"], known["gene_id"]))
    hpo_set = set(zip(hpo_pairs["disease_id"], hpo_pairs["gene_id"]))
    clin_set = set(zip(clin_pairs["disease_id"], clin_pairs["gene_id"]))
    summary = {
        "our_diseases": len(diseases),
        "our_pairs": len(known_pairs),
        "hpo_pairs": len(hpo_set),
        "hpo_diseases": len({d for d, _ in hpo_set}),
        "hpo_genes": len({g for _, g in hpo_set}),
        "hpo_overlap_with_ours": len(hpo_set & known_pairs),
        "hpo_new_pairs": len(hpo_set - known_pairs),
        "clinvar_pairs": len(clin_set),
        "clinvar_diseases": len({d for d, _ in clin_set}),
        "clinvar_genes": len({g for _, g in clin_set}),
        "clinvar_overlap": len(clin_set & known_pairs),
        "clinvar_new_pairs": len(clin_set - known_pairs),
        "union_pairs": len(known_pairs | hpo_set | clin_set),
        "union_diseases": len({d for d, _ in (known_pairs | hpo_set | clin_set)}),
        "kegg_pathways": int(kegg["pathway"].nunique()),
        "kegg_genes_in_universe": int(kegg["gene_id"].nunique()),
        "kegg_covers_known_genes": len({g for _, g in known_pairs} & set(kegg["gene_id"])),
        "huri_edges_in_universe": int(len(huri_u)),
        "huri_genes_in_universe": int(len(set(huri_u.gene_a) | set(huri_u.gene_b))),
        "huri_covers_known_genes": len({g for _, g in known_pairs} & (set(huri_u.gene_a) | set(huri_u.gene_b))),
        "g2p_genes_in_universe": int(g2p_u["gene_id"].nunique()),
        "g2p_terms": int(g2p_u["hpo_id"].nunique()),
        "g2p_covers_known_genes": len({g for _, g in known_pairs} & set(g2p_u["gene_id"])),
    }
    (EXT / "data_chain_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
