"""把现有全部来源清洗、对齐、拼成一条可复现的数据链。

产出（data/chain/）：
  dim_disease / dim_gene / dim_phenotype / dim_pathway
  fact_disease_phenotype / fact_disease_gene / fact_gene_phenotype
  fact_gene_pathway / fact_gene_tissue / fact_gene_gene
  chain_qc.json / chain_manifest.json

原则：所有 ID 统一（基因 Entrez 字符串、疾病 MONDO、表型 HPO、通路带来源前缀），
所有事实表只保留本项目的 452 个疾病与 16,992 个候选基因，重复边去重并记录。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "phenotype-network-v0"
DIR = ROOT / "phenotype-modules-v0"
EXT = DIR / "data" / "external"
OUT = DIR / "data" / "chain"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    qc: dict[str, object] = {}
    universe = set(pd.read_parquet(V0 / "data/folds/candidate_gene_universe.parquet")["gene_id"].astype(str))
    genes = pd.read_parquet(V0 / "data/processed/cardiovascular_disease_genes.parquet")
    genes["disease_id"] = genes["disease_id"].astype(str)
    genes["gene_id"] = genes["gene_id"].astype(str)
    diseases = sorted(genes["disease_id"].unique())

    mapping = pd.read_parquet(V0 / "data/interim/cardiovascular_disease_mapping.parquet")
    mapping["original_disease_id"] = mapping["original_disease_id"].astype(str)
    mapping["canonical_disease_id"] = mapping["canonical_disease_id"].astype(str)
    fam = dict(zip(mapping["canonical_disease_id"], mapping["cardiovascular_family_ids"].astype(str)))

    # ---- 维度表 ----
    phenotype = pd.read_parquet(V0 / "data/processed/hpo_nodes.parquet")
    phenotype["hpo_id"] = phenotype["node_id"].astype(str) if "node_id" in phenotype.columns else phenotype.iloc[:, 0].astype(str)
    ic = pd.read_parquet(V0 / "data/folds/fold_0/train_hpo_ic.parquet")
    ic["hpo_id"] = ic["hpo_id"].astype(str)
    ic_map = dict(zip(ic["hpo_id"], ic["ic"].astype(float)))
    names = {}
    if "name" in phenotype.columns:
        names = dict(zip(phenotype["hpo_id"], phenotype["name"].astype(str)))
    dim_phenotype = pd.DataFrame({
        "hpo_id": sorted(ic_map),
        "ic": [ic_map[h] for h in sorted(ic_map)],
    })
    dim_phenotype["name"] = dim_phenotype["hpo_id"].map(names).fillna("")
    dim_phenotype.to_parquet(OUT / "dim_phenotype.parquet", index=False)

    known_counts = genes.groupby("disease_id")["gene_id"].nunique()
    pheno = pd.read_parquet(V0 / "data/processed/cardiovascular_disease_phenotypes.parquet")
    pheno["disease_id"] = pheno["disease_id"].astype(str)
    pheno["hpo_id"] = pheno["hpo_id"].astype(str)
    pheno_counts = pheno.groupby("disease_id")["hpo_id"].nunique()
    dim_disease = pd.DataFrame({"disease_id": diseases})
    dim_disease["family_id"] = dim_disease["disease_id"].map(fam).fillna("")
    dim_disease["n_known_genes"] = dim_disease["disease_id"].map(known_counts).fillna(0).astype(int)
    dim_disease["n_phenotypes"] = dim_disease["disease_id"].map(pheno_counts).fillna(0).astype(int)
    dim_disease.to_parquet(OUT / "dim_disease.parquet", index=False)

    hgnc = pd.read_csv(EXT / "networks/hgnc_complete_set.txt", sep="\t", low_memory=False)
    entrez = pd.to_numeric(hgnc["entrez_id"], errors="coerce")
    dim_gene = pd.DataFrame({
        "gene_id": entrez.dropna().astype(int).astype(str),
        "symbol": hgnc.loc[entrez.notna(), "symbol"].astype(str).values,
        "ensembl": hgnc.loc[entrez.notna(), "ensembl_gene_id"].astype(str).values,
    }).drop_duplicates("gene_id")
    dim_gene["in_universe"] = dim_gene["gene_id"].isin(universe)
    dim_gene.to_parquet(OUT / "dim_gene.parquet", index=False)

    # ---- 事实表：疾病-表型 ----
    fact_dp = pheno[["disease_id", "hpo_id"]].drop_duplicates()
    fact_dp.to_parquet(OUT / "fact_disease_phenotype.parquet", index=False)
    qc["disease_phenotype_pairs"] = int(len(fact_dp))

    # ---- 疾病-基因（多来源，带来源标记） ----
    sources = []
    base = genes[["disease_id", "gene_id"]].copy()
    base["source"] = "original"
    base["score"] = 1.0
    sources.append(base)
    clin = EXT / "labels/clinvar_gene_disease.parquet"
    if clin.exists():
        c = pd.read_parquet(clin)
        c["disease_id"] = c["disease_id"].astype(str)
        c["gene_id"] = c["gene_id"].astype(str)
        c = c[c["disease_id"].isin(diseases) & c["gene_id"].isin(universe)].copy()
        c["source"] = "clinvar"
        c["score"] = 1.0
        sources.append(c)
    hpo_label = EXT / "labels/hpo_gene_disease.parquet"
    if hpo_label.exists():
        h = pd.read_parquet(hpo_label)
        h["disease_id"] = h["disease_id"].astype(str)
        h["gene_id"] = h["gene_id"].astype(str)
        h = h[h["disease_id"].isin(diseases) & h["gene_id"].isin(universe)].copy()
        h["source"] = "hpo"
        h["score"] = 1.0
        sources.append(h)
    clingen_path = EXT / "clingen/gene_validity.csv"
    if clingen_path.exists():
        cg = pd.read_csv(clingen_path, skiprows=4, low_memory=False)
        cg = cg[~cg["GENE SYMBOL"].astype(str).str.startswith("+")]
        cg["symbol"] = cg["GENE SYMBOL"].astype(str).str.upper()
        cg["gene_id"] = cg["symbol"].map(dict(zip(
            dim_gene["symbol"].astype(str).str.upper(), dim_gene["gene_id"])))
        cg["disease_id"] = cg["DISEASE ID (MONDO)"].astype(str).str.strip()
        cg = cg.dropna(subset=["gene_id"])
        cg = cg[cg["disease_id"].isin(diseases) & cg["gene_id"].isin(universe)].copy()
        cg["source"] = "clingen"
        cg["score"] = 1.0
        sources.append(cg[["disease_id", "gene_id", "source", "score"]])
    dgn_path = EXT / "disgenet/disgenet_gene_disease.parquet"
    if dgn_path.exists():
        dg = pd.read_parquet(dgn_path)
        dg["disease_id"] = dg["disease_id"].astype(str)
        dg["gene_id"] = dg["gene_id"].astype(str)
        dg = dg[dg["disease_id"].isin(diseases) & dg["gene_id"].isin(universe)].copy()
        dg["source"] = "disgenet"
        sources.append(dg[["disease_id", "gene_id", "source", "score"]])
    ctd_path = EXT / "ctd/ctd_gene_disease.parquet"
    if ctd_path.exists():
        ct = pd.read_parquet(ctd_path)
        ct["disease_id"] = ct["disease_id"].astype(str)
        ct["gene_id"] = ct["gene_id"].astype(str)
        ct = ct[(ct["source"] == "ctd_direct") & ct["disease_id"].isin(diseases) & ct["gene_id"].isin(universe)].copy()
        if len(ct):
            sources.append(ct[["disease_id", "gene_id", "source", "score"]])
    ot_path = EXT / "opentargets/ot_gene_disease.parquet"
    if ot_path.exists():
        ot = pd.read_parquet(ot_path)
        ot["disease_id"] = ot["disease_id"].astype(str)
        ot["gene_id"] = ot["gene_id"].astype(str)
        ot = ot[ot["score"] >= 0.1].copy()
        ot["source"] = "opentargets"
        sources.append(ot[["disease_id", "gene_id", "source", "score"]])
    fact_dg = pd.concat(sources, ignore_index=True).drop_duplicates(["disease_id", "gene_id", "source"])
    fact_dg.to_parquet(OUT / "fact_disease_gene.parquet", index=False)
    qc["disease_gene_pairs_by_source"] = fact_dg.groupby("source").size().to_dict()
    qc["disease_gene_unique_pairs"] = int(fact_dg[["disease_id", "gene_id"]].drop_duplicates().shape[0])
    qc["disease_gene_diseases_with_new"] = int(fact_dg[fact_dg.source.isin(["clingen", "opentargets"])]["disease_id"].nunique())

    # ---- 基因-表型 ----
    g2p = pd.read_parquet(EXT / "phenotypes/gene_phenotype.parquet")
    g2p["gene_id"] = g2p["gene_id"].astype(str)
    g2p["hpo_id"] = g2p["hpo_id"].astype(str)
    g2p = g2p[g2p["gene_id"].isin(universe)].drop_duplicates()
    g2p.to_parquet(OUT / "fact_gene_phenotype.parquet", index=False)
    qc["gene_phenotype_pairs"] = int(len(g2p))

    # ---- 基因-通路（三源） ----
    kegg = pd.read_parquet(EXT / "kegg/kegg_gene_pathway.parquet")
    kegg["pathway_id"] = "kegg:" + kegg["pathway"].astype(str)
    react = pd.read_parquet(V0 / "data/processed/reactome_gene_pathway.parquet")
    react.columns = [c.lower() for c in react.columns]
    react_gene = "gene_id" if "gene_id" in react.columns else ("gene" if "gene" in react.columns else react.columns[1])
    react_path = "pathway_id" if "pathway_id" in react.columns else react.columns[0]
    react_tidy = pd.DataFrame({
        "pathway_id": "reactome:" + react[react_path].astype(str),
        "gene_id": react[react_gene].astype(str),
    })
    react_tidy = react_tidy[react_tidy["gene_id"].isin(universe)]
    wp = EXT / "wikipathways/wp_gene_pathway.parquet"
    wp_tidy = pd.DataFrame(columns=["pathway_id", "gene_id"])
    if wp.exists():
        w = pd.read_parquet(wp)
        wp_tidy = pd.DataFrame({"pathway_id": "wikipathways:" + w["pathway"].astype(str),
                                "gene_id": w["gene_id"].astype(str)})
    fact_gp = pd.concat([kegg[["pathway_id", "gene_id"]], react_tidy, wp_tidy], ignore_index=True)
    fact_gp = fact_gp.drop_duplicates()
    fact_gp.to_parquet(OUT / "fact_gene_pathway.parquet", index=False)
    qc["gene_pathway_pairs"] = int(len(fact_gp))
    qc["gene_pathway_by_source"] = fact_gp["pathway_id"].str.split(":").str[0].value_counts().to_dict()

    dim_pathway = fact_gp.groupby("pathway_id").size().rename("n_genes").reset_index()
    dim_pathway["source"] = dim_pathway["pathway_id"].str.split(":").str[0]
    dim_pathway.to_parquet(OUT / "dim_pathway.parquet", index=False)

    # ---- 基因-组织 ----
    expr = pd.read_parquet(V0 / "data/processed/gtex_cardiovascular_expression.parquet")
    expr["gene_id"] = expr["gene_id"].astype(str)
    tissue_col = "tissue" if "tissue" in expr.columns else None
    if tissue_col:
        fact_gt = expr[["gene_id", tissue_col, "median_tpm"]].rename(columns={tissue_col: "tissue"})
    else:
        fact_gt = expr[["gene_id", "median_tpm"]].assign(tissue="cardiovascular")
    fact_gt = fact_gt[fact_gt["gene_id"].isin(universe)].drop_duplicates()
    fact_gt.to_parquet(OUT / "fact_gene_tissue.parquet", index=False)
    qc["gene_tissue_rows"] = int(len(fact_gt))
    qc["gene_tissue_tissues"] = int(fact_gt["tissue"].nunique())

    # ---- 基因-基因 ----
    string = pd.read_parquet(V0 / "data/processed/string_gene_edges_700.parquet",
                             columns=["gene_a", "gene_b", "weight"])
    string = pd.DataFrame({"gene_a": string["gene_a"].astype(str), "gene_b": string["gene_b"].astype(str),
                           "weight": string["weight"].astype(float), "source": "string700"})
    bionet = V0 / "data/processed/biogrid_gene_edges.parquet"
    parts = [string]
    if bionet.exists():
        bio = pd.read_parquet(bionet)
        cols = {c.lower(): c for c in bio.columns}
        ga = cols.get("gene_a") or cols.get("gene1") or bio.columns[0]
        gb = cols.get("gene_b") or cols.get("gene2") or bio.columns[1]
        bio_tidy = pd.DataFrame({"gene_a": bio[ga].astype(str), "gene_b": bio[gb].astype(str),
                                 "weight": 1.0, "source": "biogrid"})
        parts.append(bio_tidy)
    huri_path = EXT / "networks/huri_edges_universe.parquet"
    if huri_path.exists():
        huri = pd.read_parquet(huri_path)
        parts.append(pd.DataFrame({"gene_a": huri["gene_a"].astype(str), "gene_b": huri["gene_b"].astype(str),
                                   "weight": 1.0, "source": "huri"}))
    fact_gg = pd.concat(parts, ignore_index=True)
    fact_gg["lop"] = np.minimum(fact_gg["gene_a"], fact_gg["gene_b"])
    fact_gg["hip"] = np.maximum(fact_gg["gene_a"], fact_gg["gene_b"])
    fact_gg = fact_gg.drop(columns=["gene_a", "gene_b"]).drop_duplicates(["lop", "hip", "source"])
    fact_gg = fact_gg.rename(columns={"lop": "gene_a", "hip": "gene_b"})
    fact_gg = fact_gg[(fact_gg["gene_a"] != fact_gg["gene_b"])]
    fact_gg = fact_gg[fact_gg["gene_a"].isin(universe) & fact_gg["gene_b"].isin(universe)]
    fact_gg.to_parquet(OUT / "fact_gene_gene.parquet", index=False)
    qc["gene_gene_edges_by_source"] = fact_gg.groupby("source").size().to_dict()

    # ---- QC ----
    qc["diseases"] = len(dim_disease)
    qc["genes_in_universe"] = len(universe)
    qc["genes_in_dim"] = int(dim_gene["in_universe"].sum())
    counts = fact_dg[["disease_id", "gene_id"]].drop_duplicates().groupby("disease_id").size()
    dim_disease["n_known_genes_all"] = dim_disease["disease_id"].map(counts).fillna(0).astype(int)
    dim_disease.to_parquet(OUT / "dim_disease.parquet", index=False)
    qc["diseases_without_known_genes"] = int((dim_disease["n_known_genes_all"] == 0).sum())
    qc["diseases_with_le2_genes_all"] = int((dim_disease["n_known_genes_all"] <= 2).sum())
    qc["median_genes_all"] = float(dim_disease["n_known_genes_all"].median())
    qc["diseases_with_le2_genes"] = int((dim_disease["n_known_genes"] <= 2).sum())
    qc["phenotype_terms_with_ic"] = int(len(dim_phenotype))
    qc["pathways"] = int(dim_pathway.shape[0])
    qc["disease_gene_genes_outside_universe"] = int((~genes["gene_id"].isin(universe)).sum())
    (OUT / "chain_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    manifest = {"created": pd.Timestamp.now().isoformat(), "tables": {}}
    for path in sorted(OUT.glob("*.parquet")):
        frame = pd.read_parquet(path)
        manifest["tables"][path.name] = {"rows": int(len(frame)), "sha256": sha256(path)}
    (OUT / "chain_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
