"""给按 p 选出的模块补 hub / GO 并分级（p 与复现率已在 selected_byp.parquet 里）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "phenotype-modules-v0"
V0 = ROOT / "phenotype-network-v0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    base = DIR / f"outputs/pareto/byp_fold{args.fold}"
    sel = pd.read_parquet(base / "selected_byp.parquet")
    mem = pd.read_parquet(base / "selected_byp_members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    sets = mem.groupby("disease_id")["gene_id"].apply(set).to_dict()

    ppi = pd.read_parquet(V0 / "data/processed/string_gene_edges_700.parquet",
                          columns=["gene_a", "gene_b", "weight"])
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    def hub(genes: set[str]) -> float:
        weighted = {g: sum(w for n, w in adjacency.get(g, {}).items() if n in genes) for g in genes}
        total = sum(weighted.values())
        return float(2 * max(weighted.values(), default=0) / total) if total else 1.0

    go = pd.read_parquet(V0 / "data/processed/go_gene_annotations.parquet",
                         columns=["gene_id", "go_id", "evidence_code"])
    go["gene_id"] = go["gene_id"].astype(str)
    go = go[~go["evidence_code"].astype(str).isin({"IEA"})]
    term_genes: dict[str, set[str]] = {}
    gene_terms: dict[str, list[str]] = {}
    for gene, term in zip(go["gene_id"], go["go_id"].astype(str)):
        term_genes.setdefault(term, set()).add(gene)
        gene_terms.setdefault(gene, []).append(term)
    term_genes = {t: g for t, g in term_genes.items() if 10 <= len(g) <= 500}
    gene_terms = {g: [t for t in ts if t in term_genes] for g, ts in gene_terms.items()}

    def go_fdr(genes: set[str]) -> float:
        target = set(genes)
        terms = set()
        for gene in target:
            terms.update(gene_terms.get(gene, []))
        if not terms:
            return np.nan
        ordered = sorted(terms)
        k = np.array([len(term_genes[t]) for t in ordered])
        x = np.array([len(term_genes[t] & target) for t in ordered])
        return float(multipletests(hypergeom.sf(x - 1, 16992, k, len(target)), method="fdr_bh")[1].min())

    sel["hub"] = [hub(sets[d]) for d in sel.disease_id]
    sel["go_fdr"] = [go_fdr(sets[d]) for d in sel.disease_id]

    def grade(row) -> str:
        if row.p_spec <= 0.01 and row.replication >= 0.7 and row.hub <= 0.5 and row.go_fdr <= 0.05:
            return "A"
        if row.p_spec <= 0.05 and row.replication >= 0.6 and row.hub <= 0.5:
            return "B"
        return "C"

    sel["grade"] = [grade(r) for r in sel.itertuples(index=False)]
    sel.to_parquet(base / "graded.parquet", index=False)
    summary = {"fold": args.fold, "diseases": int(len(sel)),
               "A": int((sel.grade == "A").sum()), "B": int((sel.grade == "B").sum()),
               "C": int((sel.grade == "C").sum()),
               "frac_A": round(float((sel.grade == "A").mean()), 4),
               "z_median": round(float(sel.z.median()), 3),
               "rep_mean": round(float(sel.replication.mean()), 3)}
    (base / "graded_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
