"""跨系统阳性对照：这条管线在"区分得开的疾病"上，到底能不能区分？

做法（不用标签）：
    1. 分别取四个疾病系统（心血管、神经、免疫、代谢）的疾病，各 10 个；
    2. 用同一张 fold 0 图、同一套 RWR + top-200 + Leiden 造模块；
    3. 比较系统内与系统间的模块 Jaccard。

如果系统间 Jaccard 明显低于系统内，说明管线有区分能力，心血管内部的高重叠是疾病同质造成的；
如果系统间也一样高，说明问题在管线/网络，换数据集也救不了。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pronto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

ROOTS = {
    "cardiovascular": "MONDO:0004995",
    "nervous": "MONDO:0005071",
    "immune": "MONDO:0005046",
    "metabolic": "MONDO:0005066",
}


def mondo_descendants(ontology: pronto.Ontology, root: str) -> set[str]:
    if root not in ontology:
        return set()
    return {term.id for term in ontology[root].subclasses(with_self=True) if not term.obsolete}


def map_diseases(ontology: pronto.Ontology, disease_ids: set[str]) -> dict[str, str]:
    """把 OMIM/ORPHA 等原始 ID 映射到 MONDO，用 xref 精确匹配。"""
    lookup: dict[str, str] = {}
    for term in ontology.terms():
        if not term.id.startswith("MONDO:") or term.obsolete:
            continue
        for xref in term.xrefs:
            lookup.setdefault(str(xref.id), term.id)
    return {d: lookup[d] for d in disease_ids if d in lookup}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--per-system", type=int, default=10)
    parser.add_argument("--min-phenotypes", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "cross_system.log", "11_cross_system")
    started = time.time()
    out_dir = paths.out("evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    hpoa = pd.read_csv(
        paths.v0("data/raw/hpo/phenotype.hpoa"), sep="\t", comment="#", low_memory=False,
        dtype=str,
    )
    hpoa = hpoa.rename(columns={c: c.strip() for c in hpoa.columns})
    hpoa = hpoa[["database_id", "hpo_id", "aspect"]].dropna()
    hpoa["database_id"] = hpoa["database_id"].astype(str)
    hpoa["hpo_id"] = hpoa["hpo_id"].astype(str)

    ontology = pronto.Ontology(str(paths.v0("data/raw/mondo/mondo.obo")), encoding="utf-8")
    species = map_diseases(ontology, set(hpoa["database_id"]))
    hpoa["mondo_id"] = hpoa["database_id"].map(species)
    hpoa = hpoa.dropna(subset=["mondo_id"])

    rng = np.random.default_rng(args.seed)
    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    detect, _ = load_v0_models(paths.v0_root)
    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    phenotype_index = {
        str(row.node_id): int(row.node_index)
        for row in node_map.loc[node_map["node_type"].eq("phenotype")].itertuples(index=False)
    }
    gene_rows = node_map.loc[node_map["node_type"].eq("gene"), "node_index"].to_numpy(dtype=int)
    row_of_gene = {str(v): i for i, v in enumerate(node_map.loc[node_map["node_type"].eq("gene"), "node_id"])}
    align = np.array([row_of_gene[g] for g in matrix.gene_ids])
    transition = load_npz = None
    from scipy.sparse import load_npz as _load_npz

    transition = _load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(
        paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json").read_text(encoding="utf-8")
    )["selected_alpha"])
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    import importlib as _il
    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = _il.import_module("phenotype_network_v0.models.rwr")

    module_sets: dict[str, dict[str, set[str]]] = {}
    coverage_rows = []

    for system, root in ROOTS.items():
        descendants = mondo_descendants(ontology, root)
        frame = hpoa.loc[hpoa["mondo_id"].isin(descendants)]
        by_disease = frame.groupby("mondo_id")["hpo_id"].apply(lambda s: sorted(set(s)))
        candidates = [d for d, terms in by_disease.items() if len(terms) >= args.min_phenotypes]
        if not candidates:
            log.error("no_candidates", system=system, root=root)
            continue
        picked = list(rng.choice(np.array(sorted(candidates)), size=min(args.per_system, len(candidates)), replace=False))
        module_sets[system] = {}

        for disease in picked:
            terms = [t for t in by_disease[disease] if t in phenotype_index]
            usable = [t for t in terms if t in ic_lookup]
            coverage_rows.append({
                "system": system,
                "disease_id": disease,
                "n_terms": len(terms),
                "n_with_ic": len(usable),
            })
            if len(usable) < args.min_phenotypes:
                continue
            weights = np.array([ic_lookup[t] for t in usable], dtype=float)
            weights /= weights.sum()
            seeds = np.zeros((len(node_map), 1), dtype=float)
            for term, weight in zip(usable, weights, strict=True):
                seeds[phenotype_index[term], 0] = weight
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            pct = result.scores[gene_rows, 0]
            order = np.argsort(-pct, kind="stable")[: args.top_k]
            genes = [matrix.gene_ids[i] for i in order]
            modules = detect(genes, ppi, seed=abs(hash((system, disease))) % (2**31))
            union = set().union(*[set(m) for m in modules]) if modules else set()
            module_sets[system][disease] = union
        log.info("system_done", system=system, diseases=len(module_sets[system]))

    def pair_stats(sets: dict[str, set[str]]) -> dict:
        keys = sorted(sets)
        values = [jaccard(sets[a], sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
        if not values:
            return {"pairs": 0}
        values = np.array(values)
        return {"pairs": len(values), "median": round(float(np.median(values)), 4),
                "q75": round(float(np.quantile(values, 0.75)), 4), "max": round(float(values.max()), 4)}

    within = {system: pair_stats(sets) for system, sets in module_sets.items()}
    within_pairs: list[float] = []
    for sets in module_sets.values():
        keys = sorted(sets)
        within_pairs.extend(jaccard(sets[a], sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:])
    across_pairs: list[float] = []
    across = []
    systems = sorted(module_sets)
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            values = [jaccard(sa, sb) for sa in module_sets[a].values() for sb in module_sets[b].values()]
            if values:
                values = np.array(values)
                across_pairs.extend(values.tolist())
                across.append({"pair": f"{a}|{b}", "pairs": len(values),
                               "median": round(float(np.median(values)), 4),
                               "q75": round(float(np.quantile(values, 0.75)), 4)})

    summary = {
        "fold": args.fold,
        "per_system": args.per_system,
        "within_system": within,
        "across_system": across,
        "module_sizes": {s: round(float(np.median([len(v) for v in sets.values()])), 1) if sets else None
                         for s, sets in module_sets.items()},
        "ic_coverage": {
            "median_terms": int(np.median([r["n_terms"] for r in coverage_rows])) if coverage_rows else None,
            "median_with_ic": int(np.median([r["n_with_ic"] for r in coverage_rows])) if coverage_rows else None,
        },
        "within_vs_across": {
            "within_median": round(float(np.median(within_pairs)), 4) if within_pairs else None,
            "across_median": round(float(np.median(across_pairs)), 4) if across_pairs else None,
            "mannwhitney_p": float(__import__("scipy.stats", fromlist=["mannwhitneyu"]).mannwhitneyu(
                within_pairs, across_pairs, alternative="greater").pvalue) if within_pairs and across_pairs else None,
        },
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "cross_system_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(coverage_rows).to_csv(out_dir / "cross_system_coverage.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
