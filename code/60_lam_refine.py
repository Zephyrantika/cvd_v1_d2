"""带稳定性惩罚的 z_spec 精修：目标 f = z_spec + λ·mean(共识频率)。

纯 z 精修（59 号）在 s7–s8 才勉强越过基线的 z，复现率已开始掉。
这里给每个成员加上共识频率（0–1，反映它在 100 次扰动里出现的比例）作为奖励，
让换人只在「换来的人也有共识支持」时才划算，试图在更高 z 上保住复现率。
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
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25
BACKDROP = 20
RUNS = 10
COVER = 0.6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--ppi", default="700")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(2026)
    log = JsonlLogger(paths.out("qc") / "lam_refine.log", "60_lam_refine")
    started = time.time()
    suffix = ("" if args.topk == 200 else f"_top{args.topk}") + ("" if args.ppi == "700" else f"_ppi{args.ppi}")
    out_dir = paths.out("lam_refine", f"fold_{args.fold}_lam{args.lam}_mu{args.mu}_size{args.size}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0(f"data/processed/string_gene_edges_{args.ppi}.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

    cons_mods = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_modules.parquet"))
    cons_mem = pd.read_parquet(paths.out("consensus", f"fold_{args.fold}", "consensus_members.parquet"))
    cons_mem["gene_id"] = cons_mem["gene_id"].astype(str)
    largest = cons_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    cons_sets = cons_mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    frequency = {(r.module_id, str(r.gene_id)): float(r.frequency) for r in cons_mem.itertuples(index=False)}

    base_mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    base_mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    base_mem["gene_id"] = base_mem["gene_id"].astype(str)
    base_largest = base_mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = base_mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(hpo_ic["hpo_id"].astype(str), hpo_ic["ic"].astype(float)))
    folds_frame = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                                  columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds_frame["disease_id"].astype(str), folds_frame["disease_family_id"].astype(str)))
    backgrounds = {}
    for disease in matrix.disease_ids:
        scored = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        backgrounds[disease] = [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_of(disease: str, genes: set[str]) -> float:
        cols = [position[g] for g in genes if g in position]
        if len(cols) < 5:
            return np.nan
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in backgrounds[disease]])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    def hub_of(genes: set[str]) -> float:
        member = set(genes)
        weighted = {}
        for gene in member:
            total = 0.0
            for neighbour, weight in adjacency.get(gene, {}).items():
                if neighbour in member:
                    total += weight
            weighted[gene] = total
        overall = sum(weighted.values())
        return float(2.0 * max(weighted.values(), default=0.0) / overall) if overall else 1.0

    def objective(disease: str, module_id: str, genes: set[str]) -> float:
        z = z_of(disease, genes)
        freq = np.mean([frequency.get((module_id, g), 0.0) for g in genes]) if genes else 0.0
        return float(z + args.lam * freq - args.mu * hub_of(genes))

    refined, base_sel, cons_sel = {}, {}, {}
    member_rows = []
    for disease in matrix.disease_ids:
        b_id, c_id = base_largest.get(disease), largest.get(disease)
        if not b_id or not c_id:
            continue
        candidate = set(matrix.top_k(disease, args.topk))
        base_genes = sorted(base_sets.get(b_id, set()), key=lambda g: -percentiles[matrix.disease_index(disease), position[g]])[:args.size]
        cons_genes = sorted(cons_sets.get(c_id, set()), key=lambda g: -percentiles[matrix.disease_index(disease), position[g]])[:args.size]
        if len(base_genes) < 5 or len(cons_genes) < 5:
            continue
        base_sel[disease] = set(base_genes)
        cons_sel[disease] = set(cons_genes)
        current = set(cons_genes)
        for _ in range(args.steps):
            neighbours = set()
            for gene in current:
                neighbours.update(adjacency.get(gene, {}).keys())
            pool = (neighbours & candidate) - current
            if not pool:
                break
            weakest = sorted(current, key=lambda g: percentiles[matrix.disease_index(disease), position[g]])[:5]
            best_move, best_value = None, objective(disease, c_id, current)
            for out_gene in weakest:
                for in_gene in sorted(pool, key=lambda g: -percentiles[matrix.disease_index(disease), position[g]])[:10]:
                    trial = (current - {out_gene}) | {in_gene}
                    value = objective(disease, c_id, trial)
                    if value > best_value:
                        best_move, best_value = (out_gene, in_gene), value
            if best_move is None:
                break
            current = (current - {best_move[0]}) | {best_move[1]}
        refined[disease] = current
        for gene in current:
            member_rows.append({"module_id": f"{disease}:lam{args.lam}", "disease_id": disease, "gene_id": gene})
        log.info("refined", disease=disease)

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    hpo_node = dict(zip(node_map.loc[node_map["node_type"].eq("phenotype"), "node_id"].astype(str),
                        node_map.loc[node_map["node_type"].eq("phenotype"), "node_index"].astype(int)))
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()

    rows = []
    for disease in base_sel:
        terms = sorted(terms_of.get(disease, set()))
        if len(terms) < 5:
            continue
        partitions = []
        for run in range(args.runs):
            kept = [t for t in terms if rng.random() > 0.10]
            indices = [hpo_node[t] for t in kept if t in hpo_node]
            if not indices:
                continue
            seeds = np.zeros((transition.shape[0], 1))
            unique = np.unique(indices)
            seeds[unique, 0] = 1.0 / len(unique)
            result = v0_rwr.run_rwr(transition, seeds, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), 0]
            top = np.argsort(-scores)[:200]
            detections = detect(tuple(gene_ids[i] for i in top), ppi,
                                seed=stable_seed(2026, args.fold, disease, run))
            partitions.append([set(m) for m in detections])
        for name, members in (("baseline", base_sel[disease]), ("consensus", cons_sel[disease]),
                              ("refined", refined[disease])):
            covered = sum(1 for p in partitions if any(len(members & m) / len(members) >= COVER for m in p))
            rows.append({"disease_id": disease, "variant": name, "z": z_of(disease, members),
                         "replication": covered / max(len(partitions), 1)})

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "lam_refine.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out_dir / "lam_refine_members.parquet", index=False)
    summary = table.groupby("variant").agg(z=("z", "median"), rep=("replication", "mean"),
                                           n=("disease_id", "size")).round(3).reset_index()
    (out_dir / "lam_refine.json").write_text(
        json.dumps({"fold": args.fold, "lam": args.lam, "steps": args.steps,
                    "table": summary.to_dict(orient="records"),
                    "seconds": round(time.time() - started, 2)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(summary.to_string(index=False))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
