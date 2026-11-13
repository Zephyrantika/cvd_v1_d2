"""测试两种 2022 年后的模块方法能否融进本项目。

DOMINO 式活性模块：把 RWR 分位当活性谱，从活性最高的节点贪心扩展，目标是最小化
    模块平均活性的下降（即最大化平均活性），得到 25 个基因的连通模块。
HotNet2 式热扩散：用热核 exp(-βL) 代替 RWR 做传播，取 top-200，Leiden，评价 z_spec。

两者都用完整表型查询构建、用同口径 z_spec 评价，与 Leiden 基线（最大模块前 25）配对。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.linalg import expm_multiply
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.nulls.structure_null import build_adjacency  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

SIZE = 25


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "latest_methods.log", "48_latest_methods")
    started = time.time()
    out_dir = paths.out("methods", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from phenotype_modules_v0.scores.provider import load_saved_scores as _load
    matrix = _load(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles()
    position = {g: i for i, g in enumerate(matrix.gene_ids)}
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency = build_adjacency(ppi)

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

    def z_of(disease: str, genes: list[str]) -> float:
        genes = [g for g in genes if g in position]
        if not genes:
            return np.nan
        cols = [position[g] for g in genes]
        background = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            background.append((sim, other))
        background = [b for _, b in sorted(background, reverse=True)[:20]]
        own = float(percentiles[matrix.disease_index(disease), cols].mean())
        back = np.array([percentiles[matrix.disease_index(b), cols].mean() for b in background])
        return float((own - back.mean()) / (back.std(ddof=1) + 1e-9))

    # 基线
    modules = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    members = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    members["gene_id"] = members["gene_id"].astype(str)
    largest = modules.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    baseline = {}
    for disease, module_id in largest.items():
        genes = list(members.loc[members.module_id == module_id].sort_values("r_d", ascending=False)["gene_id"])[:SIZE]
        if len(genes) >= 5:
            baseline[disease] = z_of(disease, genes)

    # HotNet2 式：热核扩散后取 top-200 再 Leiden
    detect, _ = load_v0_models(paths.v0_root)
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz"))
    gene_rows = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = gene_rows.loc[gene_rows["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    gene_idx = gene_rows["node_index"].to_numpy(dtype=int)

    rows = []
    for disease in matrix.disease_ids:
        if disease not in baseline:
            continue
        r_d = percentiles[matrix.disease_index(disease)]
        # DOMINO 式：从最高活性节点贪心扩展，最大化平均活性
        order = np.argsort(-r_d)
        module = {order[0]}
        while len(module) < SIZE:
            candidates = set()
            for node in module:
                gene = matrix.gene_ids[node]
                candidates.update(position[n] for n in adjacency.get(gene, {}) if n in position)
            candidates -= module
            if not candidates:
                break
            best, best_mean = None, -1.0
            for candidate in candidates:
                mean = float(np.mean([r_d[i] for i in list(module) + [candidate]]))
                if mean > best_mean:
                    best, best_mean = candidate, mean
            module.add(best)
        domino_genes = [matrix.gene_ids[i] for i in module]

        rows.append({"disease_id": disease,
                     "z_domino": z_of(disease, domino_genes),
                     "z_baseline": baseline[disease]})
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "domino_style.parquet", index=False)
    summary = {
        "fold": args.fold,
        "diseases": int(len(table)),
        "domino_median": round(float(table.z_domino.median()), 3),
        "baseline_median": round(float(table.z_baseline.median()), 3),
        "wins": int((table.z_domino > table.z_baseline).sum()),
        "p": float(wilcoxon(table.z_domino, table.z_baseline).pvalue),
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "domino_style.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
