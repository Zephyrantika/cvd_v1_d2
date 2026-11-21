"""按文档定义的 core Jaccard：把 100 次扰动对半分，两半各自定核心（freq≥0.8），
比较两个核心之间的 Jaccard。重复多次随机对半，取均值。

与「整模块 Jaccard」不同，核心通常只有 8–15 个基因，规模效应小得多。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DIR = Path(__file__).resolve().parents[1]
TAU = 0.8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--splits", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    parts = pd.read_parquet(DIR / f"outputs/partitions/fold_{args.fold}/partitions.parquet")
    parts["gene_id"] = parts["gene_id"].astype(str)
    parts["disease_id"] = parts["disease_id"].astype(str)
    modules_by_run: dict[str, dict[int, list[set[str]]]] = {}
    for (disease, run), group in parts.groupby(["disease_id", "run"]):
        modules_by_run.setdefault(disease, {})[int(run)] = [
            set(block["gene_id"]) for _, block in group.groupby("module_index")]

    mods = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/modules.parquet")
    mem = pd.read_parquet(DIR / f"outputs/modules_doc/fold_{args.fold}/members.parquet")
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()

    refined = pd.read_parquet(DIR / f"outputs/lam_refine/fold_{args.fold}_lam1.0_mu1.0/lam_refine_members.parquet")
    refined["gene_id"] = refined["gene_id"].astype(str)
    r1 = pd.read_parquet(DIR / f"outputs/pareto_r1/members_fold{args.fold}.parquet")
    r1["gene_id"] = r1["gene_id"].astype(str)

    variants = {
        "baseline": {d: set(list(base_sets.get(m, set()))[:25]) for d, m in largest.items()},
        "refined": refined.groupby("disease_id")["gene_id"].apply(set).to_dict(),
        "r1": r1.groupby("disease_id")["gene_id"].apply(set).to_dict(),
    }

    def core(module: set[str], runs: list[int], disease: str) -> set[str]:
        counts = {g: 0 for g in module}
        for run in runs:
            best, best_j = None, -1.0
            for other in modules_by_run[disease][run]:
                inter = len(module & other)
                if not inter:
                    continue
                j = inter / len(module | other)
                if j > best_j:
                    best, best_j = other, j
            if best is None:
                continue
            for g in module:
                if g in best:
                    counts[g] += 1
        threshold = TAU * max(len(runs), 1)
        return {g for g, c in counts.items() if c >= threshold}

    rows = []
    for disease, mapping in variants.items():
        pass
    for disease in sorted(variants["baseline"]):
        if disease not in modules_by_run:
            continue
        runs = sorted(modules_by_run[disease])
        if len(runs) < 20:
            continue
        for name, sets in variants.items():
            module = sets.get(disease)
            if not module or len(module) < 5:
                continue
            values = []
            for _ in range(args.splits):
                shuffled = list(rng.permutation(runs))
                half = len(shuffled) // 2
                c1 = core(module, shuffled[:half], disease)
                c2 = core(module, shuffled[half:], disease)
                union = c1 | c2
                values.append(len(c1 & c2) / len(union) if union else 0.0)
            rows.append({"disease_id": disease, "variant": name,
                         "core_jaccard": float(np.mean(values))})

    table = pd.DataFrame(rows)
    out = DIR / f"outputs/core_jaccard/fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "core_jaccard.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant", values="core_jaccard")
    summary = {}
    for a, b in (("baseline", "refined"), ("baseline", "r1")):
        pair = pivot[[a, b]].dropna()
        summary[f"{a}_vs_{b}"] = {"n": int(len(pair)),
                                  "a": round(float(pair[a].mean()), 4),
                                  "b": round(float(pair[b].mean()), 4),
                                  "wins": int((pair[b] > pair[a]).sum()),
                                  "p": float(wilcoxon(pair[b], pair[a]).pvalue)}
    payload = {"fold": args.fold, "tau": TAU, "splits": args.splits, "summary": summary}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
