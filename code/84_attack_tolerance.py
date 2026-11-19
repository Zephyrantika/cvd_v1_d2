"""R2 模块级攻击容忍度：针对模块内加权度最高的 3 个成员删除，看剩余子图还剩多少。

对照：随机删 3 个成员 100 次取均值（随机损失）。
指标：
  lcc_keep   删除后最大连通分量占剩余成员的比例
  z_keep     剩余成员的 z_spec 与原始模块 z_spec 的比值
比较：基线模块 vs 精修模块（λ=1、μ=1），五折合并。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

BACKDROP = 20
DROP = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "attack_tolerance.log", "84_attack_tolerance")
    out_dir = paths.out("attack", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", f"fold_{args.fold}", "score_matrix.npz"))
    percentiles = matrix.percentiles().T          # (基因 × 疾病)
    position = {g: i for i, g in enumerate(matrix.gene_ids)}

    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        adjacency.setdefault(a, {})[b] = float(w)
        adjacency.setdefault(b, {})[a] = float(w)

    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    phenotypes["hpo_id"] = phenotypes["hpo_id"].astype(str)
    terms_of = phenotypes.groupby("disease_id")["hpo_id"].apply(set).to_dict()
    ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    ic_lookup = dict(zip(ic["hpo_id"].astype(str), ic["ic"].astype(float)))
    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id"])
    family = dict(zip(fam["disease_id"].astype(str), fam["disease_family_id"].astype(str)))

    mods = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "modules.parquet"))
    mem = pd.read_parquet(paths.out("modules_doc", f"fold_{args.fold}", "members.parquet"))
    mem["gene_id"] = mem["gene_id"].astype(str)
    largest = mods.sort_values("size", ascending=False).groupby("disease_id").first()["module_id"].to_dict()
    base_sets = mem.groupby("module_id")["gene_id"].apply(set).to_dict()
    ref = pd.read_parquet(paths.out("lam_refine", f"fold_{args.fold}_lam1.0_mu1.0", "lam_refine_members.parquet"))
    ref["gene_id"] = ref["gene_id"].astype(str)
    ref_sets = ref.groupby("disease_id")["gene_id"].apply(set).to_dict()

    def background(disease: str) -> list[str]:
        scored = []
        for other in matrix.disease_ids:
            if other == disease or family.get(other) == family.get(disease):
                continue
            shared = terms_of.get(disease, set()) & terms_of.get(other, set())
            sim = 0.0 if not shared else sum(ic_lookup.get(t, 0.0) for t in shared) / (
                np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(disease, set())))
                * np.sqrt(sum(ic_lookup.get(t, 0.0) for t in terms_of.get(other, set()))) + 1e-9)
            scored.append((sim, other))
        return [d for _, d in sorted(scored, reverse=True)[:BACKDROP]]

    def z_of(disease: str, genes: set[str]) -> float:
        idx = [position[g] for g in genes if g in position]
        if len(idx) < 3:
            return np.nan
        col = matrix.disease_index(disease)
        own = float(percentiles[idx, col].mean())
        back = np.array([percentiles[idx, matrix.disease_index(b)].mean() for b in background(disease)])
        if len(back) < 2 or back.std(ddof=1) == 0:
            return np.nan
        return float((own - back.mean()) / back.std(ddof=1))

    def lcc_fraction(genes: set[str]) -> float:
        if not genes:
            return 0.0
        seen, best = set(), 0
        for start in genes:
            if start in seen:
                continue
            stack, size = [start], 0
            seen.add(start)
            while stack:
                node = stack.pop()
                size += 1
                for nb in adjacency.get(node, {}):
                    if nb in genes and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            best = max(best, size)
        return best / len(genes)

    def weighted_degree(genes: set[str]) -> dict[str, float]:
        return {g: sum(w for n, w in adjacency.get(g, {}).items() if n in genes) for g in genes}

    rows = []
    for disease, module_id in largest.items():
        if disease not in ref_sets:
            continue
        for label, genes in (("baseline", set(list(base_sets.get(module_id, set()))[:25])),
                             ("refined", ref_sets[disease])):
            if len(genes) < 10:
                continue
            z0 = z_of(disease, genes)
            wd = weighted_degree(genes)
            targets = sorted(wd, key=lambda g: -wd[g])[:DROP]
            after = genes - set(targets)
            targeted = {"lcc": lcc_fraction(after), "z_keep": z_of(disease, after) / z0 if z0 else np.nan}
            rand_lcc, rand_z = [], []
            for _ in range(args.draws):
                drop = set(rng.choice(sorted(genes), size=DROP, replace=False).tolist())
                rest = genes - drop
                rand_lcc.append(lcc_fraction(rest))
                value = z_of(disease, rest)
                rand_z.append(value / z0 if z0 else np.nan)
            rows.append({
                "disease_id": disease, "variant": label, "size": len(genes),
                "lcc_targeted": targeted["lcc"], "lcc_random": float(np.mean(rand_lcc)),
                "lcc_excess": targeted["lcc"] - float(np.mean(rand_lcc)),
                "z_targeted": targeted["z_keep"], "z_random": float(np.nanmean(rand_z)),
                "z_excess": targeted["z_keep"] - float(np.nanmean(rand_z)),
                "hub": float(2 * max(wd.values()) / sum(wd.values())) if sum(wd.values()) else 1.0,
            })
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "attack.parquet", index=False)
    pivot = table.pivot_table(index="disease_id", columns="variant",
                              values=["lcc_excess", "z_excess", "lcc_targeted", "z_targeted"]).dropna()
    payload = {"fold": args.fold, "diseases": int(len(pivot)), "drops": DROP}
    for metric in ["lcc_excess", "z_excess"]:
        a = pivot[(metric, "baseline")]
        b = pivot[(metric, "refined")]
        payload[metric] = {
            "baseline_mean": round(float(a.mean()), 4),
            "refined_mean": round(float(b.mean()), 4),
            "refined_better": int((b > a).sum()),
            "n": int(len(a)),
            "p": float(wilcoxon(b, a).pvalue),
        }
    (out_dir / "attack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
