"""R4 前置：保存 100 次扰动的原始划分（原 34 号只存了共现结果）。

输出：outputs/partitions/fold_N/partitions.parquet
列：run, disease_id, module_index, gene_id
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.discovery.baseline import load_v0_models, stable_seed  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

BATCH = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    log = JsonlLogger(paths.out("qc") / "partitions.log", "87_save_partitions")
    started = time.time()
    out_dir = paths.out("partitions", f"fold_{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(paths.v0_root / "src"))
    v0_rwr = importlib.import_module("phenotype_network_v0.models.rwr")
    detect, _ = load_v0_models(paths.v0_root)

    node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{args.fold}/node_map.parquet"))
    gene_rows = node_map.loc[node_map["node_type"].eq("gene")]
    gene_ids = gene_rows["node_id"].astype(str).tolist()
    phenotypes = pd.read_parquet(paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
                                 columns=["disease_id", "hpo_id", "frequency_weight"])
    phenotypes["disease_id"] = phenotypes["disease_id"].astype(str)
    hpo_ic = pd.read_parquet(paths.v0(f"data/folds/fold_{args.fold}/train_hpo_ic.parquet"))
    alpha = float(json.loads(paths.v0(f"outputs/rankings/fold_{args.fold}/rwr/rwr_qc.json")
                             .read_text(encoding="utf-8"))["selected_alpha"])
    from scipy.sparse import load_npz
    transition = load_npz(paths.v0(f"data/graphs/fold_{args.fold}/transition_matrix.npz")).tocsr()
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])

    test_ids = [line.strip() for line in
                paths.v0(f"data/folds/fold_{args.fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    seeds, _ = v0_rwr.build_seed_matrix(tuple(test_ids), phenotypes, hpo_ic, node_map)

    rows = []
    for index, disease in enumerate(test_ids):
        base = seeds[:, index]
        nonzero = np.flatnonzero(base > 0)
        if len(nonzero) == 0:
            continue
        draw = 0
        while draw < args.runs:
            count = min(BATCH, args.runs - draw)
            batch = np.zeros((seeds.shape[0], count))
            for b in range(count):
                chosen = rng.choice(nonzero, size=max(1, int(round(0.10 * len(nonzero)))), replace=False)
                vec = base.copy()
                vec[chosen] = 0.0
                total = vec.sum()
                batch[:, b] = vec / total if total > 0 else base
            result = v0_rwr.run_rwr(transition, batch, alpha=alpha)
            scores = result.scores[gene_rows["node_index"].to_numpy(dtype=int), :]
            for b in range(count):
                ranking = np.argsort(-scores[:, b], kind="stable")[:200]
                candidate = tuple(gene_ids[i] for i in ranking)
                detections = detect(candidate, ppi, seed=stable_seed(args.seed, args.fold, disease, draw + b))
                for module_index, module in enumerate(detections):
                    for gene in module:
                        rows.append({"run": draw + b, "disease_id": disease,
                                     "module_index": module_index, "gene_id": str(gene)})
            draw += count
        log.info("disease_done", disease=disease)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "partitions.parquet", index=False)
    summary = {"fold": args.fold, "diseases": len(test_ids), "runs": args.runs,
               "rows": int(len(table)), "seconds": round(time.time() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
