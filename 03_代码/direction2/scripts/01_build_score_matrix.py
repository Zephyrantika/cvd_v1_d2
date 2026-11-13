"""M3-1：把 V0 的排名表变成疾病×基因分数矩阵。

产出：
    outputs/scores/fold_N/score_matrix.npz   分数、排名、疾病与基因顺序
    outputs/qc/score_matrix_fold_N.json      形状、校验结果、耗时

分数矩阵只做格式转换，不改数值：scores 与 rankings.parquet 的 raw_rwr_score 逐元素相等。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402
from phenotype_modules_v0.scores.provider import load_fold_scores, save_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    qc_dir = paths.out("qc")
    qc_dir.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(qc_dir / "score_matrix.log", "01_build_score_matrix")

    failures = []
    for fold in folds:
        started = time.time()
        matrix = load_fold_scores(paths.v0_root, fold)

        test_ids = {
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        checks = {
            "diseases_match_test": set(matrix.disease_ids) == test_ids,
            "n_diseases": len(matrix.disease_ids),
            "n_genes": len(matrix.gene_ids),
            "grid_complete": matrix.scores.shape == (len(matrix.disease_ids), len(matrix.gene_ids)),
            "no_nan": bool(np.isfinite(matrix.scores).all()),
            "rank_range": [int(matrix.ranks.min()), int(matrix.ranks.max())],
        }
        percentiles = matrix.percentiles()
        checks["percentile_top_is_one"] = bool(np.allclose(percentiles.max(axis=1), 1.0))
        checks["percentile_bottom_is_zero"] = bool(np.allclose(percentiles.min(axis=1), 0.0))
        checks["percentile_monotone"] = bool(
            np.all(np.diff(np.sort(percentiles, axis=1), axis=1) >= -1e-6)
        )

        out_path = paths.out("scores", f"fold_{fold}", "score_matrix.npz")
        save_scores(matrix, out_path)
        ok = all(
            v for k, v in checks.items()
            if isinstance(v, bool)
        )
        if not ok:
            failures.append(fold)

        record = {
            "fold": fold,
            "path": str(out_path),
            "shape": list(matrix.scores.shape),
            "checks": checks,
            "seconds": round(time.time() - started, 2),
            "seed": args.seed,
        }
        (qc_dir / f"score_matrix_fold_{fold}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (log.info if ok else log.error)(
            "fold_done", fold=fold, shape=f"{matrix.scores.shape[0]}x{matrix.scores.shape[1]}",
            ok=ok, seconds=record["seconds"],
        )

    print(f"folds: {len(folds)}, failed: {failures or 'none'}")
    log.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
