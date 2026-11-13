"""可靠性框架的预测效度：用模块的可靠性维度去预测它是否被独立证据支持。

标签：模块基因的 GWAS 富集 z > 2（来自 31 号脚本，未参与模块构建）。
特征：z_spec、density、conductance、hub、stability、size、core 占比、p_spec。
做法：按疾病家族分组的 5 折交叉验证，比较单特征与组合特征的 AUC。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

FEATURES = ["z_spec", "density", "conductance", "hub", "stability", "size", "core_frac", "p_spec"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    log = JsonlLogger(paths.out("qc") / "predictive_validity.log", "43_predictive_validity")
    started = time.time()
    out_dir = paths.out("predictive")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = pd.read_parquet(paths.out("modules_doc", "modules.parquet"))
    external = pd.read_parquet(paths.out("external_evidence", "module_external_evidence.parquet"))
    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "disease_family_id"])
    family = dict(zip(folds["disease_id"].astype(str), folds["disease_family_id"].astype(str)))

    frame = modules.merge(external[["module_id", "gwas_z", "expr_vs_random"]], on="module_id", how="left")
    frame["core_frac"] = frame["core_size"] / frame["size"].clip(lower=1)
    frame["family"] = frame["disease_id"].astype(str).map(family)
    frame = frame.dropna(subset=["gwas_z", "family"])
    frame["label"] = (frame["gwas_z"] > 2).astype(int)
    log.info("rows", n=len(frame), positives=int(frame["label"].sum()))

    groups = frame["family"].to_numpy()
    y = frame["label"].to_numpy()
    X = frame[FEATURES].to_numpy()

    def cv_auc(columns: list[str]) -> float:
        matrix = frame[columns].to_numpy()
        predictions = np.zeros(len(frame))
        for train, test in GroupKFold(n_splits=5).split(matrix, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            model.fit(matrix[train], y[train])
            predictions[test] = model.predict_proba(matrix[test])[:, 1]
        return float(roc_auc_score(y, predictions))

    single = {column: round(cv_auc([column]), 3) for column in FEATURES}
    combined = round(cv_auc(FEATURES), 3)
    continuous = {
        column: round(float(spearmanr(frame[column], frame["gwas_z"]).statistic), 3) for column in FEATURES
    }
    log.info("cv_done", combined=combined)

    summary = {
        "modules": int(len(frame)),
        "positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 3),
        "single_feature_auc": single,
        "combined_auc": combined,
        "best_single": max(single, key=single.get),
        "best_single_auc": max(single.values()),
        "spearman_with_gwas_z": continuous,
        "seconds": round(time.time() - started, 2),
    }
    (out_dir / "predictive_validity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    frame[["module_id", "disease_id", "family"] + FEATURES + ["gwas_z", "label"]].to_parquet(
        out_dir / "predictive_validity_frame.parquet", index=False
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
