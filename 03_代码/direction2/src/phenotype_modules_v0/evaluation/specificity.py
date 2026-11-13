"""疾病特异性 z_spec：模块对当前疾病的传播分位，相对同一折内其它疾病的分布。

这是设计文档 03 第 4 节定义的量，不依赖随机表型查询，也不依赖已知基因：
    S(M,d)  = mean_{g in M} r_d(g)
    z_spec  = (S(M,d) - mean_{d' in D_bg} S(M,d')) / (sd + eps)
背景 D_bg 是同一折内除 d 及其同家族疾病之外的测试疾病。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phenotype_modules_v0.scores.provider import ScoreMatrix


@dataclass(frozen=True)
class SpecificityResult:
    module_id: str
    disease_id: str
    module_size: int
    score_self: float
    background_mean: float
    background_sd: float
    z_spec: float
    percentile_in_background: float
    n_background: int


def background_diseases(
    disease: str,
    fold_diseases: tuple[str, ...],
    family_by_disease: dict[str, str],
) -> tuple[str, ...]:
    family = family_by_disease.get(disease)
    return tuple(
        other
        for other in fold_diseases
        if other != disease and (family is None or family_by_disease.get(other) != family)
    )


def module_score(matrix: ScoreMatrix, percentiles: np.ndarray, disease_row: int, genes: list[str], gene_position: dict[str, int]) -> float:
    rows = [gene_position[g] for g in genes if g in gene_position]
    if not rows:
        return float("nan")
    return float(np.mean(percentiles[disease_row, rows]))


def specificity_for_module(
    matrix: ScoreMatrix,
    percentiles: np.ndarray,
    gene_position: dict[str, int],
    module_id: str,
    disease: str,
    genes: list[str],
    background: tuple[str, ...],
    family_by_disease: dict[str, str],
) -> SpecificityResult:
    disease_row = matrix.disease_index(disease)
    rows = [gene_position[g] for g in genes if g in gene_position]
    score_self = float(np.mean(percentiles[disease_row, rows]))
    background_rows = [matrix.disease_index(d) for d in background if d in matrix.disease_ids]
    background_scores = np.array([np.mean(percentiles[row, rows]) for row in background_rows])
    mean = float(background_scores.mean())
    sd = float(background_scores.std(ddof=1)) if len(background_scores) > 1 else 0.0
    z = (score_self - mean) / (sd + 1e-9)
    percentile = float((background_scores < score_self).mean()) if len(background_scores) else float("nan")
    return SpecificityResult(
        module_id=module_id,
        disease_id=disease,
        module_size=len(rows),
        score_self=score_self,
        background_mean=mean,
        background_sd=sd,
        z_spec=z,
        percentile_in_background=percentile,
        n_background=len(background_rows),
    )

def phenotype_sets(phenotypes: pd.DataFrame) -> dict[str, set[str]]:
    """疾病 -> HPO 术语集合，用于按表型相似度排背景。"""
    grouped = phenotypes.assign(
        disease_id=phenotypes["disease_id"].astype(str),
        hpo_id=phenotypes["hpo_id"].astype(str),
    ).groupby("disease_id")["hpo_id"]
    return {str(disease): set(terms) for disease, terms in grouped}


def background_by_similarity(
    disease: str,
    fold_diseases: tuple[str, ...],
    sets: dict[str, set[str]],
    threshold: float = 0.5,
) -> tuple[str, ...]:
    """排除与当前疾病表型 Jaccard >= threshold 的疾病，其余作背景。

    家族划分在本数据上非常粗（fold 0 里 91 个测试疾病只有 17 个家族，
    其中一个家族占 75 个），按家族排除会把背景压到 16 个。
    """
    target = sets.get(disease, set())
    background = []
    for other in fold_diseases:
        if other == disease:
            continue
        candidate = sets.get(other, set())
        union = target | candidate
        jaccard = len(target & candidate) / len(union) if union else 0.0
        if jaccard < threshold:
            background.append(other)
    return tuple(background)
