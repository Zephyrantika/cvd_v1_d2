"""逐基因超额分位：把基因热度从特异性里扣掉。

r_d(g) 是基因 g 在疾病 d 下的分位；e(g,d) = r_d(g) - mean_{d'} r_d'(g)。
模块超额分 S(M,q) = mean_{g in M} e(g,q)。对任意查询 q（真实疾病或随机表型查询），
用同一折内背景疾病的 S(M,d') 做标准化：

    z_excess = (S(M,q) - mean_bg S) / (sd_bg S + eps)

选择效应（都是从 q 的 top-k 里切模块）在真实和随机两侧同构，所以两侧可比；
热度效应在 e 里已经逐基因减掉，不会再出现"冷门模块 z 更高"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gene_baseline(percentiles: np.ndarray) -> np.ndarray:
    """每个基因在全部疾病上的平均分位，作为它的热度基线。"""
    return percentiles.mean(axis=0)


def module_excess_by_disease(
    percentiles: np.ndarray,
    baseline: np.ndarray,
    gene_indices: list[int],
) -> np.ndarray:
    """返回长度 = 疾病数的数组：每个疾病下该模块的平均超额分位。"""
    if not gene_indices:
        return np.full(percentiles.shape[0], np.nan)
    excess = percentiles[:, gene_indices] - baseline[None, gene_indices]
    return excess.mean(axis=1)


@dataclass(frozen=True)
class ExcessResult:
    module_id: str
    disease_id: str
    module_size: int
    score_self: float
    background_mean: float
    background_sd: float
    z_excess: float
    p_empirical: float
    n_background: int


def excess_z(
    percentiles: np.ndarray,
    baseline: np.ndarray,
    gene_indices: list[int],
    self_row: int,
    background_rows: np.ndarray,
) -> tuple[float, float, float, float]:
    """返回 (S_self, 背景均值, 背景标准差, z)。"""
    scores = module_excess_by_disease(percentiles, baseline, gene_indices)
    self_score = float(scores[self_row])
    background = scores[background_rows]
    mean = float(np.mean(background))
    sd = float(np.std(background, ddof=1)) if len(background) > 1 else 0.0
    z = (self_score - mean) / (sd + 1e-9)
    return self_score, mean, sd, z


def empirical_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        return float("nan")
    return (1 + int((null_values >= observed).sum())) / (len(null_values) + 1)
