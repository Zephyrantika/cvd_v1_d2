"""随机表型零模型：抽匹配的表型查询，重跑 RWR，看模块的传播信号是否仍然特异地高。

匹配规则（本原型采用）：
    - 查询术语数量与原疾病一致；
    - 每个术语在同一个 IC 五分位内抽取；
    - 排除与原始任何术语在 HPO 图上距离 ≤2 的术语；
    - 权重沿用原始频率权重的多重集（排序后逐个分配）。

已知简化：没有匹配 aspect（P/C/I/M），因为 hpo_nodes 里没有 aspect 字段；
这一点写进报告的限制节，正式版本要补。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


def term_neighborhoods(edges: pd.DataFrame, hops: int = 2) -> dict[str, set[str]]:
    """HPO 图上的无向 hops 跳邻域（父子关系两边都走）。"""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for child, parent in zip(edges["child_hpo_id"].astype(str), edges["parent_hpo_id"].astype(str), strict=True):
        adjacency[child].add(parent)
        adjacency[parent].add(child)

    cache: dict[str, set[str]] = {}

    def expand(term: str) -> set[str]:
        if term in cache:
            return cache[term]
        seen = {term}
        frontier = {term}
        for _ in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= adjacency.get(node, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        cache[term] = seen
        return seen

    for term in adjacency:
        expand(term)
    return cache


def ic_bins(ic: pd.Series, n_bins: int = 5) -> tuple[dict[str, int], dict[int, list[str]]]:
    values = ic.astype(float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    labels = np.digitize(values.to_numpy(), edges[1:-1], right=False)
    term_bin = {str(term): int(label) for term, label in zip(values.index, labels, strict=True)}
    pool: dict[int, list[str]] = defaultdict(list)
    for term, label in term_bin.items():
        pool[label].append(term)
    return term_bin, pool


def sample_matched_query(
    terms: list[str],
    term_bin: dict[str, int],
    pool: dict[int, list[str]],
    neighborhood: dict[str, set[str]],
    rng: np.random.Generator,
) -> list[str]:
    forbidden: set[str] = set()
    for term in terms:
        forbidden |= neighborhood.get(term, {term})
    chosen: set[str] = set()
    for term in terms:
        label = term_bin.get(term)
        candidates = [g for g in pool.get(label, []) if g not in forbidden and g not in chosen]
        if not candidates:
            candidates = [g for g in pool.get(label, []) if g not in chosen]
        if not candidates:
            candidates = [g for g in pool.get(0, []) if g not in chosen]
        if not candidates:
            raise ValueError("IC 池里没有可用的替代术语")
        pick = str(rng.choice(np.array(candidates)))
        chosen.add(pick)
    return sorted(chosen)


@dataclass
class PhenotypeNullRecord:
    disease_id: str
    module_id: str
    module_size: int
    observed_mean_percentile: float
    null_mean: float
    null_sd: float
    z_spec: float
    p_spec: float
    draws: int


def module_mean_percentile(
    percentiles: np.ndarray,
    gene_ids: tuple[str, ...],
    genes: list[str],
) -> float:
    position = {gene: i for i, gene in enumerate(gene_ids)}
    values = [percentiles[position[g]] for g in genes if g in position]
    return float(np.mean(values)) if values else float("nan")


def rwr_percentiles(scores: np.ndarray, gene_rows: np.ndarray) -> np.ndarray:
    """把每个查询的基因分数转成疾病内百分位（列内，1 = 分数最高）。

    注意方向：rank 1 必须对应最高分，否则整套特异性会被算反。
    """
    gene_scores = scores[gene_rows, :]
    order = np.argsort(-gene_scores, axis=0, kind="stable")
    ranks = np.empty_like(order)
    columns = np.arange(gene_scores.shape[1])[None, :]
    ranks[order, columns] = np.arange(1, gene_scores.shape[0] + 1)[:, None]
    n = gene_scores.shape[0]
    if n == 1:
        return np.ones_like(gene_scores)
    return 1.0 - (ranks - 1) / (n - 1)
