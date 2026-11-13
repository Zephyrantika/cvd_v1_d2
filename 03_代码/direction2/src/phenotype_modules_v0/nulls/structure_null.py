"""结构零模型：在候选集内按度数分层抽同样大小的基因集，估计密度与 hub 的零分布。

用邻接索引算内部边，不再每抽一次就过滤一遍全量 PPI（原写法 120 模块 ×200 次要 5 分钟）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

Adjacency = dict[str, dict[str, float]]


def build_adjacency(ppi_edges: pd.DataFrame) -> Adjacency:
    adjacency: Adjacency = defaultdict(dict)
    for gene_a, gene_b, weight in zip(
        ppi_edges["gene_a"].astype(str),
        ppi_edges["gene_b"].astype(str),
        ppi_edges["weight"].astype(float),
        strict=True,
    ):
        adjacency[gene_a][gene_b] = weight
        adjacency[gene_b][gene_a] = weight
    return adjacency


def gene_degrees(adjacency: Adjacency) -> pd.Series:
    return pd.Series({gene: float(sum(neighbours.values())) for gene, neighbours in adjacency.items()})


def degree_strata(pool: list[str], degrees: pd.Series, bins: int = 10) -> dict[str, str]:
    values = np.array([float(degrees.get(gene, 0.0)) for gene in pool])
    if len(np.unique(values)) == 1:
        return {gene: "s0" for gene in pool}
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    labels = np.digitize(values, edges[1:-1], right=False)
    return {gene: f"s{int(label)}" for gene, label in zip(pool, labels, strict=True)}


def stratified_sample(pool: list[str], strata: dict[str, str], size: int, rng: np.random.Generator) -> list[str]:
    groups: dict[str, list[str]] = {}
    for gene in pool:
        groups.setdefault(strata[gene], []).append(gene)
    total = sum(len(v) for v in groups.values())
    if total < size:
        raise ValueError("候选池小于所需模块大小")
    chosen: list[str] = []
    for key, genes in groups.items():
        quota = max(1, round(size * len(genes) / total))
        take = min(quota, len(genes), size - len(chosen))
        if take > 0:
            chosen.extend(rng.choice(np.array(genes), size=take, replace=False).tolist())
    if len(chosen) < size:
        picked = set(chosen)
        rest = [g for g in pool if g not in picked]
        chosen.extend(rng.choice(np.array(rest), size=size - len(chosen), replace=False).tolist())
    return chosen[:size]


def density_and_hub(members: list[str], adjacency: Adjacency) -> tuple[float, float]:
    member_set = set(members)
    internal = 0
    weighted_degree = {gene: 0.0 for gene in member_set}
    for gene in member_set:
        for neighbour, weight in adjacency.get(gene, {}).items():
            if neighbour in member_set:
                internal += 1
                weighted_degree[gene] += weight
    edges = internal / 2
    possible = len(member_set) * (len(member_set) - 1) / 2
    density = edges / possible if possible else 0.0
    total = sum(weighted_degree.values())
    hub = 2.0 * max(weighted_degree.values(), default=0.0) / total if total else 1.0
    return float(density), float(hub)


@dataclass(frozen=True)
class StructureNullResult:
    module_id: str
    disease_id: str
    module_size: int
    observed_density: float
    null_density_mean: float
    null_density_sd: float
    p_density: float
    observed_hub: float
    null_hub_mean: float
    null_hub_sd: float
    p_hub: float
    draws: int


def structure_null(
    modules: pd.DataFrame,
    members_by_module: dict[str, list[str]],
    candidate_genes: dict[str, list[str]],
    adjacency: Adjacency,
    *,
    draws: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    degrees = gene_degrees(adjacency)
    rng = np.random.default_rng(seed)
    results: list[StructureNullResult] = []

    for row in modules.itertuples(index=False):
        disease = str(row.disease_id)
        module_id = str(row.module_id)
        members = members_by_module[module_id]
        pool = candidate_genes[disease]
        strata = degree_strata(pool, degrees)

        observed_density, observed_hub = density_and_hub(members, adjacency)
        null_density = np.empty(draws)
        null_hub = np.empty(draws)
        for draw in range(draws):
            sample = stratified_sample(pool, strata, len(members), rng)
            null_density[draw], null_hub[draw] = density_and_hub(sample, adjacency)

        results.append(StructureNullResult(
            module_id=module_id,
            disease_id=disease,
            module_size=len(members),
            observed_density=observed_density,
            null_density_mean=float(null_density.mean()),
            null_density_sd=float(null_density.std(ddof=1)) if draws > 1 else 0.0,
            p_density=(1 + int((null_density >= observed_density).sum())) / (draws + 1),
            observed_hub=observed_hub,
            null_hub_mean=float(null_hub.mean()),
            null_hub_sd=float(null_hub.std(ddof=1)) if draws > 1 else 0.0,
            p_hub=(1 + int((null_hub >= observed_hub).sum())) / (draws + 1),
            draws=draws,
        ))

    return pd.DataFrame([r.__dict__ for r in results])
