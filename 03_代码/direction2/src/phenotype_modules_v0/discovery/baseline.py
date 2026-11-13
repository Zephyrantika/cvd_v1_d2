"""复现 V0 的「RWR top-K + Leiden」基线。

直接 import V0 的模块函数，不复制算法，保证与 outputs/modules/ 里的模块逐成员一致。
函数把 detect / topology 作为参数传入，避免模块级全局状态。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

DetectFn = Callable[..., list[tuple[str, ...]]]
TopologyFn = Callable[..., dict]


def load_v0_models(v0_root: Path) -> tuple[DetectFn, TopologyFn]:
    src = str(Path(v0_root) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from phenotype_network_v0.models.modules import detect_leiden_modules, module_topology

    return detect_leiden_modules, module_topology


def stable_seed(seed: int, *values: object) -> int:
    """与 V0 scripts/11_detect_modules.py 的 _stable_seed 完全一致的派生方式。"""
    payload = "|".join(map(str, (seed, *values))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def detect_base_modules(
    disease_ids: Sequence[str],
    genes_by_disease: dict[str, tuple[str, ...]],
    ppi_edges: pd.DataFrame,
    *,
    detect: DetectFn,
    seed_base: int,
    fold: int,
    minimum_size: int = 5,
    maximum_size: int = 100,
) -> dict[str, list[tuple[str, ...]]]:
    modules: dict[str, list[tuple[str, ...]]] = {}
    for disease_id in disease_ids:
        modules[str(disease_id)] = detect(
            genes_by_disease[str(disease_id)],
            ppi_edges,
            seed=stable_seed(seed_base, fold, disease_id, "base"),
            minimum_size=minimum_size,
            maximum_size=maximum_size,
        )
    return modules


def drop_single_hub_modules(
    modules: dict[str, list[tuple[str, ...]]],
    ppi_edges: pd.DataFrame,
    *,
    topology: TopologyFn,
) -> tuple[dict[str, list[tuple[str, ...]]], list[dict]]:
    """与 V0 一致：hub dominance > 0.5 的模块判为单 hub 主导，移出合格表。"""
    kept: dict[str, list[tuple[str, ...]]] = {}
    rejected: list[dict] = []
    for disease_id, group in modules.items():
        kept[disease_id] = []
        for members in group:
            info = topology(members, ppi_edges)
            if info["single_hub_dominated"]:
                rejected.append({"disease_id": disease_id, "members": list(members), **info})
            else:
                kept[disease_id].append(members)
    return kept, rejected


def module_sets(modules: dict[str, list[tuple[str, ...]]]) -> dict[str, set[frozenset[str]]]:
    return {
        disease: {frozenset(map(str, members)) for members in group}
        for disease, group in modules.items()
    }
