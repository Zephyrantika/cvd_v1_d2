"""路径与配置加载。

v0_root 的解析顺序：显式参数 > 环境变量 MODULES_V0_ROOT > configs/paths.yaml > 自动探测。
自动探测覆盖两种布局：交接包（../v0）与原仓库（../../SENIOR_MODEL_STRUCTURE/phenotype-network-v0）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # phenotype-modules-v0/
AUTODETECT_CANDIDATES = (
    PACKAGE_ROOT.parent / "v0",                        # 交接包：direction2/ 与 v0/ 并列
    PACKAGE_ROOT.parent / "phenotype-network-v0",      # 原仓库：两个项目目录并列
)


class ConfigError(RuntimeError):
    """配置无法解析时抛出，信息里带上尝试过的路径。"""


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层不是映射: {path}")
    return data


def looks_like_v0(path: Path) -> bool:
    """用三个标志性文件判断目录是不是一个完整的 V0 根。"""
    return (
        (path / "data/folds/candidate_gene_universe.parquet").exists()
        and (path / "data/processed/cardiovascular_disease_genes.parquet").exists()
        and (path / "outputs/modules/modules.parquet").exists()
    )


def resolve_v0_root(explicit: str | Path | None = None, config_path: Path | None = None) -> Path:
    """解析 V0 根目录。显式来源（命令行、环境变量）给了就必须有效，不回退到自动探测。"""
    invalid: list[tuple[str, Path]] = []

    if explicit is not None:
        resolved = Path(explicit).expanduser().resolve()
        if looks_like_v0(resolved):
            return resolved
        invalid.append(("--v0-root", resolved))

    env = os.environ.get("MODULES_V0_ROOT")
    if env:
        resolved = Path(env).expanduser().resolve()
        if looks_like_v0(resolved):
            return resolved
        invalid.append(("MODULES_V0_ROOT", resolved))

    if invalid:
        name, path = invalid[0]
        raise ConfigError(
            f"{name} is not a valid V0 root: {path}\n"
            "它必须同时包含 data/folds/candidate_gene_universe.parquet、"
            "data/processed/cardiovascular_disease_genes.parquet、outputs/modules/modules.parquet。"
        )

    if config_path is None:
        config_path = PACKAGE_ROOT / "configs" / "paths.yaml"
    if config_path.exists():
        configured = load_yaml(config_path).get("v0_root")
        if configured:
            resolved = (PACKAGE_ROOT / str(configured)).resolve()
            if looks_like_v0(resolved):
                return resolved

    for path in AUTODETECT_CANDIDATES:
        if looks_like_v0(path):
            return path.resolve()

    raise ConfigError(
        "cannot resolve V0 root; tried configs/paths.yaml and: "
        + ", ".join(str(p) for p in AUTODETECT_CANDIDATES)
        + ". Pass --v0-root or set MODULES_V0_ROOT."
    )


@dataclass(frozen=True)
class Paths:
    package_root: Path
    v0_root: Path
    outputs_root: Path

    def out(self, *parts: str) -> Path:
        return self.outputs_root.joinpath(*parts)

    def v0(self, rel: str) -> Path:
        return self.v0_root / rel


def resolve_paths(v0_root: str | Path | None = None) -> Paths:
    config_path = PACKAGE_ROOT / "configs" / "paths.yaml"
    outputs_root = PACKAGE_ROOT / "outputs"
    if config_path.exists():
        cfg = load_yaml(config_path)
        outputs_root = (PACKAGE_ROOT / str(cfg.get("outputs_root", "outputs"))).resolve()
    return Paths(
        package_root=PACKAGE_ROOT,
        v0_root=resolve_v0_root(v0_root, config_path),
        outputs_root=outputs_root,
    )
