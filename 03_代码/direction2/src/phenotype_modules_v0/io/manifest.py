"""输入清单：逐文件 sha256、大小、行数，作为数据版本凭证。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_fingerprint(path: Path, columns: tuple[str, ...] = ()) -> dict[str, Any]:
    """行数、列名、以及指定列的取值规模，用于快速判断文件是否被替换。"""
    frame = pd.read_parquet(path)
    info: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    for column in columns:
        if column in frame.columns:
            info[f"nunique_{column}"] = int(frame[column].nunique())
    return info


@dataclass
class ManifestEntry:
    path: str
    kind: str
    size_bytes: int
    sha256: str
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": self.columns,
            "extra": self.extra,
        }


def build_entry(rel: str, path: Path, kind: str, columns: tuple[str, ...] = ()) -> ManifestEntry:
    entry = ManifestEntry(
        path=rel,
        kind=kind,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )
    if kind == "parquet":
        info = parquet_fingerprint(path, columns)
        entry.rows = info["rows"]
        entry.columns = info["columns"]
        entry.extra = {k: v for k, v in info.items() if k.startswith("nunique_")}
    return entry


def compare_with_previous(manifest: dict[str, Any], previous_path: Path) -> list[dict[str, Any]]:
    """和上一次的 manifest 比对，返回有变化的文件。没有旧文件时返回空。"""
    import json

    if not previous_path.exists():
        return []
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    old = {item["path"]: item for item in previous.get("files", [])}
    changes: list[dict[str, Any]] = []
    for item in manifest["files"]:
        before = old.get(item["path"])
        if before is None:
            changes.append({"path": item["path"], "change": "added"})
        elif before["sha256"] != item["sha256"]:
            changes.append(
                {
                    "path": item["path"],
                    "change": "modified",
                    "sha256_before": before["sha256"],
                    "sha256_after": item["sha256"],
                    "rows_before": before.get("rows"),
                    "rows_after": item.get("rows"),
                }
            )
    for path in old:
        if path not in {item["path"] for item in manifest["files"]}:
            changes.append({"path": path, "change": "removed"})
    return changes
