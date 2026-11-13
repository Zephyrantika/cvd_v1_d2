"""结构化日志：控制台只写 ASCII 摘要，详细记录写 JSON Lines 文件。

Windows 控制台默认 GBK，中文日志会乱码；所以文件用 UTF-8，控制台消息保持英文。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, log_path: Path, run_name: str) -> None:
        self.log_path = log_path
        self.run_name = run_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = log_path.open("a", encoding="utf-8")
        self._t0 = time.time()

    def _emit(self, level: str, event: str, **fields: Any) -> dict:
        record = {
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - self._t0, 3),
            "run": self.run_name,
            "level": level,
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return record

    def info(self, event: str, **fields: Any) -> None:
        record = self._emit("info", event, **fields)
        detail = " ".join(f"{k}={v}" for k, v in record.items() if k not in {"ts", "elapsed_s", "run", "level", "event"})
        print(f"[info] {event} {detail}".rstrip(), file=sys.stdout)

    def error(self, event: str, **fields: Any) -> None:
        record = self._emit("error", event, **fields)
        detail = " ".join(f"{k}={v}" for k, v in record.items() if k not in {"ts", "elapsed_s", "run", "level", "event"})
        print(f"[error] {event} {detail}".rstrip(), file=sys.stderr)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
