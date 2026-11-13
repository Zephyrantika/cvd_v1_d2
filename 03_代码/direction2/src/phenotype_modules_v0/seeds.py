"""可复现的种子派生，与 V0 的做法一致。

seed = sha256(f"{global_seed}|{fold}|{disease}|{purpose}|{run}") 取前 8 字节转整数。
同一组输入永远得到同一个种子，且不依赖 Python 的 hash 随机化。
"""

from __future__ import annotations

import hashlib

DEFAULT_SEED = 42


def derive_seed(
    global_seed: int = DEFAULT_SEED,
    fold: int | str = "-",
    disease: str = "-",
    purpose: str = "generic",
    run: int = 0,
) -> int:
    key = f"{global_seed}|{fold}|{disease}|{purpose}|{run}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")
