"""五折汇总 core Jaccard：读取各折结果，合并后做配对检验。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    frames = []
    per_fold = []
    for fold in range(5):
        path = DIR / f"outputs/core_jaccard/fold_{fold}/core_jaccard.parquet"
        if not path.exists():
            print(f"[warn] fold {fold} missing")
            continue
        table = pd.read_parquet(path)
        table["fold"] = fold
        frames.append(table)
        pivot = table.pivot_table(index="disease_id", columns="variant", values="core_jaccard")
        per_fold.append({"fold": fold, "diseases": int(len(pivot)),
                         "baseline": round(float(pivot["baseline"].mean()), 4),
                         "refined": round(float(pivot["refined"].mean()), 4),
                         "r1": round(float(pivot["r1"].mean()), 4)})

    data = pd.concat(frames, ignore_index=True)
    pivot = data.pivot_table(index="disease_id", columns="variant", values="core_jaccard")
    summary = {"diseases": int(len(pivot)), "per_fold": per_fold}
    for a, b in (("baseline", "refined"), ("baseline", "r1")):
        pair = pivot[[a, b]].dropna()
        summary[f"{a}_vs_{b}"] = {"a": round(float(pair[a].mean()), 4),
                                  "b": round(float(pair[b].mean()), 4),
                                  "wins": int((pair[b] > pair[a]).sum()),
                                  "p": float(wilcoxon(pair[b], pair[a]).pvalue)}
    out = DIR / "outputs/core_jaccard/five_fold.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
