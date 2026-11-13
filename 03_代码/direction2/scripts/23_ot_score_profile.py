"""检查 Open Targets 关联的分数分布：多少疾病在更高证据阈值下仍有足够靶点。

总数会把文献挖掘和表达证据也算进来，所以抽样取每个疾病的关联行，看 score 分布，
估计 score ≥ 0.2 / 0.5 时还剩多少靶点。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\ComputerScience_file\VERTICVS_RES_PVBLICVS_STRVCTVS\ARBEIT_G3S1\SENIOR_MODEL_STRUCTURE\phenotype-modules-v0")
API = "https://api.platform.opentargets.org/api/v4/graphql"
QUERY = """
query($id:String!, $size:Int!){
  disease(efoId:$id){
    id
    associatedTargets(page:{index:0,size:$size}){
      count
      rows { score target { id } }
    }
  }
}
"""


def post(ot_id: str, size: int = 500) -> dict:
    payload = json.dumps({"query": QUERY, "variables": {"id": ot_id, "size": size}}).encode("utf-8")
    request = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    coverage = pd.read_parquet(BASE / "outputs/external/opentargets_coverage.parquet")
    valid = coverage.loc[coverage["n_targets"] > 0].sort_values("n_targets", ascending=False)
    sample = valid.iloc[np.linspace(0, len(valid) - 1, 20).astype(int)] if len(valid) >= 20 else valid

    rows = []
    for row in sample.itertuples(index=False):
        for attempt in range(3):
            try:
                data = post(row.ot_id)
                disease = (data.get("data") or {}).get("disease")
                if not disease:
                    break
                scores = np.array([r["score"] for r in disease["associatedTargets"]["rows"]], dtype=float)
                rows.append({
                    "disease_id": row.disease_id,
                    "ot_id": row.ot_id,
                    "count_total": disease["associatedTargets"]["count"],
                    "fetched": len(scores),
                    "n_ge_0.1": int((scores >= 0.1).sum()),
                    "n_ge_0.2": int((scores >= 0.2).sum()),
                    "n_ge_0.5": int((scores >= 0.5).sum()),
                    "score_median": round(float(np.median(scores)), 3) if len(scores) else None,
                })
                break
            except Exception:
                time.sleep(3)
        time.sleep(0.2)

    table = pd.DataFrame(rows)
    out = BASE / "outputs/external/opentargets_score_profile.parquet"
    table.to_parquet(out, index=False)
    summary = {
        "sampled_diseases": int(len(table)),
        "median_total": int(table["count_total"].median()) if len(table) else None,
        "median_ge_0.1": int(table["n_ge_0.1"].median()) if len(table) else None,
        "median_ge_0.2": int(table["n_ge_0.2"].median()) if len(table) else None,
        "frac_diseases_with_ge5_at_0.1": round(float((table["n_ge_0.1"] >= 5).mean()), 3) if len(table) else None,
        "median_ge_0.5": int(table["n_ge_0.5"].median()) if len(table) else None,
        "frac_diseases_with_ge5_at_0.5": round(float((table["n_ge_0.5"] >= 5).mean()), 3) if len(table) else None,
    }
    (BASE / "outputs/external/opentargets_score_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
