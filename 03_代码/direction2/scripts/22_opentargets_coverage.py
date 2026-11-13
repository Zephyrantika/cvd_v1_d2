"""用 Open Targets GraphQL API 检查覆盖率：我们映射到的疾病，各自有多少个有证据的靶点。

不下载 parquet（EBI FTP 从这台机器上太慢），改用官方 API 逐病查询。
输出 outputs/external/opentargets_coverage.parquet 与 summary.json。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\ComputerScience_file\VERTICVS_RES_PVBLICVS_STRVCTVS\ARBEIT_G3S1\SENIOR_MODEL_STRUCTURE\phenotype-modules-v0")
API = "https://api.platform.opentargets.org/api/v4/graphql"
QUERY = """
query($id:String!){
  disease(efoId:$id){
    id
    name
    associatedTargets(page:{index:0,size:1}){ count }
  }
}
"""


def query(ot_id: str, timeout: int = 30) -> dict:
    payload = json.dumps({"query": QUERY, "variables": {"id": ot_id}}).encode("utf-8")
    request = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    mapping = pd.read_parquet(BASE / "data/external/opentargets_26_09/mapping.parquet")
    mapping = mapping.drop_duplicates("disease_id")
    out_dir = BASE / "outputs/external"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in enumerate(mapping.itertuples(index=False)):
        for attempt in range(3):
            try:
                data = query(str(row.ot_id))
                disease = (data.get("data") or {}).get("disease")
                count = disease["associatedTargets"]["count"] if disease else 0
                name = disease["name"] if disease else None
                rows.append({"disease_id": row.disease_id, "ot_id": row.ot_id,
                             "ot_name": name, "n_targets": int(count)})
                break
            except Exception as error:  # 网络中断、限流、返回结构异常都重试
                if attempt == 2:
                    rows.append({"disease_id": row.disease_id, "ot_id": row.ot_id,
                                 "ot_name": None, "n_targets": -1, "error": f"{type(error).__name__}: {error}"})
                else:
                    time.sleep(3 + 2 * attempt)
        if i % 50 == 0:
            print(f"{i}/{len(mapping)}", flush=True)
        time.sleep(0.15)

    table = pd.DataFrame(rows)
    table.to_parquet(out_dir / "opentargets_coverage.parquet", index=False)

    valid = table.loc[table["n_targets"] >= 0]
    summary = {
        "mapped_diseases": int(len(table)),
        "queried_ok": int(len(valid)),
        "failed": int((table["n_targets"] < 0).sum()),
        "n_targets_median": int(valid["n_targets"].median()),
        "n_targets_mean": round(float(valid["n_targets"].mean()), 1),
        "frac_ge_5": round(float((valid["n_targets"] >= 5).mean()), 3),
        "frac_ge_20": round(float((valid["n_targets"] >= 20).mean()), 3),
        "frac_zero": round(float((valid["n_targets"] == 0).mean()), 3),
        "total_diseases_in_benchmark": 452,
    }
    (out_dir / "opentargets_coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
