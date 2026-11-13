"""M2 输入契约：生成 input_manifest.json 并做完整性验收。

只读 V0，不写 V0 的任何文件。产出：
    outputs/input_manifest.json   逐文件 sha256、大小、行数、列名
    outputs/qc/input_check.json   检查项、结论、环境信息、与上次清单的差异

任一必需文件缺失或任一检查不通过，退出码为 1，并打印缺失清单而不是 traceback。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.io.manifest import build_entry, compare_with_previous  # noqa: E402
from phenotype_modules_v0.io.v0_inputs import (  # noqa: E402
    ALL_SPECS,
    FOLDS,
    CheckResult,
    InputSpec,
)
from phenotype_modules_v0.logging_utils import JsonlLogger  # noqa: E402

EXPECTED = {
    "n_diseases": 452,
    "n_candidate_genes": 16992,
    "n_association_rows": 688,
    "n_association_pairs": 568,
    "n_phenotypes": 12381,
    "n_modules": 2839,
    "n_module_members": 80864,
    "n_graph_nodes": 39065,
    "fold_sizes": {0: 91, 1: 91, 2: 90, 3: 90, 4: 90},
}


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_files(paths, specs: list[InputSpec]) -> tuple[list[dict], list[CheckResult]]:
    files: list[dict] = []
    missing: list[str] = []
    too_small: list[dict] = []
    for spec in specs:
        path = paths.v0(spec.rel)
        if not path.exists():
            if spec.required:
                missing.append(spec.rel)
            continue
        entry = build_entry(spec.rel, path, spec.kind, spec.columns)
        files.append(entry.as_dict())
        if spec.min_rows is not None and (entry.rows or 0) < spec.min_rows:
            too_small.append({"path": spec.rel, "rows": entry.rows, "min_rows": spec.min_rows})
    checks = [
        CheckResult("required_files_present", not missing, {"missing": missing}),
        CheckResult("min_rows_satisfied", not too_small, {"violations": too_small}),
    ]
    return files, checks


def check_datasets(paths) -> list[CheckResult]:
    checks: list[CheckResult] = []

    folds = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                            columns=["disease_id", "fold"])
    sizes = folds.groupby("fold")["disease_id"].size().to_dict()
    checks.append(CheckResult(
        "folds_452_unique",
        len(folds) == EXPECTED["n_diseases"] and folds["disease_id"].is_unique,
        {"rows": int(len(folds)), "unique_diseases": int(folds["disease_id"].nunique())},
    ))
    checks.append(CheckResult(
        "fold_sizes",
        sizes == EXPECTED["fold_sizes"],
        {"observed": {str(k): int(v) for k, v in sizes.items()},
         "expected": {str(k): v for k, v in EXPECTED["fold_sizes"].items()}},
    ))

    universe = pd.read_parquet(paths.v0("data/folds/candidate_gene_universe.parquet"), columns=["gene_id"])
    universe_ids = set(universe["gene_id"].astype(str))
    checks.append(CheckResult(
        "candidate_universe_16992",
        len(universe) == EXPECTED["n_candidate_genes"] and len(universe_ids) == EXPECTED["n_candidate_genes"],
        {"rows": int(len(universe)), "unique": len(universe_ids)},
    ))

    genes = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_genes.parquet"),
        columns=["disease_id", "gene_id"],
    )
    pairs = genes.drop_duplicates()
    checks.append(CheckResult(
        "associations_688_rows_568_pairs",
        len(genes) == EXPECTED["n_association_rows"] and len(pairs) == EXPECTED["n_association_pairs"],
        {"rows": int(len(genes)), "unique_pairs": int(len(pairs))},
    ))

    pheno = pd.read_parquet(
        paths.v0("data/processed/cardiovascular_disease_phenotypes.parquet"),
        columns=["disease_id", "hpo_id"],
    )
    checks.append(CheckResult(
        "phenotypes_12381",
        len(pheno) == EXPECTED["n_phenotypes"],
        {"rows": int(len(pheno)), "diseases": int(pheno["disease_id"].nunique())},
    ))

    modules = pd.read_parquet(paths.v0("outputs/modules/modules.parquet"),
                              columns=["module_id", "disease_id", "fold"])
    checks.append(CheckResult(
        "modules_2839",
        len(modules) == EXPECTED["n_modules"],
        {"rows": int(len(modules)), "diseases": int(modules["disease_id"].nunique())},
    ))

    members = pd.read_parquet(paths.v0("outputs/modules/module_members.parquet"),
                              columns=["module_id", "gene_id"])
    checks.append(CheckResult(
        "module_members_80864",
        len(members) == EXPECTED["n_module_members"],
        {"rows": int(len(members))},
    ))

    for fold in FOLDS:
        test_ids = {
            line.strip()
            for line in paths.v0(f"data/folds/fold_{fold}/test_diseases.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        node_map = pd.read_parquet(paths.v0(f"data/graphs/fold_{fold}/node_map.parquet"),
                                   columns=["node_type", "node_id"])
        graph_genes = set(node_map.loc[node_map["node_type"] == "gene", "node_id"].astype(str))
        checks.append(CheckResult(
            f"graph_fold{fold}_genes_match_universe",
            graph_genes == universe_ids,
            {"n_nodes": int(len(node_map)), "n_genes": len(graph_genes)},
        ))

        rankings = pd.read_parquet(
            paths.v0(f"outputs/rankings/fold_{fold}/rwr/rankings.parquet"),
            columns=["disease_id", "gene_id"],
        )
        rank_diseases = set(rankings["disease_id"].astype(str))
        rank_genes = set(rankings["gene_id"].astype(str))
        expected_rows = len(test_ids) * EXPECTED["n_candidate_genes"]
        checks.append(CheckResult(
            f"rankings_fold{fold}_shape",
            rank_diseases == test_ids and rank_genes == universe_ids and len(rankings) == expected_rows,
            {
                "rows": int(len(rankings)),
                "expected_rows": expected_rows,
                "n_test_diseases": len(test_ids),
                "diseases_match_test": rank_diseases == test_ids,
                "genes_match_universe": rank_genes == universe_ids,
            },
        ))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--out", default=None, help="默认 <outputs>/input_manifest.json")
    parser.add_argument("--allow-modified", action="store_true",
                        help="上一次清单存在差异时仍然返回 0")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    out_dir = Path(args.out).parent if args.out else paths.outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) if args.out else out_dir / "input_manifest.json"
    log = JsonlLogger(out_dir / "qc" / "check_inputs.log", "00_check_inputs")

    log.info("resolved_paths", v0_root=str(paths.v0_root), outputs=str(paths.outputs_root))

    specs = [s for spec in ALL_SPECS for s in spec.expanded()]
    files, file_checks = check_files(paths, specs)
    checks = file_checks + check_datasets(paths)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "v0_root": str(paths.v0_root),
        "v0_git_commit": git_commit(paths.v0_root),
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "counts": {
            "specs": len(specs),
            "present": len(files),
            "missing_required": len([c for c in file_checks if not c.ok and c.name == "required_files_present"]),
        },
        "files": files,
    }

    changes = compare_with_previous(manifest, manifest_path)

    failures = [c for c in checks if not c.ok]
    result = {
        "manifest": str(manifest_path),
        "n_checks": len(checks),
        "n_failed": len(failures),
        "checks": [c.as_dict() for c in checks],
        "changes_vs_previous_manifest": changes,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "qc" / "input_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    log.info("manifest_written", path=str(manifest_path), files=len(files))
    for change in changes:
        log.info("manifest_change", **change)
    for check in checks:
        (log.info if check.ok else log.error)("check", name=check.name, ok=check.ok,
                                              detail=json.dumps(check.detail, ensure_ascii=False))

    print("")
    print(f"checks: {len(checks) - len(failures)}/{len(checks)} passed")
    if failures:
        print("failed checks:")
        for check in failures:
            print(f"  - {check.name}: {json.dumps(check.detail, ensure_ascii=False)}")
        missing = next((c.detail.get("missing") for c in file_checks
                        if c.name == "required_files_present" and not c.ok), None)
        if missing:
            print("missing required files:")
            for rel in missing:
                print(f"  - {rel}")
    if changes:
        print(f"manifest changes vs previous: {len(changes)}")
    log.close()

    if failures:
        return 1
    if changes and not args.allow_modified:
        print("previous manifest differs; rerun with --allow-modified after confirming the change")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
