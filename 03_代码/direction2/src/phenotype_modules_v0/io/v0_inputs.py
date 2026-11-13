"""V0 输入清单定义：每个文件的相对路径、类型、必需列与最低行数。

路径里的 {fold} 会被展开成 0-4。行数下限取自 2026-10-28 的审计结果，
只用于发现"文件被替换或截断"，不是精确断言（精确断言在 00_check_inputs.py 里）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

FOLDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class InputSpec:
    rel: str
    kind: str  # parquet | json | npz | text
    required: bool = True
    min_rows: int | None = None
    columns: tuple[str, ...] = ()
    unique_on: tuple[str, ...] = ()
    note: str = ""

    def expanded(self) -> list["InputSpec"]:
        if "{fold}" not in self.rel:
            return [self]
        return [
            InputSpec(
                rel=self.rel.format(fold=fold),
                kind=self.kind,
                required=self.required,
                min_rows=self.min_rows,
                columns=self.columns,
                unique_on=self.unique_on,
                note=self.note,
            )
            for fold in FOLDS
        ]


PROCESSED: tuple[InputSpec, ...] = (
    InputSpec(
        "data/processed/cardiovascular_disease_genes.parquet",
        "parquet",
        min_rows=600,
        columns=("disease_id", "gene_id", "source", "association_type"),
        note="688 行，去重后 568 对；分层与 R7 的唯一来源",
    ),
    InputSpec(
        "data/processed/cardiovascular_disease_phenotypes.parquet",
        "parquet",
        min_rows=12000,
        columns=("disease_id", "hpo_id", "frequency_weight"),
        note="12,381 行，表型种子",
    ),
    InputSpec("data/processed/string_gene_edges_700.parquet", "parquet", min_rows=100000,
              columns=("gene_a", "gene_b", "weight"), note="主 PPI，模块结构图"),
    InputSpec("data/processed/string_gene_edges_900.parquet", "parquet", min_rows=50000,
              columns=("gene_a", "gene_b", "weight"), note="阈值敏感性"),
    InputSpec("data/processed/reactome_gene_pathway.parquet", "parquet",
              columns=("gene_id", "pathway_id"), note="图内一致性，不作独立验证"),
    InputSpec("data/processed/go_gene_annotations.parquet", "parquet",
              columns=("gene_id", "go_id"), note="外部功能一致性，不参与构图"),
    InputSpec("data/processed/go_terms.parquet", "parquet", required=False),
    InputSpec("data/processed/gwas_gene_support.parquet", "parquet", required=False,
              note="外部支持，E2 级"),
    InputSpec("data/processed/gtex_cardiovascular_expression.parquet", "parquet", required=False,
              note="组织表达，E2 级"),
)

FOLDS_FILES: tuple[InputSpec, ...] = (
    InputSpec("data/folds/candidate_gene_universe.parquet", "parquet", min_rows=16992,
              columns=("gene_id",), unique_on=("gene_id",), note="16,992 个候选基因"),
    InputSpec("data/folds/disease_fold_assignments.parquet", "parquet", min_rows=452,
              columns=("disease_id", "fold", "disease_family_id"), unique_on=("disease_id",)),
    InputSpec("data/folds/fold_{fold}/train_diseases.txt", "text"),
    InputSpec("data/folds/fold_{fold}/val_diseases.txt", "text"),
    InputSpec("data/folds/fold_{fold}/test_diseases.txt", "text"),
    InputSpec("data/folds/fold_{fold}/train_hpo_ic.parquet", "parquet",
              columns=("hpo_id", "ic")),
    InputSpec("data/folds/fold_{fold}/leakage_report.json", "json", required=False),
)

GRAPHS: tuple[InputSpec, ...] = (
    InputSpec("data/graphs/fold_{fold}/node_map.parquet", "parquet", min_rows=39065,
              columns=("node_index", "node_type", "node_id"), note="39,065 节点"),
    InputSpec("data/graphs/fold_{fold}/transition_matrix.npz", "npz",
              note="重跑 RWR 与零模型都要用"),
)

RANKINGS: tuple[InputSpec, ...] = (
    InputSpec("outputs/rankings/fold_{fold}/rwr/rankings.parquet", "parquet",
              columns=("disease_id", "gene_id", "raw_rwr_score", "rank"),
              note="主模型分数，方向二的输入"),
    InputSpec("outputs/rankings/fold_{fold}/calibrated/rankings.parquet", "parquet",
              required=False,
              columns=("disease_id", "gene_id", "raw_rwr_score", "z_score"),
              note="校正分数，只作敏感性分析"),
)

MODULES: tuple[InputSpec, ...] = (
    InputSpec("outputs/modules/modules.parquet", "parquet", min_rows=2800,
              columns=("module_id", "disease_id", "fold", "module_size"), unique_on=("module_id",)),
    InputSpec("outputs/modules/module_members.parquet", "parquet", min_rows=80000,
              columns=("module_id", "gene_id", "consensus_frequency")),
)

METRICS: tuple[InputSpec, ...] = (
    InputSpec("outputs/metrics/metric_summary.parquet", "parquet", required=False),
)

ALL_SPECS: tuple[InputSpec, ...] = tuple(
    spec
    for group in (PROCESSED, FOLDS_FILES, GRAPHS, RANKINGS, MODULES, METRICS)
    for spec in group
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}
