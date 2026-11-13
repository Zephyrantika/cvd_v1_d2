"""只读 V0 排名表，构造疾病×基因分数矩阵与分位矩阵。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RANKINGS_REL = "outputs/rankings/fold_{fold}/rwr/rankings.parquet"


@dataclass(frozen=True)
class ScoreMatrix:
    fold: int
    disease_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    scores: np.ndarray  # (n_diseases, n_genes) float32
    ranks: np.ndarray   # (n_diseases, n_genes) int32，1 表示最高

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.disease_ids), len(self.gene_ids)

    def disease_index(self, disease_id: str) -> int:
        return self.disease_ids.index(str(disease_id))

    def top_k(self, disease_id: str, k: int) -> tuple[str, ...]:
        row = self.ranks[self.disease_index(disease_id)]
        order = np.argsort(row, kind="stable")[:k]
        return tuple(self.gene_ids[i] for i in order)

    def percentiles(self) -> np.ndarray:
        """疾病内百分位：第一名 1.0，最后一名 0.0。用排名算，不用分数尺度。"""
        n = self.shape[1]
        if n == 1:
            return np.ones_like(self.ranks, dtype=np.float32)
        return (1.0 - (self.ranks - 1) / (n - 1)).astype(np.float32)

    def gene_percentiles(self, disease_id: str, genes: tuple[str, ...] | list[str]) -> np.ndarray:
        row = self.percentiles()[self.disease_index(disease_id)]
        position = {gene: i for i, gene in enumerate(self.gene_ids)}
        return np.array([row[position[str(g)]] for g in genes], dtype=np.float32)


def load_fold_scores(v0_root: Path, fold: int) -> ScoreMatrix:
    path = Path(v0_root) / RANKINGS_REL.format(fold=fold)
    frame = pd.read_parquet(path, columns=["disease_id", "gene_id", "raw_rwr_score", "rank"])
    frame["disease_id"] = frame["disease_id"].astype(str)
    frame["gene_id"] = frame["gene_id"].astype(str)

    diseases = tuple(sorted(frame["disease_id"].unique()))
    genes = tuple(sorted(frame["gene_id"].unique(), key=int))
    if len(diseases) * len(genes) != len(frame):
        raise ValueError(f"fold {fold} 排名表不是完整的疾病×基因网格")

    disease_pos = {d: i for i, d in enumerate(diseases)}
    gene_pos = {g: i for i, g in enumerate(genes)}
    scores = np.zeros((len(diseases), len(genes)), dtype=np.float32)
    ranks = np.zeros((len(diseases), len(genes)), dtype=np.int32)
    scores[frame["disease_id"].map(disease_pos), frame["gene_id"].map(gene_pos)] = frame["raw_rwr_score"].astype(np.float32)
    ranks[frame["disease_id"].map(disease_pos), frame["gene_id"].map(gene_pos)] = frame["rank"].astype(np.int32)

    if not np.isfinite(scores).all():
        raise ValueError(f"fold {fold} 分数含 NaN 或 inf")
    if (ranks < 1).any() or (ranks > len(genes)).any():
        raise ValueError(f"fold {fold} 排名越界")
    return ScoreMatrix(fold=fold, disease_ids=diseases, gene_ids=genes, scores=scores, ranks=ranks)


def save_scores(matrix: ScoreMatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fold=np.array([matrix.fold], dtype=np.int32),
        disease_ids=np.array(matrix.disease_ids, dtype=object),
        gene_ids=np.array(matrix.gene_ids, dtype=object),
        scores=matrix.scores,
        ranks=matrix.ranks,
    )


def load_saved_scores(path: Path) -> ScoreMatrix:
    data = np.load(path, allow_pickle=True)
    return ScoreMatrix(
        fold=int(data["fold"][0]),
        disease_ids=tuple(str(x) for x in data["disease_ids"]),
        gene_ids=tuple(str(x) for x in data["gene_ids"]),
        scores=data["scores"],
        ranks=data["ranks"],
    )
