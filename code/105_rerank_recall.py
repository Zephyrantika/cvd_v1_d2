"""召回问题的根治尝试：轻量重排器（按疾病族分组交叉验证）。

特征（全部与目标疾病标签无关）：
  1. RWR 分位
  2. log 加权度
  3. 心血管组织表达（log1p 中位 TPM 的最大值）
  4. 跨疾病标签先验（剔除目标疾病及其家族）
  5. 基因 HPO 注释与疾病表型的 IC 加权相似度
  6. 通路成员数（log1p）

训练：4 折疾病做训练、留 1 折测试；标签为 curated 疾病—基因对。
评价：测试折上 top-200 命中率、MRR、recall@500。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenotype_modules_v0.config import resolve_paths  # noqa: E402
from phenotype_modules_v0.scores.provider import load_saved_scores  # noqa: E402

CURATED = {"original", "clinvar", "clingen", "disgenet"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", default=None)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--pool", type=int, default=2000)
    parser.add_argument("--neg", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use", default="all")
    args = parser.parse_args()

    paths = resolve_paths(args.v0_root)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    out_dir = paths.out("rerank", f"test_fold_{args.test_fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_saved_scores(paths.out("scores", "fold_0", "score_matrix.npz"))
    pct = matrix.percentiles().astype(np.float32)
    genes = list(matrix.gene_ids)
    n_genes = len(genes)
    position = {g: i for i, g in enumerate(genes)}
    n_diseases = len(matrix.disease_ids)

    chain = Path(__file__).resolve().parents[1] / "data/chain"
    ppi = pd.read_parquet(paths.v0("data/processed/string_gene_edges_700.parquet"),
                          columns=["gene_a", "gene_b", "weight"])
    deg = np.zeros(n_genes, dtype=np.float32)
    for a, b, w in zip(ppi["gene_a"].astype(str), ppi["gene_b"].astype(str), ppi["weight"].astype(float)):
        if a in position:
            deg[position[a]] += w
        if b in position:
            deg[position[b]] += w
    deg = np.log1p(deg)

    tissue = pd.read_parquet(chain / "fact_gene_tissue.parquet")
    tissue["gene_id"] = tissue["gene_id"].astype(str)
    expr = np.zeros(n_genes, dtype=np.float32)
    for g, v in tissue.groupby("gene_id")["median_tpm"].max().items():
        if g in position:
            expr[position[g]] = np.log1p(float(v))

    pathways = pd.read_parquet(chain / "fact_gene_pathway.parquet")
    pathways["gene_id"] = pathways["gene_id"].astype(str)
    path_count = np.zeros(n_genes, dtype=np.float32)
    for g, n in pathways.groupby("gene_id")["pathway_id"].nunique().items():
        if g in position:
            path_count[position[g]] = np.log1p(float(n))

    g2p = pd.read_parquet(chain / "fact_gene_phenotype.parquet")
    g2p["gene_id"] = g2p["gene_id"].astype(str)
    g2p["hpo_id"] = g2p["hpo_id"].astype(str)
    d2p = pd.read_parquet(chain / "fact_disease_phenotype.parquet")
    d2p["disease_id"] = d2p["disease_id"].astype(str)
    d2p["hpo_id"] = d2p["hpo_id"].astype(str)
    terms = sorted(set(g2p["hpo_id"]) | set(d2p["hpo_id"]))
    term_pos = {t: i for i, t in enumerate(terms)}
    ic_train = pd.read_parquet(paths.v0("data/folds/fold_0/train_hpo_ic.parquet"))
    ic_map = dict(zip(ic_train["hpo_id"].astype(str), ic_train["ic"].astype(float)))
    ic_vec = np.array([ic_map.get(t, 0.0) for t in terms], dtype=np.float32)

    g_rows, g_cols, g_vals = [], [], []
    for g, t in zip(g2p["gene_id"], g2p["hpo_id"]):
        if g in position and t in term_pos:
            g_rows.append(position[g])
            g_cols.append(term_pos[t])
            g_vals.append(ic_vec[term_pos[t]] + 1e-6)
    gene_term = csr_matrix((g_vals, (g_rows, g_cols)), shape=(n_genes, len(terms)))

    d_rows, d_cols, d_vals = [], [], []
    disease_index = {d: i for i, d in enumerate(matrix.disease_ids)}
    for d, t in zip(d2p["disease_id"], d2p["hpo_id"]):
        if d in disease_index and t in term_pos:
            d_rows.append(disease_index[d])
            d_cols.append(term_pos[t])
            d_vals.append(ic_vec[term_pos[t]] + 1e-6)
    disease_term = csr_matrix((d_vals, (d_rows, d_cols)), shape=(n_diseases, len(terms)))

    sim = (gene_term @ disease_term.T).toarray().astype(np.float32)
    sim /= (np.linalg.norm(gene_term.toarray(), axis=1, keepdims=True) + 1e-6)
    sim /= (np.linalg.norm(disease_term.toarray(), axis=1, keepdims=True).T + 1e-6)
    print(f"[info] features built {time.time() - started:.1f}s", flush=True)

    ALL = ["pct", "log_degree", "expr", "prior", "pheno_sim", "pathways"]
    wanted = ALL if args.use == "all" else args.use.split(",")
    keep_idx = [ALL.index(x) for x in wanted]

    def select(cols):
        return [cols[i] for i in keep_idx]

    labels = pd.read_parquet(chain / "fact_disease_gene.parquet")
    labels["disease_id"] = labels["disease_id"].astype(str)
    labels["gene_id"] = labels["gene_id"].astype(str)
    labels = labels[labels["source"].isin(CURATED)].drop_duplicates(["disease_id", "gene_id"])
    label_of = labels.groupby("disease_id")["gene_id"].apply(set).to_dict()

    fam = pd.read_parquet(paths.v0("data/folds/disease_fold_assignments.parquet"),
                          columns=["disease_id", "disease_family_id", "fold"])
    fam["disease_id"] = fam["disease_id"].astype(str)
    family = dict(zip(fam["disease_id"], fam["disease_family_id"].astype(str)))
    fold_of = dict(zip(fam["disease_id"], fam["fold"].astype(int)))

    families = sorted(set(family.values()))
    family_label_count = {}
    for f in families:
        vector = np.zeros(n_genes, dtype=np.float32)
        for d, gs in label_of.items():
            if family.get(d) == f:
                continue
            for g in gs:
                if g in position:
                    vector[position[g]] += 1.0
        family_label_count[f] = np.log1p(vector)

    def disease_prior(d: str) -> np.ndarray:
        return family_label_count[family.get(d, "")]

    def build_rows(diseases: list[str], is_train: bool):
        X, y = [], []
        n_pos = 0
        for d in diseases:
            di = disease_index[d]
            row = pct[di, :]
            top = np.argsort(-row)[: args.pool]
            known = {g for g in label_of.get(d, set()) if g in position}
            known_idx = np.array(sorted(position[g] for g in known), dtype=int) if known else np.array([], dtype=int)
            others = np.setdiff1d(np.arange(n_genes), known_idx)
            neg = rng.choice(others, size=min(args.neg, len(others)), replace=False)
            idx = np.unique(np.concatenate([top, known_idx, neg]))
            prior = disease_prior(d)
            cols = [row[idx], deg[idx], expr[idx], prior[idx], sim[idx, di], path_count[idx]]
            feat = np.column_stack(select(cols))
            X.append(feat)
            y.append(np.isin(idx, known_idx).astype(np.int8))
            n_pos += len(known)
        print(f"[debug] diseases={len(diseases)} positives={n_pos} rows={sum(len(a) for a in X)}", flush=True)
        return np.vstack(X), np.concatenate(y)

    # 按疾病族分组，在 fold 0 的 91 个疾病内部做 5 折（族不交叉）
    diseases = list(matrix.disease_ids)
    fam_groups: dict[str, list[str]] = {}
    for d in diseases:
        fam_groups.setdefault(family.get(d, d), []).append(d)
    groups = list(fam_groups.values())
    rng.shuffle(groups)
    bins: list[list[str]] = [[] for _ in range(5)]
    for i, group in enumerate(groups):
        bins[i % 5].extend(group)

    results = []
    for k in range(5):
        test_dis = [d for d in bins[k] if d in set(labels.disease_id)]
        train_dis = [d for d in diseases if d not in set(test_dis)]
        if not test_dis:
            continue
        Xtr, ytr = build_rows(train_dis, True)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        model.fit(scaler.transform(Xtr), ytr)

        def score_of(d: str) -> np.ndarray:
            di = disease_index[d]
            prior = disease_prior(d)
            cols = [pct[di, :], deg, expr, prior, sim[:, di], path_count]
            feat = np.column_stack(select(cols))
            return model.predict_proba(scaler.transform(feat))[:, 1]

        def metrics(score_fn, name: str) -> dict:
            hits200 = hits500 = total = 0
            rr = []
            for d in test_dis:
                known = {g for g in label_of.get(d, set()) if g in position}
                if not known:
                    continue
                order = np.argsort(-score_fn(d))
                rank_of = np.empty(n_genes, dtype=np.int32)
                rank_of[order] = np.arange(n_genes, dtype=np.int32)
                ranks = rank_of[np.array([position[g] for g in known], dtype=int)]
                hits200 += int((ranks < 200).sum())
                hits500 += int((ranks < 500).sum())
                total += len(known)
                rr.extend(1.0 / (ranks + 1))
            return {"name": name, "known": total, "top200": round(hits200 / max(total, 1), 4),
                    "top500": round(hits500 / max(total, 1), 4), "mrr": round(float(np.mean(rr)), 4)}

        results.append({"split": k,
                        "baseline": metrics(lambda d: pct[disease_index[d], :], "rwr"),
                        "rerank": metrics(score_of, "rerank"),
                        "coef": dict(zip(["pct", "log_degree", "expr", "prior", "pheno_sim", "pathways"],
                                         [round(float(c), 3) for c in model.coef_[0]]))})
        print(f"[info] split {k} done", flush=True)

    agg = {}
    for name in ("baseline", "rerank"):
        rows = [r[name] for r in results]
        agg[name] = {"top200": round(float(np.mean([x["top200"] for x in rows])), 4),
                     "top500": round(float(np.mean([x["top500"] for x in rows])), 4),
                     "mrr": round(float(np.mean([x["mrr"] for x in rows])), 4)}
    payload = {"test_fold": args.test_fold, "splits": results, "aggregate": agg,
               "features": wanted,
               "seconds": round(time.time() - started, 1)}
    (out_dir / "rerank.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
