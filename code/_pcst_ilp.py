"""PCST（prize-collecting Steiner tree）的 ILP 形式，用 HiGHS 精确求解。

与最大权重连通子图的区别：高分基因是「终端」，低分但必需的连接基因是「Steiner 节点」，
只付边成本、不计入模块规模。这样 25 个高分基因即使不直接相连也能连通成树，
不会像 MWCS 那样因为不连通而无解。

目标：max Σ p_v x_v − Σ c_e y_e
约束：x ≤ z（终端必须在树里）、y ≤ z（边两端在树里）、根出发的流量保证连通、
     Σx ≤ K（只约束终端数量）。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def solve_pcst(nodes: list[str], edges: list[tuple[int, int, float]], prizes: np.ndarray,
               max_terminals: int, root: int, time_limit: int = 30) -> list[int]:
    n = len(nodes)
    m = len(edges)
    n_var = 2 * n + 2 * m  # x, z, y, f
    cost = np.zeros(n_var)
    cost[:n] = -prizes
    for index, (_, _, weight) in enumerate(edges):
        cost[2 * n + index] = 0.05 * (1.0 - min(max(weight, 0.0), 1.0))  # 边成本取正，压到与 z 同量级

    def x(i: int) -> int:
        return i

    def z(i: int) -> int:
        return n + i

    def y(i: int) -> int:
        return 2 * n + i

    def f(i: int) -> int:
        return 2 * n + m + i

    rows, cols, values, lower, upper = [], [], [], [], []
    row = 0

    for i in range(n):
        rows.append(row)
        cols.append(x(i))
        values.append(1.0)
        rows.append(row)
        cols.append(z(i))
        values.append(-1.0)
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1

    for index, (u, v, _) in enumerate(edges):
        for endpoint in (u, v):
            rows.append(row)
            cols.append(y(index))
            values.append(1.0)
            rows.append(row)
            cols.append(z(endpoint))
            values.append(-1.0)
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1

    incidence: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for index, (u, v, _) in enumerate(edges):
        incidence[u].append((index, -1.0))   # 流出记负
        incidence[v].append((index, 1.0))    # 流入记正
    for node in range(n):
        if node == root:
            continue
        for index, sign in incidence[node]:
            rows.append(row)
            cols.append(f(index))
            values.append(sign)
        rows.append(row)
        cols.append(x(node))
        values.append(-1.0)
        lower.append(0.0)
        upper.append(0.0)
        row += 1

    for index, sign in incidence[root]:
        rows.append(row)
        cols.append(f(index))
        values.append(-sign)
    for node in range(n):
        rows.append(row)
        cols.append(x(node))
        values.append(-1.0)
    lower.append(0.0)
    upper.append(0.0)
    row += 1

    for index in range(m):
        rows.append(row)
        cols.append(f(index))
        values.append(1.0)
        rows.append(row)
        cols.append(y(index))
        values.append(-float(max_terminals))
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1
        rows.append(row)
        cols.append(f(index))
        values.append(-1.0)
        rows.append(row)
        cols.append(y(index))
        values.append(-float(max_terminals))
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1

    for node in range(n):
        rows.append(row)
        cols.append(x(node))
        values.append(1.0)
    lower.append(1.0)
    upper.append(float(max_terminals))
    row += 1

    matrix = csr_matrix((values, (rows, cols)), shape=(row, n_var))
    constraint = LinearConstraint(matrix, np.array(lower), np.array(upper))
    integrality = np.zeros(n_var)
    integrality[: 2 * n + m] = 1
    lower_bounds = np.concatenate([np.zeros(2 * n + m), np.full(m, -float(max_terminals))])
    upper_bounds = np.concatenate([np.ones(2 * n + m), np.full(m, float(max_terminals))])
    bounds = Bounds(lower_bounds, upper_bounds)
    result = milp(cost, constraints=[constraint], integrality=integrality, bounds=bounds,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    if result.x is None:
        return []
    return [i for i in range(n) if result.x[x(i)] > 0.5]
