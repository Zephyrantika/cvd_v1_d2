"""精确求解最大权重连通子图（PCST 的等价形式），供 53/54 号脚本共用。

变量：x_v（节点）、y_e（边）、f_e（从根出发的有向流量）。
目标：max Σ p_v x_v − Σ c_e y_e；约束：流量守恒保证连通、y ≤ x、Σx ≤ max_size。
求解器：scipy.optimize.milp（HiGHS），CPU 即可。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def solve_mwcs(nodes: list[str], edges: list[tuple[int, int, float]], prizes: np.ndarray,
               max_size: int, root: int, time_limit: int = 20, min_size: int = 1) -> list[int]:
    n = len(nodes)
    m = len(edges)
    n_var = n + m + m
    cost = np.zeros(n_var)
    cost[:n] = -prizes
    for index, (_, _, weight) in enumerate(edges):
        cost[n + index] = -np.log(max(weight, 1e-6) / 0.9)

    rows, cols, values, lower, upper = [], [], [], [], []
    row = 0
    for index, (u, v, _) in enumerate(edges):
        rows += [row, row]
        cols += [n + index, u]
        values += [1.0, -1.0]
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1
        rows += [row, row]
        cols += [n + index, v]
        values += [1.0, -1.0]
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1

    incidence: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for index, (u, v, _) in enumerate(edges):
        incidence[u].append((index, -1.0))
        incidence[v].append((index, 1.0))
    for node in range(n):
        if node == root:
            continue
        for index, sign in incidence[node]:
            rows.append(row)
            cols.append(n + m + index)
            values.append(sign)
        rows.append(row)
        cols.append(node)
        values.append(-1.0)
        lower.append(0.0)
        upper.append(0.0)
        row += 1

    for index in range(m):
        rows += [row, row]
        cols += [n + m + index, n + index]
        values += [1.0, -float(max_size)]
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1

    for node in range(n):
        rows.append(row)
        cols.append(node)
        values.append(1.0)
    lower.append(-np.inf)
    upper.append(float(max_size))
    row += 1

    if min_size > 1:
        for node in range(n):
            rows.append(row)
            cols.append(node)
            values.append(1.0)
        lower.append(float(min_size))
        upper.append(np.inf)
        row += 1

    matrix = csr_matrix((values, (rows, cols)), shape=(row, n_var))
    constraint = LinearConstraint(matrix, np.array(lower), np.array(upper))
    integrality = np.zeros(n_var)
    integrality[: n + m] = 1
    bounds = Bounds(np.zeros(n_var), np.concatenate([np.ones(n + m), np.full(m, float(max_size))]))
    result = milp(cost, constraints=[constraint], integrality=integrality, bounds=bounds,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    if result.x is None:
        return []
    return [i for i in range(n) if result.x[i] > 0.5]
