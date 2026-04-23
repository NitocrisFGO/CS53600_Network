from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB


def validate_hose_matrix(T, n: int = 8, d: int = 4, tol: float = 1e-9) -> None:
    """
    Validate that T is an n x n hose-model traffic matrix:
      - T[i][i] = 0
      - row sums <= d
      - column sums <= d
      - all entries >= 0
    """
    if len(T) != n:
        raise ValueError(f"T must have {n} rows, got {len(T)}.")
    for row in T:
        if len(row) != n:
            raise ValueError(f"T must be {n}x{n}.")

    for i in range(n):
        for j in range(n):
            if T[i][j] < -tol:
                raise ValueError(f"T[{i}][{j}] is negative: {T[i][j]}")
        if abs(T[i][i]) > tol:
            raise ValueError(f"T[{i}][{i}] must be 0, got {T[i][i]}")

    for i in range(n):
        row_sum = sum(T[i][j] for j in range(n))
        if row_sum > d + tol:
            raise ValueError(f"Row {i} sum is {row_sum}, which exceeds {d}.")

    for j in range(n):
        col_sum = sum(T[i][j] for i in range(n))
        if col_sum > d + tol:
            raise ValueError(f"Column {j} sum is {col_sum}, which exceeds {d}.")

    positive_demands = sum(
        1 for i in range(n) for j in range(n) if i != j and T[i][j] > tol
    )
    if positive_demands == 0:
        raise ValueError(
            "The traffic matrix has no positive off-diagonal demand. "
            "In that case, lambda is vacuous/unbounded for this formulation."
        )


def solve_best_topology(
    T,
    n: int = 8,
    d: int = 4,
    time_limit: float | None = None,
    verbose: bool = True,
    tol: float = 1e-9,
):
    """
    Solve the MILP for Assignment 4 Q2.

    Variables:
      x[i,j] in {0,1}      whether directed edge i->j is selected
      f[s,t,i,j] >= 0      flow of commodity (s,t) on edge i->j
      lambda >= 0          concurrent-flow scaling factor

    Returns a dictionary with:
      - lambda
      - selected_edges
      - adjacency_matrix
      - nonzero_flows
      - model
      - status
    """
    validate_hose_matrix(T, n=n, d=d, tol=tol)

    V = list(range(n))
    A = [(i, j) for i in V for j in V if i != j]
    K = [(s, t) for s in V for t in V if s != t and T[s][t] > tol]

    model = gp.Model("assignment4_q2")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Decision variables
    x = model.addVars(A, vtype=GRB.BINARY, name="x")
    f = model.addVars(K, A, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    lam = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="lambda")

    # Objective: maximize lambda
    model.setObjective(lam, GRB.MAXIMIZE)

    # Out-degree constraints: sum_j x[i,j] = d
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in V if j != i) == d for i in V),
        name="outdeg",
    )

    # In-degree constraints: sum_i x[i,j] = d
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in V if i != j) == d for j in V),
        name="indeg",
    )

    # Flow conservation for each commodity (s,t) at each node v
    for (s, t) in K:
        demand = T[s][t]
        for v in V:
            outflow = gp.quicksum(f[s, t, v, j] for j in V if j != v)
            inflow = gp.quicksum(f[s, t, j, v] for j in V if j != v)

            if v == s:
                rhs = lam * demand
            elif v == t:
                rhs = -lam * demand
            else:
                rhs = 0.0

            model.addConstr(
                outflow - inflow == rhs,
                name=f"flow_{s}_{t}_{v}",
            )

    # Capacity-linking constraints:
    # total flow on edge (i,j) <= x[i,j]
    model.addConstrs(
        (
            gp.quicksum(f[s, t, i, j] for (s, t) in K) <= x[i, j]
            for (i, j) in A
        ),
        name="cap",
    )

    model.optimize()

    status = model.Status

    if status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("assignment4_q2_iis.ilp")
        raise RuntimeError(
            "Model is infeasible. IIS written to assignment4_q2_iis.ilp"
        )

    if status == GRB.INF_OR_UNBD:
        raise RuntimeError(
            "Model is infeasible or unbounded. "
            "Try setting DualReductions = 0 and re-optimizing."
        )

    if status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Unexpected solver status: {status}")

    if model.SolCount == 0:
        raise RuntimeError("Solver finished without an incumbent feasible solution.")

    lambda_value = lam.X

    selected_edges = sorted((i, j) for (i, j) in A if x[i, j].X > 0.5)

    adjacency_matrix = [[0 for _ in V] for _ in V]
    for (i, j) in selected_edges:
        adjacency_matrix[i][j] = 1

    nonzero_flows = {}
    flow_tol = 1e-8
    for (s, t) in K:
        for (i, j) in A:
            val = f[s, t, i, j].X
            if val > flow_tol:
                nonzero_flows[(s, t, i, j)] = val

    return {
        "lambda": lambda_value,
        "selected_edges": selected_edges,
        "adjacency_matrix": adjacency_matrix,
        "nonzero_flows": nonzero_flows,
        "model": model,
        "status": status,
        "commodities": K,
    }


def check_solution(T, result, n: int = 8, d: int = 4, tol: float = 1e-6) -> None:
    """
    Verify:
      - each node has out-degree d
      - each node has in-degree d
      - every selected edge has capacity respected
      - flow conservation holds for every commodity
    """
    lam = result["lambda"]
    adj = result["adjacency_matrix"]
    flows = result["nonzero_flows"]
    K = result["commodities"]

    # Degree checks
    for i in range(n):
        outdeg = sum(adj[i][j] for j in range(n) if j != i)
        if abs(outdeg - d) > tol:
            raise AssertionError(f"Out-degree check failed at node {i}: {outdeg} != {d}")

    for j in range(n):
        indeg = sum(adj[i][j] for i in range(n) if i != j)
        if abs(indeg - d) > tol:
            raise AssertionError(f"In-degree check failed at node {j}: {indeg} != {d}")

    # Capacity checks
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total_edge_flow = sum(flows.get((s, t, i, j), 0.0) for (s, t) in K)
            if total_edge_flow - adj[i][j] > tol:
                raise AssertionError(
                    f"Capacity violated on edge ({i},{j}): "
                    f"flow={total_edge_flow}, edge={adj[i][j]}"
                )

    # Flow conservation checks
    for (s, t) in K:
        demand = T[s][t]
        for v in range(n):
            outflow = sum(flows.get((s, t, v, j), 0.0) for j in range(n) if j != v)
            inflow = sum(flows.get((s, t, j, v), 0.0) for j in range(n) if j != v)

            if v == s:
                rhs = lam * demand
            elif v == t:
                rhs = -lam * demand
            else:
                rhs = 0.0

            if abs((outflow - inflow) - rhs) > tol:
                raise AssertionError(
                    f"Flow conservation failed for commodity ({s},{t}) at node {v}: "
                    f"LHS={outflow - inflow}, RHS={rhs}"
                )

    print("All checks passed.")
    print(f"Verified lambda = {lam:.8f}")


def print_result(result):
    print("\n===== Solve Summary =====")
    print(f"Status code: {result['status']}")
    print(f"Optimal / best-known lambda: {result['lambda']:.8f}")

    print("\nSelected directed edges:")
    for e in result["selected_edges"]:
        print(e)

    print("\nAdjacency matrix:")
    for row in result["adjacency_matrix"]:
        print(row)

    print(f"\nNumber of nonzero commodity-edge flows: {len(result['nonzero_flows'])}")


if __name__ == "__main__":
    # Example hose-model traffic matrix (8x8)
    # Each row sum <= 4, each column sum <= 4, diagonal = 0
    T = [
        [0, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 2, 2],
        [2, 0, 0, 0, 0, 0, 0, 2],
        [2, 2, 0, 0, 0, 0, 0, 0],
    ]

    result = solve_best_topology(T, n=8, d=4, time_limit=None, verbose=True)
    print_result(result)
    check_solution(T, result, n=8, d=4)