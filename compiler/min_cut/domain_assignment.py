from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

from .cost_model import CostModel
from .profiler_db import Domain

# Section C: unify the operator graph types.
#
# The min_cut package previously defined its own `OperatorNode`,
# `DataEdge` and `OperatorGraph` dataclasses that were structurally
# identical to `ir.types` modulo an `attributes` field. Having two
# parallel types made it impossible for a single compiler pipeline to
# consume graphs emitted by the fx frontend and the legacy JSON loader
# interchangeably, because `isinstance(x, ir.types.OperatorGraph)` was
# False for min_cut graphs and vice versa.
#
# Re-exporting from `ir.types` merges the two:
# - existing positional-arg constructors in this file still work
#   (`OperatorNode(node_id, op_type, input_shape, output_shape)` — the
#   extra `attributes` field has a default).
# - `isinstance` checks against either module now succeed for both.
# - every call site that imported from `.domain_assignment` keeps
#   working without modification.
from ir.types import DataEdge, OperatorGraph, OperatorNode  # noqa: F401  (re-export)


@dataclass(frozen=True)
class AssignmentResult:
    assignment: Dict[str, Domain]
    node_cost_ms: float
    conversion_cost_ms: float
    total_cost_ms: float
    per_node_costs: Dict[str, Dict[str, float]]


def _as_shape(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def load_graph_json(path: str | Path) -> OperatorGraph:
    payload = json.loads(Path(path).read_text())
    nodes = [
        OperatorNode(
            node_id=str(node["node_id"]),
            op_type=str(node["op_type"]),
            input_shape=_as_shape(node["input_shape"]),
            output_shape=_as_shape(node["output_shape"]),
        )
        for node in payload["nodes"]
    ]
    edges = [
        DataEdge(
            src=str(edge["src"]),
            dst=str(edge["dst"]),
            tensor_shape=_as_shape(edge["tensor_shape"]),
        )
        for edge in payload["edges"]
    ]
    return OperatorGraph(graph_id=str(payload.get("graph_id", "graph")), nodes=nodes, edges=edges)


def _add_capacity(capacity: Dict[str, Dict[str, float]], u: str, v: str, cap: float) -> None:
    if cap < 0:
        raise ValueError(f"Negative capacity not allowed: {u}->{v}={cap}")
    capacity[u][v] = capacity[u].get(v, 0.0) + float(cap)
    capacity.setdefault(v, {})
    capacity[v].setdefault(u, 0.0)


def _edmonds_karp_min_cut(
    capacity: Dict[str, Dict[str, float]],
    source: str,
    sink: str,
) -> Tuple[float, Dict[str, Dict[str, float]], set[str]]:
    residual = {u: dict(vs) for u, vs in capacity.items()}
    max_flow = 0.0

    while True:
        parent: Dict[str, str] = {}
        q = deque([source])
        seen = {source}
        while q and sink not in seen:
            u = q.popleft()
            for v, cap in residual.get(u, {}).items():
                if cap > 1e-12 and v not in seen:
                    seen.add(v)
                    parent[v] = u
                    q.append(v)
        if sink not in seen:
            break

        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] = residual.get(v, {}).get(u, 0.0) + path_flow
            v = u
        max_flow += path_flow

    reachable = set()
    q = deque([source])
    reachable.add(source)
    while q:
        u = q.popleft()
        for v, cap in residual.get(u, {}).items():
            if cap > 1e-12 and v not in reachable:
                reachable.add(v)
                q.append(v)
    return max_flow, residual, reachable


def evaluate_assignment_cost(
    graph: OperatorGraph,
    assignment: Dict[str, Domain],
    cost_model: CostModel,
) -> Tuple[float, float, float]:
    nodes_by_id = {n.node_id: n for n in graph.nodes}
    node_cost_ms = 0.0
    conversion_cost_ms = 0.0

    for node in graph.nodes:
        domain = assignment[node.node_id]
        node_cost_ms += cost_model.estimate_node_cost(
            op_type=node.op_type,
            domain=domain,
            input_shape=node.input_shape,
            output_shape=node.output_shape,
        ).latency_ms

    for edge in graph.edges:
        src_domain = assignment[edge.src]
        dst_domain = assignment[edge.dst]
        if src_domain != dst_domain:
            conversion_cost_ms += cost_model.estimate_conversion_cost(
                tensor_shape=edge.tensor_shape,
                from_domain=src_domain,
                to_domain=dst_domain,
            ).latency_ms

    return node_cost_ms, conversion_cost_ms, node_cost_ms + conversion_cost_ms


def assign_domains_min_cut(graph: OperatorGraph, cost_model: CostModel) -> AssignmentResult:
    """Solve binary HE/MPC assignment via s-t min-cut.

    Construction:
    - source side => HE
    - sink side => MPC
    - terminal edges:
      * source -> node capacity = cost_MPC(node)
      * node -> sink capacity = cost_HE(node)
    - pairwise edges:
      * node_i <-> node_j capacity = conversion_penalty(edge i->j)
    """

    source = "__SRC_HE__"
    sink = "__SNK_MPC__"
    capacity: Dict[str, Dict[str, float]] = defaultdict(dict)
    per_node_costs: Dict[str, Dict[str, float]] = {}

    for node in graph.nodes:
        he_cost = cost_model.estimate_node_cost(
            op_type=node.op_type,
            domain="HE",
            input_shape=node.input_shape,
            output_shape=node.output_shape,
        ).latency_ms
        mpc_cost = cost_model.estimate_node_cost(
            op_type=node.op_type,
            domain="MPC",
            input_shape=node.input_shape,
            output_shape=node.output_shape,
        ).latency_ms
        per_node_costs[node.node_id] = {"HE": he_cost, "MPC": mpc_cost}

        # If node stays on HE side, cut pays node->sink (HE cost).
        # If node stays on MPC side, cut pays source->node (MPC cost).
        _add_capacity(capacity, source, node.node_id, mpc_cost)
        _add_capacity(capacity, node.node_id, sink, he_cost)

    for edge in graph.edges:
        he_to_mpc = cost_model.estimate_conversion_cost(
            tensor_shape=edge.tensor_shape,
            from_domain="HE",
            to_domain="MPC",
        ).latency_ms
        mpc_to_he = cost_model.estimate_conversion_cost(
            tensor_shape=edge.tensor_shape,
            from_domain="MPC",
            to_domain="HE",
        ).latency_ms
        penalty = 0.5 * (he_to_mpc + mpc_to_he)
        # Undirected disagreement penalty encoded as symmetric capacities.
        _add_capacity(capacity, edge.src, edge.dst, penalty)
        _add_capacity(capacity, edge.dst, edge.src, penalty)

    _, _, reachable_from_source = _edmonds_karp_min_cut(capacity, source, sink)
    assignment: Dict[str, Domain] = {}
    for node in graph.nodes:
        assignment[node.node_id] = "HE" if node.node_id in reachable_from_source else "MPC"

    node_cost_ms, conversion_cost_ms, total_cost_ms = evaluate_assignment_cost(graph, assignment, cost_model)
    return AssignmentResult(
        assignment=assignment,
        node_cost_ms=node_cost_ms,
        conversion_cost_ms=conversion_cost_ms,
        total_cost_ms=total_cost_ms,
        per_node_costs=per_node_costs,
    )


def make_uniform_assignment(graph: OperatorGraph, domain: Domain) -> Dict[str, Domain]:
    return {node.node_id: domain for node in graph.nodes}

