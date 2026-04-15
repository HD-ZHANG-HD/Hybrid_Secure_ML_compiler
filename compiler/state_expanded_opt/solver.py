from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product
from typing import Dict, List, Tuple

from compiler.capability_checker import default_capability_checker
from compiler.min_cut.profiler_db import Domain
from compiler.min_cut.runtime_plan_adapter import resolve_method_name
from ir.types import DataEdge, OperatorGraph, OperatorNode

from .cost_model import StateExpandedCostModel
from .graph_model import GraphView
from .state_space import ActionRecord, NodeExecution, State


@dataclass
class SolverResult:
    graph_id: str
    strategy: str
    best_path: List[dict[str, object]]
    total_cost_ms: float
    start_state: dict[str, object]
    goal_state: dict[str, object]
    state_trace: List[dict[str, object]]
    decision_trace: List[dict[str, object]]
    cost_breakdown: Dict[str, float]
    final_assignment: Dict[str, Dict[str, object]]
    stage_summaries: List[Dict[str, object]]
    actions: List[ActionRecord]


@dataclass
class _ChainPrev:
    prev_state: State | None
    actions: List[ActionRecord]


def _method_valid(node: OperatorNode, domain: Domain) -> bool:
    method = resolve_method_name(node.op_type, domain)
    return default_capability_checker.is_method_valid(node.op_type, method, node.input_shape, node.attributes)


def _he_feasible(node: OperatorNode, cost_model: StateExpandedCostModel) -> bool:
    """Per-node HE feasibility gate (Section D).

    Consults the method's `cost_signature()` via `cost_model.feasible()` so
    that HE branches for BERT-base LayerNorm (B*S>16), out-of-contract
    Attention (S>128), etc. are pruned at graph construction rather than
    paid as infinite cost. Falls back to `_method_valid` for nodes whose
    method has no registered cost signature.
    """
    if not _method_valid(node, "HE"):
        return False
    feasible_fn = getattr(cost_model, "feasible", None)
    if feasible_fn is None:
        return True
    try:
        return bool(feasible_fn(node, "HE"))
    except Exception:
        return True


def _he_method_bootstrap_supported(node: OperatorNode) -> bool:
    """Return True if this operator's HE method reports bootstrap support.

    Used by the chain solver: when the direct HE-execute transition is
    infeasible for level reasons AND bootstrap is unsupported, the solver
    must detour through MPC rather than emit a fake bootstrap step.
    """
    try:
        from compiler.cost_model import cost_signature_for_method, default_method_for

        method = default_method_for(node.op_type, "HE")
        if method is None:
            return False
        sig = cost_signature_for_method(
            node.op_type, method, node.input_shape, node.output_shape
        )
        if sig is None:
            return False
        return bool(sig.he_bootstrap_supported)
    except Exception:
        return False


def _make_action(
    step_id: int,
    kind: str,
    node: OperatorNode,
    latency_ms: float,
    reason: str,
    from_state: State,
    to_state: State,
    from_node: str | None = None,
    to_node: str | None = None,
    tensor_shape: tuple[int, ...] | None = None,
) -> ActionRecord:
    return ActionRecord(
        step_id=f"s{step_id}",
        kind=kind,
        node_id=node.node_id,
        op_type=node.op_type,
        estimated_latency_ms=float(latency_ms),
        reason=reason,
        from_state=from_state.as_dict(),
        to_state=to_state.as_dict(),
        from_node=from_node,
        to_node=to_node,
        from_domain=from_state.domain,
        to_domain=to_state.domain,
        tensor_shape=tensor_shape,
        level_before=from_state.level_bucket,
        level_after=to_state.level_bucket,
    )


def solve_state_expanded(graph: OperatorGraph, cost_model: StateExpandedCostModel) -> SolverResult:
    view = GraphView(graph)
    if view.is_chain():
        return _solve_chain_exact(graph, view, cost_model)
    return _solve_stage_local_dag(graph, view, cost_model)


def _solve_chain_exact(graph: OperatorGraph, view: GraphView, cost_model: StateExpandedCostModel) -> SolverResult:
    nodes = [view.node_map[node_id] for node_id in view.topological_order]
    max_level = cost_model.max_level()
    start_states = [State(position=1, domain="HE", level_bucket=max_level), State(position=1, domain="MPC", level_bucket=None)]
    dist: Dict[State, float] = {state: 0.0 for state in start_states}
    prev: Dict[State, _ChainPrev] = {state: _ChainPrev(prev_state=None, actions=[]) for state in start_states}
    queue: List[Tuple[float, int, State]] = []
    counter = 0
    for state in start_states:
        heapq.heappush(queue, (0.0, counter, state))
        counter += 1

    next_step_id = 1
    while queue:
        current_cost, _, state = heapq.heappop(queue)
        if current_cost > dist[state] + 1e-12:
            continue
        if state.position == len(nodes) + 1:
            continue
        node = nodes[state.position - 1]
        transitions, next_step_id = _enumerate_chain_transitions(state, node, cost_model, next_step_id)
        for next_state, actions, edge_cost in transitions:
            new_cost = current_cost + edge_cost
            if new_cost + 1e-12 < dist.get(next_state, float("inf")):
                dist[next_state] = new_cost
                prev[next_state] = _ChainPrev(prev_state=state, actions=actions)
                heapq.heappush(queue, (new_cost, counter, next_state))
                counter += 1

    goal_candidates = [state for state in dist if state.position == len(nodes) + 1]
    if not goal_candidates:
        raise ValueError("No feasible state-expanded plan found for chain graph")
    goal = min(goal_candidates, key=lambda item: dist[item])

    actions: List[ActionRecord] = []
    states: List[State] = [goal]
    cursor = goal
    while True:
        chain_prev = prev[cursor]
        if chain_prev.prev_state is None:
            break
        actions = chain_prev.actions + actions
        cursor = chain_prev.prev_state
        states.append(cursor)
    states.reverse()
    return _assemble_result(graph, "chain_exact", actions, states, dist[goal], [])


def _enumerate_chain_transitions(
    state: State,
    node: OperatorNode,
    cost_model: StateExpandedCostModel,
    step_id: int,
) -> tuple[List[tuple[State, List[ActionRecord], float]], int]:
    transitions: List[tuple[State, List[ActionRecord], float]] = []
    # Feasibility gate: the direct HE-execute transition is only
    # considered when the method can actually run at this node's shape
    # (cost_signature().feasible). Previously an infeasible HE branch
    # would silently price through.
    he_feasible = _he_feasible(node, cost_model)
    if state.domain == "HE" and he_feasible:
        delta = cost_model.level_delta(node, "HE")
        op_cost = cost_model.operator_cost(node, "HE").latency_ms
        # Strict l >= delta enforcement: only emit the direct execute
        # transition when the level budget actually covers the op.
        if state.level_bucket is not None and state.level_bucket >= delta:
            next_state = State(position=state.position + 1, domain="HE", level_bucket=state.level_bucket - delta)
            transitions.append((
                next_state,
                [_make_action(step_id, "execute_he", node, op_cost, "execute", state, next_state)],
                op_cost,
            ))
            step_id += 1
        # Bootstrap-then-execute path is only offered when the op's HE
        # method actually supports in-place bootstrap. If not, the solver
        # must pay the HE->MPC->HE detour (handled below via the
        # conversion transition).
        if _he_method_bootstrap_supported(node):
            bootstrap_cost = cost_model.bootstrap_cost(node).latency_ms
            boot_state = State(position=state.position, domain="HE", level_bucket=cost_model.max_level())
            next_state = State(position=state.position + 1, domain="HE", level_bucket=cost_model.max_level() - delta)
            transitions.append((
                next_state,
                [
                    _make_action(step_id, "bootstrap", node, bootstrap_cost, "budget_reset", state, boot_state),
                    _make_action(step_id + 1, "execute_he", node, op_cost, "execute_after_bootstrap", boot_state, next_state),
                ],
                bootstrap_cost + op_cost,
            ))
            step_id += 2
    if state.domain == "HE":
        # HE->MPC->execute detour is always considered when an MPC method
        # exists, regardless of HE feasibility; this is how the solver
        # escapes infeasible HE shapes (Section D).
        conv_cost = cost_model.conversion_cost(node.input_shape, "HE", "MPC").latency_ms
        conv_state = State(position=state.position, domain="MPC", level_bucket=None)
        next_state = State(position=state.position + 1, domain="MPC", level_bucket=None)
        mpc_cost = cost_model.operator_cost(node, "MPC").latency_ms if _method_valid(node, "MPC") else None
        if mpc_cost is not None:
            transitions.append((
                next_state,
                [
                    _make_action(step_id, "conversion", node, conv_cost, "domain_switch", state, conv_state),
                    _make_action(step_id + 1, "execute_mpc", node, mpc_cost, "execute_after_conversion", conv_state, next_state),
                ],
                conv_cost + mpc_cost,
            ))
            step_id += 2

    if state.domain == "MPC" and _method_valid(node, "MPC"):
        op_cost = cost_model.operator_cost(node, "MPC").latency_ms
        next_state = State(position=state.position + 1, domain="MPC", level_bucket=None)
        transitions.append((
            next_state,
            [_make_action(step_id, "execute_mpc", node, op_cost, "execute", state, next_state)],
            op_cost,
        ))
        step_id += 1
    if state.domain == "MPC" and he_feasible:
        conv_cost = cost_model.conversion_cost(node.input_shape, "MPC", "HE").latency_ms
        delta = cost_model.level_delta(node, "HE")
        op_cost = cost_model.operator_cost(node, "HE").latency_ms
        he_state = State(position=state.position, domain="HE", level_bucket=cost_model.max_level())
        next_state = State(position=state.position + 1, domain="HE", level_bucket=cost_model.max_level() - delta)
        transitions.append((
            next_state,
            [
                _make_action(step_id, "conversion", node, conv_cost, "domain_switch", state, he_state),
                _make_action(step_id + 1, "execute_he", node, op_cost, "execute_after_conversion", he_state, next_state),
            ],
            conv_cost + op_cost,
        ))
        step_id += 2
    return transitions, step_id


def _solve_stage_local_dag(graph: OperatorGraph, view: GraphView, cost_model: StateExpandedCostModel) -> SolverResult:
    node_states: Dict[str, State] = {}
    node_steps: Dict[str, List[ActionRecord]] = {}
    node_costs: Dict[str, float] = {}
    node_exec: Dict[str, NodeExecution] = {}
    stage_summaries: List[Dict[str, object]] = []
    step_id = 1

    for stage_nodes in view.stages():
        stage_summary: Dict[str, object] = {"stage_nodes": list(stage_nodes), "decisions": []}
        for node_id in stage_nodes:
            node = view.node_map[node_id]
            result = _choose_stage_local_node_plan(node, view.incoming.get(node_id, []), node_states, cost_model, step_id)
            chosen_state, actions, total_cost, execution, option_summary, step_id = result
            node_states[node_id] = chosen_state
            node_steps[node_id] = actions
            node_costs[node_id] = total_cost
            node_exec[node_id] = execution
            stage_summary["decisions"].append(option_summary)
        stage_summaries.append(stage_summary)

    actions: List[ActionRecord] = []
    for node_id in view.topological_order:
        actions.extend(node_steps[node_id])
    states = [State(position=1, domain="HE", level_bucket=cost_model.max_level())]
    states.extend(node_states[node_id] for node_id in view.topological_order)
    total_cost = sum(node_costs.values())
    return _assemble_result(graph, "stage_local_dag", actions, states, total_cost, stage_summaries, node_exec=node_exec)


def _choose_stage_local_node_plan(
    node: OperatorNode,
    incoming_edges: List[DataEdge],
    node_states: Dict[str, State],
    cost_model: StateExpandedCostModel,
    step_id: int,
) -> tuple[State, List[ActionRecord], float, NodeExecution, Dict[str, object], int]:
    candidate_domains: List[Domain] = []
    if _he_feasible(node, cost_model):
        candidate_domains.append("HE")
    if _method_valid(node, "MPC"):
        candidate_domains.append("MPC")
    if not candidate_domains:
        raise ValueError(f"No valid execution domain for node {node.node_id}:{node.op_type}")

    start_state = State(position=0, domain="HE", level_bucket=cost_model.max_level())
    if incoming_edges:
        start_state = node_states[incoming_edges[0].src]

    best: tuple[float, State, List[ActionRecord], NodeExecution, Dict[str, object], int] | None = None
    incoming_states = [node_states[edge.src] for edge in incoming_edges]
    aligned_he_levels = [state.level_bucket for state in incoming_states if state.domain == "HE" and state.level_bucket is not None]

    for domain in candidate_domains:
        actions: List[ActionRecord] = []
        local_step = step_id
        incremental_cost = 0.0
        aligned_level: int | None = None
        if not incoming_edges:
            if domain == "HE":
                aligned_level = cost_model.max_level()
            else:
                aligned_level = None
        for edge in incoming_edges:
            pred_state = node_states[edge.src]
            if pred_state.domain == domain:
                if domain == "HE" and pred_state.level_bucket is not None:
                    aligned_level = pred_state.level_bucket if aligned_level is None else min(aligned_level, pred_state.level_bucket)
                continue
            to_level = None if domain == "MPC" else cost_model.max_level()
            conv_state = State(position=0, domain=domain, level_bucket=to_level)
            conv_cost = cost_model.conversion_cost(edge.tensor_shape, pred_state.domain, domain).latency_ms
            actions.append(
                _make_action(
                    local_step,
                    "conversion",
                    node,
                    conv_cost,
                    "merge_align" if len(incoming_edges) > 1 else "domain_switch",
                    pred_state,
                    conv_state,
                    from_node=edge.src,
                    to_node=edge.dst,
                    tensor_shape=edge.tensor_shape,
                )
            )
            local_step += 1
            incremental_cost += conv_cost
            if domain == "HE":
                aligned_level = to_level if aligned_level is None else min(aligned_level, to_level)

        if domain == "HE":
            delta = cost_model.level_delta(node, "HE")
            if aligned_level is None:
                aligned_level = cost_model.max_level()
            level_before = aligned_level
            if aligned_level < delta:
                boot_state = State(position=0, domain="HE", level_bucket=cost_model.max_level())
                boot_cost = cost_model.bootstrap_cost(node).latency_ms
                actions.append(_make_action(local_step, "bootstrap", node, boot_cost, "budget_reset", start_state, boot_state))
                local_step += 1
                incremental_cost += boot_cost
                aligned_level = cost_model.max_level()
            op_cost = cost_model.operator_cost(node, "HE").latency_ms
            next_state = State(position=0, domain="HE", level_bucket=aligned_level - delta)
            actions.append(_make_action(local_step, "execute_he", node, op_cost, "execute", State(0, "HE", aligned_level), next_state))
            local_step += 1
            incremental_cost += op_cost
            execution = NodeExecution(
                node_id=node.node_id,
                domain="HE",
                level_before=aligned_level,
                level_after=aligned_level - delta,
                incremental_cost_ms=incremental_cost,
            )
            option_summary = {
                "node_id": node.node_id,
                "chosen_domain_candidate": "HE",
                "incoming_domains": [state.domain for state in incoming_states],
                "level_before": level_before,
                "level_after": aligned_level - delta,
                "estimated_cost_ms": incremental_cost,
            }
        else:
            op_cost = cost_model.operator_cost(node, "MPC").latency_ms
            next_state = State(position=0, domain="MPC", level_bucket=None)
            actions.append(_make_action(local_step, "execute_mpc", node, op_cost, "execute", State(0, "MPC", None), next_state))
            local_step += 1
            incremental_cost += op_cost
            execution = NodeExecution(
                node_id=node.node_id,
                domain="MPC",
                level_before=None,
                level_after=None,
                incremental_cost_ms=incremental_cost,
            )
            option_summary = {
                "node_id": node.node_id,
                "chosen_domain_candidate": "MPC",
                "incoming_domains": [state.domain for state in incoming_states],
                "level_before": None,
                "level_after": None,
                "estimated_cost_ms": incremental_cost,
            }
        if best is None or incremental_cost < best[0]:
            best = (incremental_cost, next_state, actions, execution, option_summary, local_step)

    assert best is not None
    return best[1], best[2], best[0], best[3], best[4], best[5]


def _assemble_result(
    graph: OperatorGraph,
    strategy: str,
    actions: List[ActionRecord],
    states: List[State],
    total_cost: float,
    stage_summaries: List[Dict[str, object]],
    node_exec: Dict[str, NodeExecution] | None = None,
) -> SolverResult:
    normalized_actions = [
        ActionRecord(
            step_id=f"s{idx}",
            kind=action.kind,
            node_id=action.node_id,
            op_type=action.op_type,
            estimated_latency_ms=action.estimated_latency_ms,
            reason=action.reason,
            from_state=action.from_state,
            to_state=action.to_state,
            from_node=action.from_node,
            to_node=action.to_node,
            from_domain=action.from_domain,
            to_domain=action.to_domain,
            tensor_shape=action.tensor_shape,
            level_before=action.level_before,
            level_after=action.level_after,
        )
        for idx, action in enumerate(actions, start=1)
    ]
    operator_cost_ms = sum(action.estimated_latency_ms for action in normalized_actions if action.kind in {"execute_he", "execute_mpc"})
    conversion_cost_ms = sum(action.estimated_latency_ms for action in normalized_actions if action.kind == "conversion")
    bootstrap_cost_ms = sum(action.estimated_latency_ms for action in normalized_actions if action.kind == "bootstrap")
    final_assignment: Dict[str, Dict[str, object]] = {}
    for action in normalized_actions:
        if action.kind not in {"execute_he", "execute_mpc"}:
            continue
        final_assignment[action.node_id] = {
            "domain": "HE" if action.kind == "execute_he" else "MPC",
            "level_before": action.level_before,
            "level_after": action.level_after,
        }
    if node_exec:
        for node_id, execution in node_exec.items():
            final_assignment[node_id] = {
                "domain": execution.domain,
                "level_before": execution.level_before,
                "level_after": execution.level_after,
            }
    return SolverResult(
        graph_id=graph.graph_id,
        strategy=strategy,
        best_path=[state.as_dict() for state in states],
        total_cost_ms=float(total_cost),
        start_state=states[0].as_dict(),
        goal_state=states[-1].as_dict(),
        state_trace=[state.as_dict() for state in states],
        decision_trace=[
            {
                "step_id": action.step_id,
                "kind": action.kind,
                "node_id": action.node_id,
                "reason": action.reason,
                "estimated_latency_ms": action.estimated_latency_ms,
            }
            for action in normalized_actions
        ],
        cost_breakdown={
            "operator_cost_ms": operator_cost_ms,
            "conversion_cost_ms": conversion_cost_ms,
            "bootstrap_cost_ms": bootstrap_cost_ms,
            "total_cost_ms": float(total_cost),
        },
        final_assignment=final_assignment,
        stage_summaries=stage_summaries,
        actions=normalized_actions,
    )
