from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from compiler.state_expanded_opt.cost_model import StateExpandedCostModel
from ir.types import DataEdge, OperatorGraph, OperatorNode

from .region_analysis import RegionAnalysisResult
from .region_types import BlockSummary, BoundaryState, SummaryEntry


@dataclass(frozen=True)
class BlockPathDecision:
    block_id: str
    block_kind: str
    input_state: BoundaryState
    output_state: BoundaryState
    incremental_cost_ms: float
    actions: Tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "block_kind": self.block_kind,
            "input_state": self.input_state.as_dict(),
            "output_state": self.output_state.as_dict(),
            "incremental_cost_ms": self.incremental_cost_ms,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class GlobalSolveResult:
    graph_id: str
    supported: bool
    strategy: str
    total_cost_ms: float | None
    start_state: BoundaryState | None
    goal_state: BoundaryState | None
    block_order: Tuple[str, ...]
    block_decisions: Tuple[BlockPathDecision, ...]
    unsupported_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "supported": self.supported,
            "strategy": self.strategy,
            "total_cost_ms": self.total_cost_ms,
            "start_state": None if self.start_state is None else self.start_state.as_dict(),
            "goal_state": None if self.goal_state is None else self.goal_state.as_dict(),
            "block_order": list(self.block_order),
            "block_decisions": [decision.as_dict() for decision in self.block_decisions],
            "unsupported_reason": self.unsupported_reason,
        }


def _unsupported(graph_id: str, block_order: Tuple[str, ...], reason: str) -> GlobalSolveResult:
    return GlobalSolveResult(
        graph_id=graph_id,
        supported=False,
        strategy="block_dp_linear",
        total_cost_ms=None,
        start_state=None,
        goal_state=None,
        block_order=block_order,
        block_decisions=tuple(),
        unsupported_reason=reason,
    )


@dataclass(frozen=True)
class _FrontierKey:
    values: Tuple[Tuple[str, BoundaryState], ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, BoundaryState], frontier_order: Tuple[str, ...]) -> "_FrontierKey":
        return cls(tuple((block_id, mapping[block_id]) for block_id in frontier_order))

    def as_mapping(self) -> Dict[str, BoundaryState]:
        return {block_id: state for block_id, state in self.values}


def _validate_linear_block_graph(region_result: RegionAnalysisResult) -> str | None:
    order = region_result.block_topological_order
    if not order:
        return "Region analysis produced an empty block graph"

    indegree: Dict[str, int] = {block_id: 0 for block_id in order}
    outdegree: Dict[str, int] = {block_id: 0 for block_id in order}
    for src, dst in region_result.block_edges:
        outdegree[src] += 1
        indegree[dst] += 1

    source_count = sum(1 for block_id in order if indegree[block_id] == 0)
    sink_count = sum(1 for block_id in order if outdegree[block_id] == 0)
    if source_count != 1 or sink_count != 1:
        return f"Linear block DP expects exactly 1 source and 1 sink block, got sources={source_count}, sinks={sink_count}"

    for idx, block_id in enumerate(order):
        expected_in = 0 if idx == 0 else 1
        expected_out = 0 if idx == len(order) - 1 else 1
        if indegree[block_id] != expected_in or outdegree[block_id] != expected_out:
            return (
                "Linear block DP currently expects a single block chain after SESE compression; "
                f"block={block_id} has indegree={indegree[block_id]} outdegree={outdegree[block_id]}"
            )
    return None


def _block_adjacency(region_result: RegionAnalysisResult) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    successors: Dict[str, List[str]] = {block_id: [] for block_id in region_result.block_topological_order}
    predecessors: Dict[str, List[str]] = {block_id: [] for block_id in region_result.block_topological_order}
    for src, dst in region_result.block_edges:
        successors[src].append(dst)
        predecessors[dst].append(src)
    for block_id in region_result.block_topological_order:
        successors[block_id].sort()
        predecessors[block_id].sort()
    return successors, predecessors


def _topo_index(block_order: Tuple[str, ...]) -> Dict[str, int]:
    return {block_id: idx for idx, block_id in enumerate(block_order)}


def _frontier_after_index(
    block_order: Tuple[str, ...],
    successors: Mapping[str, List[str]],
    topo_index: Mapping[str, int],
    processed_index: int,
) -> Tuple[str, ...]:
    frontier: List[str] = []
    for idx, block_id in enumerate(block_order[: processed_index + 1]):
        if any(topo_index[succ] > processed_index for succ in successors[block_id]):
            frontier.append(block_id)
    return tuple(frontier)


def _candidate_input_states(
    block_id: str,
    frontier_assignment: Mapping[str, BoundaryState],
    predecessors: Mapping[str, List[str]],
    summary: BlockSummary,
) -> List[BoundaryState]:
    preds = predecessors[block_id]
    if not preds:
        return list(summary.boundary_states)

    pred_states = [frontier_assignment[pred] for pred in preds]
    unique_states = {state for state in pred_states}
    if len(unique_states) == 1:
        state = pred_states[0]
        if state in summary.boundary_states:
            return [state]
    return []


def _node_to_block(region_result: RegionAnalysisResult) -> Dict[str, str]:
    node_to_block: Dict[str, str] = {}
    for block in region_result.blocks:
        for node_id in block.nodes:
            node_to_block[node_id] = block.block_id
    return node_to_block


def _incoming_inter_block_edges(
    graph: OperatorGraph,
    region_result: RegionAnalysisResult,
) -> Dict[Tuple[str, str], List[DataEdge]]:
    node_to_block = _node_to_block(region_result)
    incoming: Dict[Tuple[str, str], List[DataEdge]] = {}
    for edge in graph.edges:
        src_block = node_to_block[edge.src]
        dst_block = node_to_block[edge.dst]
        if src_block == dst_block:
            continue
        incoming.setdefault((src_block, dst_block), []).append(edge)
    return incoming


def _action_dict(
    kind: str,
    node: OperatorNode,
    latency_ms: float,
    from_state: BoundaryState,
    to_state: BoundaryState,
    reason: str,
    edge: DataEdge | None = None,
) -> dict[str, object]:
    payload = {
        "kind": kind,
        "node_id": node.node_id,
        "op_type": node.op_type,
        "latency_ms": float(latency_ms),
        "from_state": from_state.as_dict(),
        "to_state": to_state.as_dict(),
        "reason": reason,
    }
    if edge is not None:
        payload["edge"] = {
            "src": edge.src,
            "dst": edge.dst,
            "tensor_shape": list(edge.tensor_shape),
        }
    return payload


def _compatibility_options_for_pred(
    pred_state: BoundaryState,
    target_state: BoundaryState,
    entry_node: OperatorNode,
    incoming_edges: Sequence[DataEdge],
    cost_model: StateExpandedCostModel,
) -> Tuple[bool, float, Tuple[dict[str, object], ...]]:
    if target_state.domain == "MPC":
        if pred_state.domain == "MPC":
            return True, 0.0, tuple()
        total_cost = 0.0
        actions: List[dict[str, object]] = []
        for edge in incoming_edges:
            cost = cost_model.conversion_cost(edge.tensor_shape, "HE", "MPC").latency_ms
            total_cost += cost
            actions.append(
                _action_dict(
                    "convert_he_to_mpc",
                    entry_node,
                    cost,
                    pred_state,
                    target_state,
                    "join_align_to_mpc",
                    edge=edge,
                )
            )
        return True, total_cost, tuple(actions)

    target_level = target_state.level_bucket
    if target_level is None:
        return False, 0.0, tuple()

    if pred_state.domain == "MPC":
        if target_level != cost_model.max_level():
            return False, 0.0, tuple()
        total_cost = 0.0
        actions: List[dict[str, object]] = []
        for edge in incoming_edges:
            cost = cost_model.conversion_cost(edge.tensor_shape, "MPC", "HE").latency_ms
            total_cost += cost
            actions.append(
                _action_dict(
                    "convert_mpc_to_he",
                    entry_node,
                    cost,
                    pred_state,
                    target_state,
                    "join_align_to_he_fresh",
                    edge=edge,
                )
            )
        return True, total_cost, tuple(actions)

    pred_level = pred_state.level_bucket
    if pred_level == target_level:
        return True, 0.0, tuple()
    if target_level != cost_model.max_level():
        return False, 0.0, tuple()

    total_cost = 0.0
    actions: List[dict[str, object]] = []
    for edge in incoming_edges:
        cost = cost_model.bootstrap_cost(entry_node).latency_ms
        total_cost += cost
        actions.append(
            _action_dict(
                "bootstrap",
                entry_node,
                cost,
                pred_state,
                target_state,
                "join_align_to_he_fresh",
                edge=edge,
            )
        )
    return True, total_cost, tuple(actions)


def _join_compatibility_options(
    block_id: str,
    frontier_assignment: Mapping[str, BoundaryState],
    predecessors: Mapping[str, List[str]],
    summary: BlockSummary,
    node_map: Mapping[str, OperatorNode],
    incoming_edge_map: Mapping[Tuple[str, str], List[DataEdge]],
    cost_model: StateExpandedCostModel,
) -> Dict[BoundaryState, Tuple[float, Tuple[dict[str, object], ...]]]:
    preds = predecessors[block_id]
    if not preds:
        return {state: (0.0, tuple()) for state in summary.boundary_states}

    entry_node = node_map[summary.block.entry]
    options: Dict[BoundaryState, Tuple[float, Tuple[dict[str, object], ...]]] = {}
    for target_state in summary.boundary_states:
        total_cost = 0.0
        actions: List[dict[str, object]] = []
        feasible = True
        for pred_block in preds:
            pred_state = frontier_assignment[pred_block]
            incoming_edges = incoming_edge_map.get((pred_block, block_id), [])
            ok, pred_cost, pred_actions = _compatibility_options_for_pred(
                pred_state=pred_state,
                target_state=target_state,
                entry_node=entry_node,
                incoming_edges=incoming_edges,
                cost_model=cost_model,
            )
            if not ok:
                feasible = False
                break
            total_cost += pred_cost
            actions.extend(pred_actions)
        if feasible:
            options[target_state] = (total_cost, tuple(actions))
    return options


def solve_block_graph_dag(
    region_result: RegionAnalysisResult,
    block_summaries: Mapping[str, BlockSummary],
    graph: OperatorGraph,
    cost_model: StateExpandedCostModel,
) -> GlobalSolveResult:
    block_order = region_result.block_topological_order
    if not block_order:
        return _unsupported(region_result.graph_id, block_order, "Region analysis produced an empty block graph")

    for block_id in block_order:
        summary = block_summaries.get(block_id)
        if summary is None:
            return _unsupported(region_result.graph_id, block_order, f"Missing summary for block {block_id}")
        if not summary.supported:
            reason = summary.unsupported_reason or f"Summary for block {block_id} is unsupported"
            return _unsupported(region_result.graph_id, block_order, reason)

    successors, predecessors = _block_adjacency(region_result)
    node_map = {node.node_id: node for node in graph.nodes}
    incoming_edge_map = _incoming_inter_block_edges(graph, region_result)
    topo_index = _topo_index(block_order)
    frontier_orders: List[Tuple[str, ...]] = [
        _frontier_after_index(block_order, successors, topo_index, idx) for idx in range(len(block_order))
    ]

    empty_key = _FrontierKey(tuple())
    dp: Dict[_FrontierKey, float] = {empty_key: 0.0}
    prev: Dict[Tuple[int, _FrontierKey], Tuple[Tuple[int, _FrontierKey] | None, BlockPathDecision | None]] = {
        (-1, empty_key): (None, None)
    }

    for block_index, block_id in enumerate(block_order):
        summary = block_summaries[block_id]
        frontier_before = tuple() if block_index == 0 else frontier_orders[block_index - 1]
        frontier_after = frontier_orders[block_index]
        entries_by_input: Dict[BoundaryState, List[SummaryEntry]] = {}
        for entry in summary.summary_entries:
            entries_by_input.setdefault(entry.input_state, []).append(entry)

        next_dp: Dict[_FrontierKey, float] = {}
        for frontier_key, cost_so_far in dp.items():
            frontier_assignment = frontier_key.as_mapping()
            compatibility_options = _join_compatibility_options(
                block_id=block_id,
                frontier_assignment=frontier_assignment,
                predecessors=predecessors,
                summary=summary,
                node_map=node_map,
                incoming_edge_map=incoming_edge_map,
                cost_model=cost_model,
            )
            for input_state, (compat_cost, compat_actions) in compatibility_options.items():
                for entry in entries_by_input.get(input_state, []):
                    next_assignment: Dict[str, BoundaryState] = {}
                    for frontier_block in frontier_after:
                        if frontier_block == block_id:
                            next_assignment[frontier_block] = entry.output_state
                        else:
                            next_assignment[frontier_block] = frontier_assignment[frontier_block]
                    next_key = _FrontierKey.from_mapping(next_assignment, frontier_after)
                    new_cost = cost_so_far + compat_cost + entry.total_cost_ms
                    if new_cost + 1e-12 < next_dp.get(next_key, float("inf")):
                        next_dp[next_key] = new_cost
                        prev[(block_index, next_key)] = (
                            (block_index - 1, frontier_key),
                            BlockPathDecision(
                                block_id=block_id,
                                block_kind=summary.block.kind,
                                input_state=entry.input_state,
                                output_state=entry.output_state,
                                incremental_cost_ms=compat_cost + entry.total_cost_ms,
                                actions=compat_actions + tuple(action.as_dict() for action in entry.actions),
                            ),
                        )

        if not next_dp:
            return _unsupported(
                region_result.graph_id,
                block_order,
                f"No feasible DAG block transition found while processing block {block_id}",
            )
        dp = next_dp

    goal_key, total_cost = min(dp.items(), key=lambda item: item[1])
    last_step = len(block_order) - 1
    decisions: List[BlockPathDecision] = []
    cursor = (last_step, goal_key)
    while True:
        parent, decision = prev[cursor]
        if parent is None or decision is None:
            break
        decisions.append(decision)
        cursor = parent
    decisions.reverse()

    start_state = decisions[0].input_state if decisions else None
    goal_state = decisions[-1].output_state if decisions else None
    return GlobalSolveResult(
        graph_id=region_result.graph_id,
        supported=True,
        strategy="block_dp_dag",
        total_cost_ms=total_cost,
        start_state=start_state,
        goal_state=goal_state,
        block_order=block_order,
        block_decisions=tuple(decisions),
    )


def solve_block_graph_linear(
    region_result: RegionAnalysisResult,
    block_summaries: Mapping[str, BlockSummary],
) -> GlobalSolveResult:
    block_order = region_result.block_topological_order
    invalid_reason = _validate_linear_block_graph(region_result)
    if invalid_reason is not None:
        return _unsupported(region_result.graph_id, block_order, invalid_reason)

    for block_id in block_order:
        summary = block_summaries.get(block_id)
        if summary is None:
            return _unsupported(region_result.graph_id, block_order, f"Missing summary for block {block_id}")
        if not summary.supported:
            reason = summary.unsupported_reason or f"Summary for block {block_id} is unsupported"
            return _unsupported(region_result.graph_id, block_order, reason)

    first_summary = block_summaries[block_order[0]]
    dist: Dict[Tuple[int, BoundaryState], float] = {}
    prev: Dict[Tuple[int, BoundaryState], Tuple[Tuple[int, BoundaryState] | None, SummaryEntry | None]] = {}
    for boundary_state in first_summary.boundary_states:
        key = (0, boundary_state)
        dist[key] = 0.0
        prev[key] = (None, None)

    for block_index, block_id in enumerate(block_order):
        summary = block_summaries[block_id]
        next_dist: Dict[Tuple[int, BoundaryState], float] = {}
        next_prev: Dict[Tuple[int, BoundaryState], Tuple[Tuple[int, BoundaryState], SummaryEntry]] = {}

        entries_by_input: Dict[BoundaryState, List[SummaryEntry]] = {}
        for entry in summary.summary_entries:
            entries_by_input.setdefault(entry.input_state, []).append(entry)

        for (_, state), cost_so_far in dist.items():
            for entry in entries_by_input.get(state, []):
                next_key = (block_index + 1, entry.output_state)
                new_cost = cost_so_far + entry.total_cost_ms
                if new_cost + 1e-12 < next_dist.get(next_key, float("inf")):
                    next_dist[next_key] = new_cost
                    next_prev[next_key] = ((block_index, state), entry)

        if not next_dist:
            return _unsupported(
                region_result.graph_id,
                block_order,
                f"No feasible block-summary transition found while processing block {block_id}",
            )
        dist = next_dist
        prev.update(next_prev)

    goal_key, total_cost = min(dist.items(), key=lambda item: item[1])
    goal_state = goal_key[1]

    decisions: List[BlockPathDecision] = []
    cursor: Tuple[int, BoundaryState] = goal_key
    while True:
        parent, entry = prev[cursor]
        if parent is None or entry is None:
            start_state = cursor[1]
            break
        block_index = cursor[0] - 1
        block_id = block_order[block_index]
        block_summary = block_summaries[block_id]
        decisions.append(
            BlockPathDecision(
                block_id=block_id,
                block_kind=block_summary.block.kind,
                input_state=entry.input_state,
                output_state=entry.output_state,
                incremental_cost_ms=entry.total_cost_ms,
                actions=tuple(action.as_dict() for action in entry.actions),
            )
        )
        cursor = parent
    decisions.reverse()

    return GlobalSolveResult(
        graph_id=region_result.graph_id,
        supported=True,
        strategy="block_dp_linear",
        total_cost_ms=total_cost,
        start_state=start_state,
        goal_state=goal_state,
        block_order=block_order,
        block_decisions=tuple(decisions),
    )
