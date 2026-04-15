from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Mapping, Tuple

from compiler.capability_checker import default_capability_checker
from compiler.min_cut.profiler_db import Domain
from compiler.min_cut.runtime_plan_adapter import resolve_method_name
from compiler.state_expanded_opt.cost_model import StateExpandedCostModel
from ir.types import OperatorGraph, OperatorNode

from .region_analysis import RegionAnalysisResult, SESEBlock
from .region_types import BlockSummary, BoundaryState, SummaryAction, SummaryEntry


@dataclass(frozen=True)
class _ChainState:
    index: int
    boundary: BoundaryState


@dataclass(frozen=True)
class _BranchPlan:
    producer: OperatorNode
    output_state: BoundaryState
    total_cost_ms: float
    actions: Tuple[SummaryAction, ...]


def enumerate_boundary_states(max_level_bucket: int) -> Tuple[BoundaryState, ...]:
    states = [BoundaryState(domain="HE", level_bucket=level) for level in range(max_level_bucket + 1)]
    states.append(BoundaryState(domain="MPC", level_bucket=None))
    return tuple(states)


def _method_valid(node: OperatorNode, domain: Domain) -> bool:
    method = resolve_method_name(node.op_type, domain)
    return default_capability_checker.is_method_valid(node.op_type, method, node.input_shape, node.attributes)


def _make_action(
    kind: str,
    node: OperatorNode,
    latency_ms: float,
    from_state: BoundaryState,
    to_state: BoundaryState,
    reason: str,
) -> SummaryAction:
    return SummaryAction(
        kind=kind,
        node_id=node.node_id,
        op_type=node.op_type,
        latency_ms=float(latency_ms),
        from_state=from_state,
        to_state=to_state,
        reason=reason,
    )


def _enumerate_chain_transitions(
    node: OperatorNode,
    current_state: BoundaryState,
    cost_model: StateExpandedCostModel,
) -> List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]]:
    transitions: List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]] = []
    max_level = cost_model.max_level()

    if current_state.domain == "HE" and _method_valid(node, "HE"):
        delta = cost_model.level_delta(node, "HE")
        op_cost = cost_model.operator_cost(node, "HE").latency_ms
        if current_state.level_bucket is not None and current_state.level_bucket >= delta:
            next_state = BoundaryState(domain="HE", level_bucket=current_state.level_bucket - delta)
            transitions.append((
                next_state,
                (_make_action("execute_he", node, op_cost, current_state, next_state, "execute"),),
                op_cost,
            ))

        if max_level >= delta:
            boot_state = BoundaryState(domain="HE", level_bucket=max_level)
            next_state = BoundaryState(domain="HE", level_bucket=max_level - delta)
            bootstrap_cost = cost_model.bootstrap_cost(node).latency_ms
            transitions.append((
                next_state,
                (
                    _make_action("bootstrap", node, bootstrap_cost, current_state, boot_state, "budget_reset"),
                    _make_action("execute_he", node, op_cost, boot_state, next_state, "execute_after_bootstrap"),
                ),
                bootstrap_cost + op_cost,
            ))

        if _method_valid(node, "MPC"):
            conv_cost = cost_model.conversion_cost(node.input_shape, "HE", "MPC").latency_ms
            mpc_state = BoundaryState(domain="MPC", level_bucket=None)
            mpc_cost = cost_model.operator_cost(node, "MPC").latency_ms
            transitions.append((
                mpc_state,
                (
                    _make_action("convert_he_to_mpc", node, conv_cost, current_state, mpc_state, "domain_switch"),
                    _make_action("execute_mpc", node, mpc_cost, mpc_state, mpc_state, "execute_after_conversion"),
                ),
                conv_cost + mpc_cost,
            ))

    if current_state.domain == "MPC" and _method_valid(node, "MPC"):
        op_cost = cost_model.operator_cost(node, "MPC").latency_ms
        next_state = BoundaryState(domain="MPC", level_bucket=None)
        transitions.append((
            next_state,
            (_make_action("execute_mpc", node, op_cost, current_state, next_state, "execute"),),
            op_cost,
        ))

    if current_state.domain == "MPC" and _method_valid(node, "HE"):
        delta = cost_model.level_delta(node, "HE")
        if max_level >= delta:
            conv_cost = cost_model.conversion_cost(node.input_shape, "MPC", "HE").latency_ms
            he_state = BoundaryState(domain="HE", level_bucket=max_level)
            next_state = BoundaryState(domain="HE", level_bucket=max_level - delta)
            op_cost = cost_model.operator_cost(node, "HE").latency_ms
            transitions.append((
                next_state,
                (
                    _make_action("convert_mpc_to_he", node, conv_cost, current_state, he_state, "domain_switch"),
                    _make_action("execute_he", node, op_cost, he_state, next_state, "execute_after_conversion"),
                ),
                conv_cost + op_cost,
            ))

    return transitions


def _summarize_chain_block(
    nodes: Iterable[OperatorNode],
    block: SESEBlock,
    cost_model: StateExpandedCostModel,
) -> BlockSummary:
    chain_nodes = list(nodes)
    boundary_states = enumerate_boundary_states(cost_model.max_level())
    entries: List[SummaryEntry] = []

    for input_state in boundary_states:
        start = _ChainState(index=0, boundary=input_state)
        dist: Dict[_ChainState, float] = {start: 0.0}
        prev: Dict[_ChainState, Tuple[_ChainState | None, Tuple[SummaryAction, ...]]] = {start: (None, tuple())}
        queue: List[Tuple[float, int, _ChainState]] = [(0.0, 0, start)]
        counter = 1

        while queue:
            current_cost, _, state = heapq.heappop(queue)
            if current_cost > dist[state] + 1e-12:
                continue
            if state.index == len(chain_nodes):
                continue
            node = chain_nodes[state.index]
            for next_boundary, actions, edge_cost in _enumerate_chain_transitions(node, state.boundary, cost_model):
                next_state = _ChainState(index=state.index + 1, boundary=next_boundary)
                new_cost = current_cost + edge_cost
                if new_cost + 1e-12 < dist.get(next_state, float("inf")):
                    dist[next_state] = new_cost
                    prev[next_state] = (state, actions)
                    heapq.heappush(queue, (new_cost, counter, next_state))
                    counter += 1

        for output_state in boundary_states:
            goal = _ChainState(index=len(chain_nodes), boundary=output_state)
            if goal not in dist:
                continue
            actions: List[SummaryAction] = []
            cursor = goal
            while True:
                parent, edge_actions = prev[cursor]
                if parent is None:
                    break
                actions = list(edge_actions) + actions
                cursor = parent
            entries.append(
                SummaryEntry(
                    input_state=input_state,
                    output_state=output_state,
                    total_cost_ms=dist[goal],
                    actions=tuple(actions),
                )
            )

    return BlockSummary(
        block=block,
        boundary_states=boundary_states,
        supported=True,
        summary_entries=tuple(entries),
    )


def _solve_chain_from_start(
    nodes: Iterable[OperatorNode],
    start_state: BoundaryState,
    cost_model: StateExpandedCostModel,
) -> Dict[BoundaryState, SummaryEntry]:
    chain_nodes = list(nodes)
    start = _ChainState(index=0, boundary=start_state)
    dist: Dict[_ChainState, float] = {start: 0.0}
    prev: Dict[_ChainState, Tuple[_ChainState | None, Tuple[SummaryAction, ...]]] = {start: (None, tuple())}
    queue: List[Tuple[float, int, _ChainState]] = [(0.0, 0, start)]
    counter = 1

    while queue:
        current_cost, _, state = heapq.heappop(queue)
        if current_cost > dist[state] + 1e-12:
            continue
        if state.index == len(chain_nodes):
            continue
        node = chain_nodes[state.index]
        for next_boundary, actions, edge_cost in _enumerate_chain_transitions(node, state.boundary, cost_model):
            next_state = _ChainState(index=state.index + 1, boundary=next_boundary)
            new_cost = current_cost + edge_cost
            if new_cost + 1e-12 < dist.get(next_state, float("inf")):
                dist[next_state] = new_cost
                prev[next_state] = (state, actions)
                heapq.heappush(queue, (new_cost, counter, next_state))
                counter += 1

    results: Dict[BoundaryState, SummaryEntry] = {}
    boundary_states = enumerate_boundary_states(cost_model.max_level())
    for output_state in boundary_states:
        goal = _ChainState(index=len(chain_nodes), boundary=output_state)
        if goal not in dist:
            continue
        actions: List[SummaryAction] = []
        cursor = goal
        while True:
            parent, edge_actions = prev[cursor]
            if parent is None:
                break
            actions = list(edge_actions) + actions
            cursor = parent
        results[output_state] = SummaryEntry(
            input_state=start_state,
            output_state=output_state,
            total_cost_ms=dist[goal],
            actions=tuple(actions),
        )
    return results


def _enumerate_block_paths(
    block: SESEBlock,
    successors: Mapping[str, List[str]],
) -> List[Tuple[str, ...]]:
    block_nodes = set(block.nodes)
    paths: List[Tuple[str, ...]] = []

    def dfs(node_id: str, path: List[str]) -> None:
        if node_id == block.exit:
            paths.append(tuple(path))
            return
        for succ in successors.get(node_id, []):
            if succ not in block_nodes:
                continue
            if succ in path:
                continue
            dfs(succ, path + [succ])

    dfs(block.entry, [block.entry])
    return paths


def _dedupe_he_options(
    options: Iterable[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]],
) -> List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]]:
    best: Dict[BoundaryState, Tuple[Tuple[SummaryAction, ...], float]] = {}
    for state, actions, cost in options:
        current = best.get(state)
        if current is None or cost + 1e-12 < current[1]:
            best[state] = (actions, cost)
    return [(state, actions, cost) for state, (actions, cost) in best.items()]


def _he_alignment_options(
    branch_name: str,
    producer: OperatorNode,
    current_state: BoundaryState,
    merge_node: OperatorNode,
    cost_model: StateExpandedCostModel,
) -> List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]]:
    if current_state.domain == "MPC":
        next_state = BoundaryState(domain="HE", level_bucket=cost_model.max_level())
        conv_cost = cost_model.conversion_cost(producer.output_shape, "MPC", "HE").latency_ms
        return [(
            next_state,
            (
                _make_action(
                    "convert_mpc_to_he",
                    merge_node,
                    conv_cost,
                    current_state,
                    next_state,
                    f"align_{branch_name}_branch_to_he",
                ),
            ),
            conv_cost,
        )]

    options: List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]] = [(current_state, tuple(), 0.0)]
    boot_state = BoundaryState(domain="HE", level_bucket=cost_model.max_level())
    if boot_state != current_state:
        bootstrap_cost = cost_model.bootstrap_cost(merge_node).latency_ms
        options.append((
            boot_state,
            (
                _make_action(
                    "bootstrap",
                    merge_node,
                    bootstrap_cost,
                    current_state,
                    boot_state,
                    f"align_{branch_name}_branch_budget",
                ),
            ),
            bootstrap_cost,
        ))
    return _dedupe_he_options(options)


def _enumerate_merge_transitions(
    left_plan: _BranchPlan,
    right_plan: _BranchPlan,
    merge_node: OperatorNode,
    cost_model: StateExpandedCostModel,
) -> List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]]:
    transitions: List[Tuple[BoundaryState, Tuple[SummaryAction, ...], float]] = []

    if _method_valid(merge_node, "MPC"):
        actions: List[SummaryAction] = []
        total_cost = 0.0
        if left_plan.output_state.domain == "HE":
            left_mpc = BoundaryState(domain="MPC", level_bucket=None)
            conv_cost = cost_model.conversion_cost(left_plan.producer.output_shape, "HE", "MPC").latency_ms
            actions.append(
                _make_action(
                    "convert_he_to_mpc",
                    merge_node,
                    conv_cost,
                    left_plan.output_state,
                    left_mpc,
                    "align_left_branch_to_mpc",
                )
            )
            total_cost += conv_cost
        if right_plan.output_state.domain == "HE":
            right_mpc = BoundaryState(domain="MPC", level_bucket=None)
            conv_cost = cost_model.conversion_cost(right_plan.producer.output_shape, "HE", "MPC").latency_ms
            actions.append(
                _make_action(
                    "convert_he_to_mpc",
                    merge_node,
                    conv_cost,
                    right_plan.output_state,
                    right_mpc,
                    "align_right_branch_to_mpc",
                )
            )
            total_cost += conv_cost
        out_state = BoundaryState(domain="MPC", level_bucket=None)
        op_cost = cost_model.operator_cost(merge_node, "MPC").latency_ms
        actions.append(_make_action("execute_mpc", merge_node, op_cost, out_state, out_state, "merge_execute"))
        transitions.append((out_state, tuple(actions), total_cost + op_cost))

    if _method_valid(merge_node, "HE"):
        delta = cost_model.level_delta(merge_node, "HE")
        op_cost = cost_model.operator_cost(merge_node, "HE").latency_ms
        left_options = _he_alignment_options("left", left_plan.producer, left_plan.output_state, merge_node, cost_model)
        right_options = _he_alignment_options("right", right_plan.producer, right_plan.output_state, merge_node, cost_model)
        for left_state, left_actions, left_cost in left_options:
            for right_state, right_actions, right_cost in right_options:
                assert left_state.level_bucket is not None
                assert right_state.level_bucket is not None
                available_level = min(left_state.level_bucket, right_state.level_bucket)
                if available_level < delta:
                    continue
                out_state = BoundaryState(domain="HE", level_bucket=available_level - delta)
                actions = list(left_actions) + list(right_actions)
                actions.append(_make_action("execute_he", merge_node, op_cost, BoundaryState("HE", available_level), out_state, "merge_execute"))
                transitions.append((out_state, tuple(actions), left_cost + right_cost + op_cost))

    best: Dict[BoundaryState, Tuple[Tuple[SummaryAction, ...], float]] = {}
    for out_state, actions, total_cost in transitions:
        current = best.get(out_state)
        if current is None or total_cost + 1e-12 < current[1]:
            best[out_state] = (actions, total_cost)
    return [(state, actions, cost) for state, (actions, cost) in best.items()]


def _summarize_residual_block(
    graph: OperatorGraph,
    node_map: Mapping[str, OperatorNode],
    block: SESEBlock,
    cost_model: StateExpandedCostModel,
) -> BlockSummary:
    boundary_states = enumerate_boundary_states(cost_model.max_level())

    succ_in_block: Dict[str, List[str]] = {}
    block_nodes = set(block.nodes)
    for edge in graph.edges:
        if edge.src in block_nodes and edge.dst in block_nodes:
            succ_in_block.setdefault(edge.src, []).append(edge.dst)

    paths = _enumerate_block_paths(block, succ_in_block)
    if len(paths) != 2:
        return BlockSummary(
            block=block,
            boundary_states=boundary_states,
            supported=False,
            summary_entries=tuple(),
            unsupported_reason=f"Residual solver currently expects exactly 2 entry->exit paths, got {len(paths)}",
        )

    branch_paths = [path[1:-1] for path in paths]
    left_nodes = set(branch_paths[0])
    right_nodes = set(branch_paths[1])
    if left_nodes & right_nodes:
        return BlockSummary(
            block=block,
            boundary_states=boundary_states,
            supported=False,
            summary_entries=tuple(),
            unsupported_reason="Residual solver currently expects the 2 branches to be node-disjoint except for entry/exit",
        )

    entry_node = node_map[block.entry]
    merge_node = node_map[block.exit]
    best_entries: Dict[Tuple[BoundaryState, BoundaryState], SummaryEntry] = {}

    for input_state in boundary_states:
        entry_results = _solve_chain_from_start([entry_node], input_state, cost_model)
        for entry_mid_state, entry_summary in entry_results.items():
            branch_plans_per_path: List[List[_BranchPlan]] = []
            for path_nodes in branch_paths:
                if not path_nodes:
                    branch_plans_per_path.append([
                        _BranchPlan(
                            producer=entry_node,
                            output_state=entry_mid_state,
                            total_cost_ms=0.0,
                            actions=tuple(),
                        )
                    ])
                    continue
                chain_nodes = [node_map[node_id] for node_id in path_nodes]
                branch_results = _solve_chain_from_start(chain_nodes, entry_mid_state, cost_model)
                branch_plans_per_path.append([
                    _BranchPlan(
                        producer=chain_nodes[-1],
                        output_state=summary.output_state,
                        total_cost_ms=summary.total_cost_ms,
                        actions=summary.actions,
                    )
                    for summary in branch_results.values()
                ])

            if len(branch_plans_per_path) != 2:
                continue
            for left_plan in branch_plans_per_path[0]:
                for right_plan in branch_plans_per_path[1]:
                    for out_state, merge_actions, merge_cost in _enumerate_merge_transitions(
                        left_plan,
                        right_plan,
                        merge_node,
                        cost_model,
                    ):
                        total_cost = (
                            entry_summary.total_cost_ms
                            + left_plan.total_cost_ms
                            + right_plan.total_cost_ms
                            + merge_cost
                        )
                        actions = (
                            entry_summary.actions
                            + left_plan.actions
                            + right_plan.actions
                            + merge_actions
                        )
                        key = (input_state, out_state)
                        current = best_entries.get(key)
                        if current is None or total_cost + 1e-12 < current.total_cost_ms:
                            best_entries[key] = SummaryEntry(
                                input_state=input_state,
                                output_state=out_state,
                                total_cost_ms=total_cost,
                                actions=actions,
                            )

    return BlockSummary(
        block=block,
        boundary_states=boundary_states,
        supported=True,
        summary_entries=tuple(best_entries[key] for key in sorted(best_entries, key=lambda item: (item[0].label(), item[1].label()))),
    )


def build_block_summaries(
    graph: OperatorGraph,
    region_result: RegionAnalysisResult,
    cost_model: StateExpandedCostModel,
) -> Dict[str, BlockSummary]:
    node_map = {node.node_id: node for node in graph.nodes}
    summaries: Dict[str, BlockSummary] = {}

    for block in region_result.blocks:
        boundary_states = enumerate_boundary_states(cost_model.max_level())
        if block.kind in {"atomic", "chain"}:
            summaries[block.block_id] = _summarize_chain_block(
                nodes=[node_map[node_id] for node_id in block.nodes],
                block=block,
                cost_model=cost_model,
            )
            continue
        if block.kind == "residual":
            summaries[block.block_id] = _summarize_residual_block(
                graph=graph,
                node_map=node_map,
                block=block,
                cost_model=cost_model,
            )
            continue

        summaries[block.block_id] = BlockSummary(
            block=block,
            boundary_states=boundary_states,
            supported=False,
            summary_entries=tuple(),
            unsupported_reason=f"Local summary builder for block kind '{block.kind}' is not implemented yet",
        )

    return summaries
