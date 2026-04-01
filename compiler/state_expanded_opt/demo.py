from __future__ import annotations

import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.domain_assignment import AssignmentResult, DataEdge, OperatorGraph, OperatorNode, assign_domains_min_cut
from compiler.min_cut.plan_builder import build_execution_plan as build_min_cut_plan
from compiler.min_cut.profiler_db import ProfilerDB
from .graph_model import load_graph_json
from .runtime_plan_adapter import compile_graph_state_expanded


def _to_min_cut_graph(graph: OperatorGraph) -> OperatorGraph:
    return OperatorGraph(
        graph_id=graph.graph_id,
        nodes=[
            OperatorNode(
                node_id=node.node_id,
                op_type=node.op_type,
                input_shape=node.input_shape,
                output_shape=node.output_shape,
            )
            for node in graph.nodes
        ],
        edges=[DataEdge(src=edge.src, dst=edge.dst, tensor_shape=edge.tensor_shape) for edge in graph.edges],
    )


def _uniform_total(graph: OperatorGraph, profiler_db: ProfilerDB, domain: str) -> float:
    from compiler.min_cut.domain_assignment import evaluate_assignment_cost, make_uniform_assignment

    cm = CostModel(profiler_db, default_strategy="auto")
    assignment = make_uniform_assignment(_to_min_cut_graph(graph), domain)  # type: ignore[arg-type]
    _, _, total = evaluate_assignment_cost(_to_min_cut_graph(graph), assignment, cm)
    return total


def _print_case(name: str, graph: OperatorGraph, profiler_db: ProfilerDB) -> None:
    print("\n" + "=" * 80)
    print(f"[case] {name}")
    solver_result, compiler_plan, _ = compile_graph_state_expanded(graph, profiler_db)
    print(f"[state_expanded] strategy={solver_result.strategy} total={solver_result.total_cost_ms:.3f} ms")
    print("[state_trace]")
    for state in solver_result.state_trace:
        print(f"  - {state}")
    print("[plan_steps]")
    for step in compiler_plan["steps"]:  # type: ignore[index]
        print(f"  - {step['step_id']} {step['kind']} node={step['node_id']} cost={step['estimated_latency_ms']:.3f} ms reason={step['reason']}")
    print(
        "[cost_breakdown] operator={op:.3f} ms conv={conv:.3f} ms bootstrap={bs:.3f} ms total={tot:.3f} ms".format(
            op=compiler_plan["cost_breakdown"]["operator_cost_ms"],  # type: ignore[index]
            conv=compiler_plan["cost_breakdown"]["conversion_cost_ms"],  # type: ignore[index]
            bs=compiler_plan["cost_breakdown"]["bootstrap_cost_ms"],  # type: ignore[index]
            tot=compiler_plan["cost_breakdown"]["total_cost_ms"],  # type: ignore[index]
        )
    )

    min_cut_graph = _to_min_cut_graph(graph)
    min_cut_result = assign_domains_min_cut(min_cut_graph, CostModel(profiler_db, default_strategy="auto"))
    min_cut_plan = build_min_cut_plan(min_cut_graph, min_cut_result.assignment, CostModel(profiler_db, default_strategy="auto"))
    print(f"[min_cut] total={min_cut_plan['cost_breakdown']['total_cost_ms']:.3f} ms")
    print(f"[baseline] all_HE={_uniform_total(graph, profiler_db, 'HE'):.3f} ms")
    print(f"[baseline] all_MPC={_uniform_total(graph, profiler_db, 'MPC'):.3f} ms")


def run_demo() -> None:
    here = Path(__file__).resolve().parent
    profiler_db = ProfilerDB.from_json(here / "test" / "profiler_with_budget.json")
    cases = {
        "chain_budget": load_graph_json(here / "test" / "graph_chain_budget.json"),
        "chain_reset_preferred": load_graph_json(here / "test" / "graph_chain_reset_preferred.json"),
        "residual_stage_local": load_graph_json(here / "test" / "graph_residual_stage_local.json"),
    }
    for name, graph in cases.items():
        _print_case(name, graph, profiler_db)


if __name__ == "__main__":
    run_demo()
