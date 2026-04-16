"""End-to-end SESE (single-entry single-exit) compiler example.

Runs the SESE strategy on two representative BERT-style graphs:

1. `chain_budget` — a pure FFN chain, where SESE packs everything into a
   single chain block and delegates to the exact state-expanded shortest
   path solver.
2. `residual_stage_local` — a residual block with a shortcut edge, where
   SESE's value is to confine the branch/merge state alignment inside
   one residual block and only expose an O(1) boundary summary to the
   global DP.

For each graph the script:

- loads an `OperatorGraph` and a profiler database (reusing the shared
  `compiler/state_expanded_opt/test/` assets)
- runs `analyze_sese_regions` to get the SESE block decomposition
- builds per-block transfer summaries with `build_block_summaries`
- runs `solve_block_graph_dag` to get the globally optimal block plan
- lowers the result to a runtime `ExecutionPlan` via
  `sese_to_runtime_plan`

Run:
    python -m examples.run_sese_example

No MPC/HE bridges are required — compilation is the output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from compiler.SESE import (
    analyze_sese_regions,
    build_block_summaries,
    solve_block_graph_dag,
)
from compiler.SESE.runtime_plan_adapter import sese_to_runtime_plan
from compiler.min_cut.profiler_db import ProfilerDB
from compiler.state_expanded_opt.cost_model import StateExpandedCostModel
from compiler.state_expanded_opt.graph_model import load_graph_json
from ir.types import OperatorGraph
from runtime.plan import BootstrapStep, ConversionStep, ExecutionPlan, OperatorStep


GRAPH_ROOT = _FRAMEWORK_ROOT / "compiler" / "state_expanded_opt" / "test"
PROFILER_JSON = GRAPH_ROOT / "profiler_with_budget.json"
CASES = {
    "chain_budget": GRAPH_ROOT / "graph_chain_budget.json",
    "residual_stage_local": GRAPH_ROOT / "graph_residual_stage_local.json",
}


def _print_blocks(graph: OperatorGraph, blocks: Iterable) -> None:
    print(f"  graph_id={graph.graph_id}, nodes={len(graph.nodes)}, edges={len(graph.edges)}")
    for block in blocks:
        node_list = ", ".join(block.nodes)
        print(
            f"  block {block.block_id}"
            f"  kind={block.kind}"
            f"  entry={block.entry}"
            f"  exit={block.exit}"
            f"  nodes=[{node_list}]"
        )


def _print_block_decisions(result) -> None:
    print(
        f"  global strategy={result.strategy}"
        f"  total_cost_ms={result.total_cost_ms:.3f}"
        f"  start={result.start_state.label() if result.start_state else None}"
        f"  goal={result.goal_state.label() if result.goal_state else None}"
    )
    for decision in result.block_decisions:
        print(
            f"  -> block {decision.block_id}"
            f"  ({decision.block_kind})"
            f"  {decision.input_state.label()} -> {decision.output_state.label()}"
            f"  +{decision.incremental_cost_ms:.3f} ms"
            f"  actions={len(decision.actions)}"
        )


def _print_runtime_plan(plan: ExecutionPlan) -> None:
    print(f"  runtime plan: {len(plan.steps)} steps")
    for idx, step in enumerate(plan.steps):
        if isinstance(step, OperatorStep):
            inputs = ",".join(step.inputs)
            outputs = ",".join(step.outputs)
            print(
                f"   [{idx:02d}] OPERATOR  {step.op_type}@{step.backend.value}/{step.method}"
                f"  in=[{inputs}]  out=[{outputs}]"
            )
        elif isinstance(step, ConversionStep):
            print(
                f"   [{idx:02d}] CONVERT   {step.from_domain.value}->{step.to_domain.value}"
                f"  tensor={step.tensor}  method={step.method}"
                f"  out={step.output_tensor}"
            )
        elif isinstance(step, BootstrapStep):
            print(
                f"   [{idx:02d}] BOOTSTRAP {step.backend.value}"
                f"  tensor={step.tensor}  method={step.method}"
                f"  out={step.output_tensor}"
            )


def run_case(name: str, graph_path: Path, cost_model: StateExpandedCostModel) -> None:
    print(f"\n===== case: {name} =====")
    graph = load_graph_json(graph_path)

    region = analyze_sese_regions(graph)
    print("[1] SESE region decomposition")
    _print_blocks(graph, region.blocks)
    print(f"  block topo order = {list(region.block_topological_order)}")

    summaries = build_block_summaries(graph, region, cost_model)
    print("\n[2] block summaries")
    for block_id in region.block_topological_order:
        s = summaries[block_id]
        status = "ok" if s.supported else f"unsupported ({s.unsupported_reason})"
        print(f"  {block_id}: {status}, {len(s.summary_entries)} summary entries")

    print("\n[3] global block-DP over SESE summaries")
    solved = solve_block_graph_dag(region, summaries, graph, cost_model)
    if not solved.supported:
        print(f"  !! global solve failed: {solved.unsupported_reason}")
        return
    _print_block_decisions(solved)

    print("\n[4] lowered runtime ExecutionPlan")
    runtime_plan = sese_to_runtime_plan(graph, solved)
    _print_runtime_plan(runtime_plan)


def main() -> None:
    profiler = ProfilerDB.from_json(PROFILER_JSON)
    cost_model = StateExpandedCostModel(profiler)
    for name, path in CASES.items():
        run_case(name, path, cost_model)


if __name__ == "__main__":
    main()
