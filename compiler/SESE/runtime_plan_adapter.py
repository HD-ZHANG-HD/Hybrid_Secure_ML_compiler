"""Lower a SESE-built compiler plan to a runtime `ExecutionPlan` (Section E).

Delegates to the existing state_expanded runtime adapter which already
knows how to walk the `{kind, node_id, from_node, to_node, ...}` step
stream produced by `plan_builder.build_execution_plan_from_sese`.
"""

from __future__ import annotations

from typing import Dict, List

from compiler.state_expanded_opt.runtime_plan_adapter import (
    compiler_plan_to_runtime_plan,
)
from ir.types import OperatorGraph
from runtime.plan import ExecutionPlan

from .global_solver import GlobalSolveResult
from .plan_builder import build_execution_plan_from_sese


def sese_to_runtime_plan(
    graph: OperatorGraph,
    sese_result: GlobalSolveResult,
    external_inputs: Dict[str, List[str]] | None = None,
) -> ExecutionPlan:
    compiler_plan = build_execution_plan_from_sese(sese_result)
    return compiler_plan_to_runtime_plan(
        graph, compiler_plan, external_inputs=external_inputs
    )


__all__ = ["sese_to_runtime_plan"]
