"""Section E: SESE plan_builder + runtime adapter.

`GlobalSolveResult` previously had no downstream lowering — the SESE
compiler path dead-ended after the block DP. This test verifies:

- `build_execution_plan_from_sese` flattens `BlockPathDecision.actions`
  into a linear `steps` list matching the shape consumed by the
  existing `compiler_plan_to_runtime_plan` adapter.
- SESE action kinds (`merge_execute`, `convert_he_to_mpc`,
  `convert_mpc_to_he`) are normalized onto the `execute_he /
  execute_mpc / conversion / bootstrap` vocabulary.
- Unsupported results raise.
- `sese_to_runtime_plan` produces a runtime `ExecutionPlan` that the
  executor can traverse without error.
"""

from __future__ import annotations

import numpy as np
import pytest

from compiler.SESE.global_solver import BlockPathDecision, GlobalSolveResult
from compiler.SESE.plan_builder import build_execution_plan_from_sese
from compiler.SESE.region_types import BoundaryState
from compiler.SESE.runtime_plan_adapter import sese_to_runtime_plan
from ir.types import DataEdge, OperatorGraph, OperatorNode
from runtime.plan import ConversionStep, ExecutionPlan, OperatorStep


def _stub_result() -> GlobalSolveResult:
    """A minimal GlobalSolveResult mimicking a one-block SESE solve.

    The block decision's action stream walks HE -> MPC via an edge
    conversion, then executes an MPC op as a merge-execute.
    """
    he = BoundaryState(domain="HE", level_bucket=3)
    mpc = BoundaryState(domain="MPC", level_bucket=None)

    actions = (
        {
            "kind": "convert_he_to_mpc",
            "node_id": "gelu_0",
            "op_type": "GeLU",
            "latency_ms": 0.5,
            "from_state": he.as_dict(),
            "to_state": mpc.as_dict(),
            "reason": "domain_switch",
            "edge": {"src": "source_0", "dst": "gelu_0", "tensor_shape": [1, 8, 768]},
        },
        {
            "kind": "merge_execute",
            "node_id": "gelu_0",
            "op_type": "GeLU",
            "latency_ms": 5.0,
            "from_state": mpc.as_dict(),
            "to_state": mpc.as_dict(),
            "reason": "execute",
        },
    )
    decision = BlockPathDecision(
        block_id="B0",
        block_kind="chain",
        input_state=he,
        output_state=mpc,
        incremental_cost_ms=5.5,
        actions=actions,
    )
    return GlobalSolveResult(
        graph_id="g0",
        supported=True,
        strategy="block_dp_linear",
        total_cost_ms=5.5,
        start_state=he,
        goal_state=mpc,
        block_order=("B0",),
        block_decisions=(decision,),
    )


def _stub_graph() -> OperatorGraph:
    return OperatorGraph(
        graph_id="g0",
        nodes=[
            OperatorNode("source_0", "Source", (1, 8, 768), (1, 8, 768)),
            OperatorNode("gelu_0", "GeLU", (1, 8, 768), (1, 8, 768)),
        ],
        edges=[DataEdge("source_0", "gelu_0", (1, 8, 768))],
    )


# -- plan_builder -------------------------------------------------------------


def test_build_execution_plan_from_sese_flattens_actions():
    result = _stub_result()
    plan = build_execution_plan_from_sese(result)
    assert plan["graph_id"] == "g0"
    assert plan["strategy"] == "sese_global"
    steps = plan["steps"]
    assert [s["kind"] for s in steps] == ["conversion", "execute_mpc"]


def test_build_execution_plan_from_sese_normalizes_conversion_action_shape():
    plan = build_execution_plan_from_sese(_stub_result())
    conv = plan["steps"][0]
    assert conv["kind"] == "conversion"
    assert conv["from_node"] == "source_0"
    assert conv["to_node"] == "gelu_0"
    assert conv["from_domain"] == "HE"
    assert conv["to_domain"] == "MPC"
    assert conv["tensor_shape"] == [1, 8, 768]


def test_build_execution_plan_from_sese_normalizes_merge_execute():
    plan = build_execution_plan_from_sese(_stub_result())
    execute_step = plan["steps"][1]
    # merge_execute on the MPC side -> execute_mpc in the normalized plan
    assert execute_step["kind"] == "execute_mpc"
    assert execute_step["op_type"] == "GeLU"
    assert execute_step["node_id"] == "gelu_0"


def test_build_execution_plan_from_sese_rejects_unsupported_result():
    bad = GlobalSolveResult(
        graph_id="g0",
        supported=False,
        strategy="block_dp_linear",
        total_cost_ms=None,
        start_state=None,
        goal_state=None,
        block_order=("B0",),
        block_decisions=tuple(),
        unsupported_reason="test",
    )
    with pytest.raises(ValueError, match="unsupported"):
        build_execution_plan_from_sese(bad)


def test_build_execution_plan_from_sese_rejects_unknown_kind():
    he = BoundaryState(domain="HE", level_bucket=3)
    result = GlobalSolveResult(
        graph_id="g0",
        supported=True,
        strategy="block_dp_linear",
        total_cost_ms=0.0,
        start_state=he,
        goal_state=he,
        block_order=("B0",),
        block_decisions=(
            BlockPathDecision(
                block_id="B0",
                block_kind="chain",
                input_state=he,
                output_state=he,
                incremental_cost_ms=0.0,
                actions=({"kind": "mystery_kind", "node_id": "x"},),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported SESE action kind"):
        build_execution_plan_from_sese(result)


def test_normalizes_bootstrap_action():
    he = BoundaryState(domain="HE", level_bucket=3)
    result = GlobalSolveResult(
        graph_id="g0",
        supported=True,
        strategy="block_dp_linear",
        total_cost_ms=12.0,
        start_state=he,
        goal_state=he,
        block_order=("B0",),
        block_decisions=(
            BlockPathDecision(
                block_id="B0",
                block_kind="chain",
                input_state=he,
                output_state=he,
                incremental_cost_ms=12.0,
                actions=(
                    {
                        "kind": "bootstrap",
                        "node_id": "gelu_0",
                        "op_type": "GeLU",
                        "latency_ms": 12.0,
                        "reason": "budget_reset",
                    },
                ),
            ),
        ),
    )
    plan = build_execution_plan_from_sese(result)
    assert plan["steps"][0]["kind"] == "bootstrap"
    assert plan["steps"][0]["node_id"] == "gelu_0"


# -- runtime_plan_adapter -----------------------------------------------------


def test_sese_to_runtime_plan_produces_execution_plan():
    graph = _stub_graph()
    result = _stub_result()
    plan = sese_to_runtime_plan(graph, result)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 2
    assert isinstance(plan.steps[0], ConversionStep)
    assert isinstance(plan.steps[1], OperatorStep)
    assert plan.steps[1].op_type == "GeLU"
    # domain propagation
    assert plan.steps[0].from_domain.value == "HE"
    assert plan.steps[0].to_domain.value == "MPC"
