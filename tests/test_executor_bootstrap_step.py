"""Section F: runtime executor handles BootstrapStep.

Previously `BootstrapStep` was defined in `runtime/plan.py` but the
executor only knew about `OperatorStep` and `ConversionStep`, so any
compiler plan that emitted a bootstrap would crash with
`TypeError: Unsupported execution step`.

The new executor treats a BootstrapStep as a pass-through in the HE
domain — semantically the level budget is "refreshed" but no tensor
transformation is applied (NEXUS HE methods do not expose an in-place
bootstrap primitive). The test below:

- Builds a 2-step plan: bootstrap then identity operator
- Runs it through the executor
- Asserts no TypeError and that the bootstrap output tensor is set
- Asserts the trace includes a BOOTSTRAP entry
- Asserts a domain mismatch between source and bootstrap.backend raises
"""

from __future__ import annotations

import numpy as np
import pytest

from runtime.executor import execute
from runtime.operator_registry import OperatorRegistry
from runtime.plan import BootstrapStep, ExecutionPlan, OperatorStep
from runtime.types import BackendType, ExecutionContext, TensorValue


def _identity_op(inputs, ctx):
    assert len(inputs) == 1
    return TensorValue(np.asarray(inputs[0].data), inputs[0].domain, meta={})


def _registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(
        "IdentityHE",
        BackendType.HE,
        _identity_op,
        method_name="method_he_test",
    )
    return reg


def test_executor_runs_bootstrap_then_operator():
    plan = ExecutionPlan(
        steps=[
            BootstrapStep(
                backend=BackendType.HE,
                tensor="x_in",
                method="method_default",
                output_tensor="x_refreshed",
            ),
            OperatorStep(
                op_type="IdentityHE",
                method="method_he_test",
                backend=BackendType.HE,
                inputs=["x_refreshed"],
                outputs=["x_out"],
            ),
        ]
    )
    tensors = {
        "x_in": TensorValue(np.array([1.0, 2.0, 3.0]), BackendType.HE, meta={}),
    }
    ctx = ExecutionContext()
    result = execute(plan, tensors, ctx=ctx, registry=_registry())

    # Bootstrap output exists and mirrors the input tensor data.
    assert "x_refreshed" in result
    np.testing.assert_array_equal(result["x_refreshed"].data, np.array([1.0, 2.0, 3.0]))
    # Downstream operator ran through.
    assert "x_out" in result
    # Trace includes a BOOTSTRAP line.
    assert any("BOOTSTRAP" in line for line in ctx.trace)


def test_executor_rejects_bootstrap_domain_mismatch():
    plan = ExecutionPlan(
        steps=[
            BootstrapStep(
                backend=BackendType.HE,
                tensor="x_in",
                method="method_default",
                output_tensor="x_out",
            ),
        ]
    )
    tensors = {
        # tensor is in MPC but bootstrap targets HE
        "x_in": TensorValue(np.array([1.0]), BackendType.MPC, meta={}),
    }
    with pytest.raises(ValueError, match="bootstrap mismatch"):
        execute(plan, tensors, ctx=ExecutionContext(), registry=_registry())


def test_executor_still_rejects_unknown_step_kinds():
    class WeirdStep:
        pass

    plan = ExecutionPlan(steps=[WeirdStep()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Unsupported execution step"):
        execute(plan, {}, ctx=ExecutionContext(), registry=_registry())
