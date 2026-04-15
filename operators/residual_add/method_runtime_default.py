from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.types import BackendType, ExecutionContext


@dataclass
class ResidualAddConfig:
    require_same_shape: bool = True


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_residual_add_semantic(
    inputs: list[np.ndarray],
    backend: BackendType,
    ctx: ExecutionContext | None = None,
    cfg: ResidualAddConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ResidualAddConfig()
    if len(inputs) != 2:
        raise ValueError(f"Residual_Add expects exactly two inputs, got {len(inputs)}")
    a = np.asarray(inputs[0], dtype=np.float64)
    b = np.asarray(inputs[1], dtype=np.float64)
    if cfg.require_same_shape and a.shape != b.shape:
        raise ValueError(f"Residual_Add requires identical input shapes, got {a.shape} and {b.shape}")
    _log(ctx, f"[residual_add_semantic] backend={backend.value} lowered_to=backend_tensor_add")
    return a + b


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, he_signature, mpc_signature


def cost_signature(
    input_shape,
    output_shape=None,
    ctx=None,
    domain: str = "HE",
) -> OperatorCostSignature:
    """Residual_Add: zero mult-depth in HE; trivial in MPC.

    The runtime-default method is backend-agnostic; callers pass ``domain``
    to select which signature is returned.
    """
    del ctx
    out = output_shape if output_shape is not None else input_shape
    if domain == "HE":
        return he_signature(
            "Residual_Add",
            input_shape=input_shape,
            output_shape=out,
            level_delta=0,
            bootstrap_supported=False,
            feasible=True,
            notes="backend_tensor_add; HE depth=0, level alignment required on deeper branch",
        )
    return mpc_signature(
        "Residual_Add",
        input_shape=input_shape,
        output_shape=out,
        feasible=True,
        notes="backend_tensor_add (MPC share add)",
    )
