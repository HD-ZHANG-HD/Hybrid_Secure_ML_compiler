from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus import NexusHeSoftmaxBridgeConfig, run_nexus_softmax_bridge
from runtime.types import ExecutionContext


@dataclass
class NexusHeSoftmaxConfig:
    """
    HE Softmax wrapper using NEXUS-based approximation.

    Wrapped NEXUS references:
    - he_compiler/NEXUS/src/softmax.cpp -> SoftmaxEvaluator::softmax
    - he_compiler/NEXUS/src/ckks_evaluator.cpp ->
      CKKSEvaluator::exp / CKKSEvaluator::inverse

    Shape contract:
    - input: tensor with ndim>=2, softmax applied over last axis
    - output: same shape as input

    Parameter handling:
    - inverse_iterations controls NEXUS-style inverse approximation depth.
    - sum_scale_factor mirrors NEXUS pre/post scaling around inverse.

    Approximation note:
    - This method is approximate by design.
    - Python wrapper execution here is plaintext emulation, not ciphertext execution.
    """

    inverse_iterations: int = 4
    sum_scale_factor: float = 0.01
    eps: float = 1e-8


def run_nexus_softmax_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeSoftmaxConfig | None = None,
) -> np.ndarray:
    del ctx  # Reserved for future runtime controls.
    cfg = cfg or NexusHeSoftmaxConfig()
    bridge_cfg = NexusHeSoftmaxBridgeConfig(
        inverse_iterations=cfg.inverse_iterations,
        sum_scale_factor=cfg.sum_scale_factor,
        eps=cfg.eps,
    )
    return run_nexus_softmax_bridge(np.asarray(x, dtype=np.float64), bridge_cfg)


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, he_signature


SOFTMAX_HE_LEVEL_DELTA = 8  # exp (1+x/128)^128 + 4 inverse iters
SOFTMAX_HE_NOTES = "NEXUS softmax: (1+x/128)^128 repeated squaring + 4 inverse iterations"


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    out = output_shape if output_shape is not None else input_shape
    return he_signature(
        "Softmax",
        input_shape=input_shape,
        output_shape=out,
        level_delta=SOFTMAX_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=True,
        notes=SOFTMAX_HE_NOTES,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "Softmax.method_he_nexus cannot bootstrap in place; "
        "solver must detour through HE->MPC->HE."
    )
