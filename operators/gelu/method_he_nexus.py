from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus import NexusHeGeluBridgeConfig, run_nexus_gelu_bridge
from runtime.types import ExecutionContext


@dataclass
class NexusHeGeluConfig:
    """
    HE GeLU wrapper using NEXUS-based approximation.

    Wrapped NEXUS references:
    - he_compiler/NEXUS/src/gelu.cpp -> GeLUEvaluator::gelu
    - he_compiler/NEXUS/data/data_generation.py -> gelu calibration target

    Shape contract:
    - input: any float tensor, typically [B,S,H]
    - output: same shape as input

    Parameter handling:
    - clamp_min / clamp_max bound the plaintext input before approximation.

    Approximation note:
    - This is an approximate HE-style method based on NEXUS math.
    - Python wrapper execution here is plaintext emulation, not ciphertext execution.
    """

    clamp_min: float = -8.0
    clamp_max: float = 8.0


def run_nexus_gelu_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeGeluConfig | None = None,
) -> np.ndarray:
    del ctx  # Reserved for future runtime controls.
    cfg = cfg or NexusHeGeluConfig()
    bridge_cfg = NexusHeGeluBridgeConfig(clamp_min=cfg.clamp_min, clamp_max=cfg.clamp_max)
    return run_nexus_gelu_bridge(np.asarray(x, dtype=np.float64), bridge_cfg)


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, he_signature


GELU_HE_LEVEL_DELTA = 4  # clamp + degree-12 polynomial eval on [-8,8]
GELU_HE_NOTES = "NEXUS polynomial gelu, clamp [-8,8], bootstrap not supported in-place"


def cost_signature(
    input_shape, output_shape=None, ctx=None
) -> OperatorCostSignature:
    """HE GeLU is shape-agnostic but spends ~4 multiplicative levels."""
    del ctx
    out = output_shape if output_shape is not None else input_shape
    return he_signature(
        "GeLU",
        input_shape=input_shape,
        output_shape=out,
        level_delta=GELU_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=True,
        notes=GELU_HE_NOTES,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    """NEXUS GeLU bakes the level chain into the primitive — no in-place BS."""
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "GeLU.method_he_nexus cannot bootstrap in place; "
        "solver must detour through HE->MPC->HE."
    )
