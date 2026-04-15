"""Out_Projection HE method — reuses FFN_Linear_1 NEXUS adapter.

Input shape: [B,S,768]  Output shape: [B,S,768]

Preserves the operator name "Out_Projection" in traces and the cost model
while delegating actual execution to the FFN_Linear_1 restricted adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)
from operators._cost_signature import (
    BootstrapUnsupportedError,
    OperatorCostSignature,
    bs_product,
    he_signature,
)
from runtime.types import ExecutionContext


OUT_PROJECTION_HE_HIDDEN = 768
OUT_PROJECTION_HE_MAX_TOKENS = 4096
OUT_PROJECTION_HE_LEVEL_DELTA = 1
OUT_PROJECTION_HE_NOTES = (
    "Out_Projection lowered onto NEXUS FFN_Linear_1: [B,S,768]->[B,S,768]"
)


@dataclass
class NexusHeOutProjectionConfig:
    hidden_size: int = 768
    out_dim: int = 768
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 3334


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_out_projection_he_nexus(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeOutProjectionConfig | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeOutProjectionConfig()
    params = {} if ctx is None else ctx.params
    weight = params.get("out_projection_he_nexus_weight")
    bias = params.get("out_projection_he_nexus_bias")
    _log(
        ctx,
        "[out_projection_he_nexus] lowered_to=FFN_Linear_1.method_he_nexus "
        "primitive=MMEvaluator::matrix_mul",
    )
    y, _, _ = run_nexus_linear_ffn1_restricted_adapter(
        np.asarray(x, dtype=np.float64),
        cfg=NexusLinearFfn1RestrictedAdapterConfig(
            hidden_size=int(params.get("out_projection_he_nexus_hidden_size", cfg.hidden_size)),
            out_dim=int(params.get("out_projection_he_nexus_out_dim", cfg.out_dim)),
            max_tokens=int(params.get("out_projection_he_nexus_max_tokens", cfg.max_tokens)),
            poly_modulus_degree=int(
                params.get("out_projection_he_nexus_poly_degree", cfg.poly_modulus_degree)
            ),
            weight_seed=int(params.get("out_projection_he_nexus_weight_seed", cfg.weight_seed)),
        ),
        weight=weight,
        bias=bias,
    )
    return y


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out_shape = output_shape if output_shape is not None else in_shape
    feasible = True
    reason = OUT_PROJECTION_HE_NOTES
    if len(in_shape) != 3 or in_shape[-1] != OUT_PROJECTION_HE_HIDDEN:
        feasible = False
        reason = (
            f"Out_Projection HE requires [B,S,{OUT_PROJECTION_HE_HIDDEN}]; got {in_shape}"
        )
    elif bs_product(in_shape) > OUT_PROJECTION_HE_MAX_TOKENS:
        feasible = False
        reason = (
            f"Out_Projection HE requires B*S<={OUT_PROJECTION_HE_MAX_TOKENS}; "
            f"got {bs_product(in_shape)}"
        )
    return he_signature(
        "Out_Projection",
        input_shape=in_shape,
        output_shape=out_shape,
        level_delta=OUT_PROJECTION_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=feasible,
        notes=reason,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    raise BootstrapUnsupportedError(
        "Out_Projection.method_he_nexus_as_ffn1 cannot bootstrap in place."
    )
