from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeLinearFfn1Config:
    """
    Restricted NEXUS-backed HE adapter for FFN_Linear_1.

    Wrapped NEXUS internals:
    - he_compiler/NEXUS/src/matrix_mul.cpp:
      - MMEvaluator::matrix_mul
      - row-pack logic from MM_test()

    Restricted contract:
    - input x: [B,S,H], with H=768
    - output y: [B,S,64]
    - requires 1 <= B*S <= 4096
    - optional weight/bias must be [768,64] and [64]

    Status label:
    - restricted-integrated
    """

    hidden_size: int = 768
    out_dim: int = 64
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 1234


def run_nexus_linear_ffn1_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeLinearFfn1Config | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeLinearFfn1Config()
    params = {} if ctx is None else ctx.params

    weight = params.get("ffn_linear1_he_nexus_weight")
    bias = params.get("ffn_linear1_he_nexus_bias")
    y, _, _ = run_nexus_linear_ffn1_restricted_adapter(
        np.asarray(x, dtype=np.float64),
        cfg=NexusLinearFfn1RestrictedAdapterConfig(
            hidden_size=int(params.get("ffn_linear1_he_nexus_hidden_size", cfg.hidden_size)),
            out_dim=int(params.get("ffn_linear1_he_nexus_out_dim", cfg.out_dim)),
            max_tokens=int(params.get("ffn_linear1_he_nexus_max_tokens", cfg.max_tokens)),
            poly_modulus_degree=int(params.get("ffn_linear1_he_nexus_poly_degree", cfg.poly_modulus_degree)),
            weight_seed=int(params.get("ffn_linear1_he_nexus_weight_seed", cfg.weight_seed)),
        ),
        weight=weight,
        bias=bias,
    )
    return y


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, bs_product, he_signature


FFN_LINEAR1_HE_LEVEL_DELTA = 1  # single row-pack matmul
FFN_LINEAR1_HE_MAX_TOKENS = 4096
FFN_LINEAR1_HE_HIDDEN = 768
FFN_LINEAR1_HE_NOTES = (
    "NEXUS row-packed FFN_Linear_1: poly_modulus_degree=4096, max B*S<=4096, H=768."
)


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out_shape = input_shape if output_shape is None else output_shape
    feasible = True
    reason = FFN_LINEAR1_HE_NOTES
    if len(in_shape) != 3 or in_shape[-1] != FFN_LINEAR1_HE_HIDDEN:
        feasible = False
        reason = f"FFN_Linear_1 HE requires [B,S,{FFN_LINEAR1_HE_HIDDEN}]; got {in_shape}"
    elif bs_product(in_shape) > FFN_LINEAR1_HE_MAX_TOKENS:
        feasible = False
        reason = (
            f"FFN_Linear_1 HE requires B*S<={FFN_LINEAR1_HE_MAX_TOKENS}; "
            f"got {bs_product(in_shape)}"
        )
    return he_signature(
        "FFN_Linear_1",
        input_shape=in_shape,
        output_shape=out_shape,
        level_delta=FFN_LINEAR1_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=feasible,
        notes=reason,
        extras={
            "poly_modulus_degree": 4096,
            "max_tokens": FFN_LINEAR1_HE_MAX_TOKENS,
        },
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "FFN_Linear_1.method_he_nexus cannot bootstrap in place."
    )
