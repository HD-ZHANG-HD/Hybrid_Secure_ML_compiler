from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeLinearFfn2Config:
    hidden_size: int = 1536
    out_dim: int = 768
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 2234


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_nexus_linear_ffn2_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeLinearFfn2Config | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeLinearFfn2Config()
    params = {} if ctx is None else ctx.params

    weight = params.get("ffn_linear2_he_nexus_weight")
    bias = params.get("ffn_linear2_he_nexus_bias")
    _log(
        ctx,
        "[ffn_linear2_he_nexus] lowered_to=FFN_Linear_1.method_he_nexus "
        "primitive=MMEvaluator::matrix_mul",
    )
    y, _, _ = run_nexus_linear_ffn1_restricted_adapter(
        np.asarray(x, dtype=np.float64),
        cfg=NexusLinearFfn1RestrictedAdapterConfig(
            hidden_size=int(params.get("ffn_linear2_he_nexus_hidden_size", cfg.hidden_size)),
            out_dim=int(params.get("ffn_linear2_he_nexus_out_dim", cfg.out_dim)),
            max_tokens=int(params.get("ffn_linear2_he_nexus_max_tokens", cfg.max_tokens)),
            poly_modulus_degree=int(params.get("ffn_linear2_he_nexus_poly_degree", cfg.poly_modulus_degree)),
            weight_seed=int(params.get("ffn_linear2_he_nexus_weight_seed", cfg.weight_seed)),
        ),
        weight=weight,
        bias=bias,
    )
    return y


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, bs_product, he_signature


FFN_LINEAR2_HE_LEVEL_DELTA = 1
FFN_LINEAR2_HE_MAX_TOKENS = 4096
FFN_LINEAR2_HE_NOTES = "NEXUS FFN_Linear_2 reuses FFN_Linear_1 adapter; same contract."


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out_shape = output_shape if output_shape is not None else in_shape
    feasible = len(in_shape) == 3 and bs_product(in_shape) <= FFN_LINEAR2_HE_MAX_TOKENS
    notes = FFN_LINEAR2_HE_NOTES if feasible else (
        f"FFN_Linear_2 HE requires rank-3 [B,S,H] with B*S<={FFN_LINEAR2_HE_MAX_TOKENS}; "
        f"got shape {in_shape}"
    )
    return he_signature(
        "FFN_Linear_2",
        input_shape=in_shape,
        output_shape=out_shape,
        level_delta=FFN_LINEAR2_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=feasible,
        notes=notes,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "FFN_Linear_2.method_he_nexus cannot bootstrap in place."
    )
