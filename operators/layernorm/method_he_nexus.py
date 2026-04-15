from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_layernorm_adapter import (
    NexusLayerNormRestrictedAdapterConfig,
    run_nexus_layernorm_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeLayerNormConfig:
    """
    Restricted NEXUS-backed HE adapter for LayerNorm.

    Wrapped NEXUS internals:
    - he_compiler/NEXUS/src/layer_norm.cpp -> LNEvaluator::layer_norm

    Restricted contract:
    - input x shape [B,S,768]
    - flattened token count 1 <= B*S <= 16
    - affine weight/bias are not supported
    - normalization follows current NEXUS execution style and is not a fully
      general HE LayerNorm

    Status label:
    - restricted-integrated
    """

    hidden_size: int = 768
    max_tokens: int = 16
    packed_len: int = 1024
    eps: float = 1e-8


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_nexus_layernorm_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeLayerNormConfig | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeLayerNormConfig()
    if ctx is not None:
        if ctx.params.get("layernorm_weight") is not None:
            raise ValueError("LayerNorm.method_he_nexus does not support affine weight under the restricted contract")
        if ctx.params.get("layernorm_bias") is not None:
            raise ValueError("LayerNorm.method_he_nexus does not support affine bias under the restricted contract")

    x = np.asarray(x, dtype=np.float64)
    _log(
        ctx,
        "[layernorm_he_nexus] source=he_compiler/NEXUS/src/layer_norm.cpp "
        "function=LNEvaluator::layer_norm(Ciphertext&, Ciphertext&, int)",
    )
    _log(
        ctx,
        f"[layernorm_he_nexus] restricted_contract input_shape={list(x.shape)} "
        f"hidden_size={cfg.hidden_size} max_tokens={cfg.max_tokens} packed_len={cfg.packed_len}",
    )
    return run_nexus_layernorm_restricted_adapter(
        x,
        cfg=NexusLayerNormRestrictedAdapterConfig(
            hidden_size=cfg.hidden_size,
            max_tokens=cfg.max_tokens,
            packed_len=cfg.packed_len,
            eps=cfg.eps,
        ),
    )


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, bs_product, he_signature


LAYERNORM_HE_LEVEL_DELTA = 3  # mean-square + inv-sqrt
LAYERNORM_HE_MAX_BS = 16
LAYERNORM_HE_HIDDEN = 768
LAYERNORM_HE_NOTES = (
    "NEXUS LayerNorm restricted contract: B*S<=16, hidden=768, no affine params."
)


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    out = output_shape if output_shape is not None else input_shape
    in_shape = tuple(int(d) for d in input_shape)
    feasible = True
    reason = LAYERNORM_HE_NOTES
    if len(in_shape) != 3 or in_shape[-1] != LAYERNORM_HE_HIDDEN:
        feasible = False
        reason = f"LayerNorm HE requires [B,S,{LAYERNORM_HE_HIDDEN}]; got {in_shape}"
    elif bs_product(in_shape) > LAYERNORM_HE_MAX_BS:
        feasible = False
        reason = f"LayerNorm HE requires B*S<={LAYERNORM_HE_MAX_BS}; got {bs_product(in_shape)}"
    # affine params are forbidden by the restricted contract
    if feasible and ctx is not None:
        if ctx.params.get("layernorm_weight") is not None or ctx.params.get("layernorm_bias") is not None:
            feasible = False
            reason = "LayerNorm HE restricted contract forbids affine weight/bias"
    return he_signature(
        "LayerNorm",
        input_shape=in_shape,
        output_shape=out,
        level_delta=LAYERNORM_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=feasible,
        notes=reason,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "LayerNorm.method_he_nexus cannot bootstrap in place; "
        "solver must detour through HE->MPC->HE."
    )
