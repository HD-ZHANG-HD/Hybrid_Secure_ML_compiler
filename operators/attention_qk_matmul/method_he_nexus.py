from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_attention_adapter import (
    NexusAttentionRestrictedConfig,
    run_nexus_attention_qk_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeAttentionQkMatMulConfig:
    """
    Restricted NEXUS-backed HE adapter for Attention_QK_MatMul.

    Wrapped NEXUS internals:
    - he_compiler/NEXUS/src/matrix_mul.cpp (MMEvaluator matrix-mul packing model)

    Restricted contract:
    - inputs: packed qkv only, shape [3,B,S,768]
    - heads fixed at 12 (head_dim=64)
    - 1<=B<=8, 1<=S<=128
    - output: [B,12,S,S]

    Status label:
    - restricted-integrated
    """

    hidden_size: int = 768
    num_heads: int = 12
    max_seq_len: int = 128
    max_batch: int = 8


def run_nexus_attention_qk_matmul_he(
    inputs: list[np.ndarray],
    ctx: ExecutionContext | None = None,
    cfg: NexusHeAttentionQkMatMulConfig | None = None,
) -> np.ndarray:
    del ctx
    cfg = cfg or NexusHeAttentionQkMatMulConfig()
    if len(inputs) != 1:
        raise ValueError(
            "Restricted Attention_QK_MatMul HE adapter requires one packed qkv input: [3,B,S,768]"
        )
    qkv_packed = np.asarray(inputs[0], dtype=np.float64)
    return run_nexus_attention_qk_restricted_adapter(
        qkv_packed,
        cfg=NexusAttentionRestrictedConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            max_seq_len=cfg.max_seq_len,
            max_batch=cfg.max_batch,
        ),
    )


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, he_signature


ATTENTION_QK_HE_LEVEL_DELTA = 1
ATTENTION_QK_HE_HIDDEN = 768
ATTENTION_QK_HE_MAX_BATCH = 8
ATTENTION_QK_HE_MAX_SEQ = 128
ATTENTION_QK_HE_NOTES = (
    "NEXUS attention QK: packed [3,B,S,768], 1<=B<=8, 1<=S<=128, 12 heads."
)


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    feasible = True
    reason = ATTENTION_QK_HE_NOTES
    if len(in_shape) != 4 or in_shape[0] != 3 or in_shape[-1] != ATTENTION_QK_HE_HIDDEN:
        feasible = False
        reason = (
            f"Attention_QK_MatMul HE requires packed [3,B,S,{ATTENTION_QK_HE_HIDDEN}]; "
            f"got {in_shape}"
        )
    else:
        b, s = in_shape[1], in_shape[2]
        if not (1 <= b <= ATTENTION_QK_HE_MAX_BATCH and 1 <= s <= ATTENTION_QK_HE_MAX_SEQ):
            feasible = False
            reason = (
                f"Attention_QK_MatMul HE requires 1<=B<={ATTENTION_QK_HE_MAX_BATCH}, "
                f"1<=S<={ATTENTION_QK_HE_MAX_SEQ}; got B={b}, S={s}"
            )
    out_shape = output_shape if output_shape is not None else in_shape
    return he_signature(
        "Attention_QK_MatMul",
        input_shape=in_shape,
        output_shape=out_shape,
        level_delta=ATTENTION_QK_HE_LEVEL_DELTA,
        bootstrap_supported=False,
        feasible=feasible,
        notes=reason,
    )


def bootstrap(tensor: np.ndarray, ctx: ExecutionContext | None = None) -> np.ndarray:
    from operators._cost_signature import BootstrapUnsupportedError
    raise BootstrapUnsupportedError(
        "Attention_QK_MatMul.method_he_nexus cannot bootstrap in place."
    )
