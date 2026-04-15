"""Out_Projection MPC method — reuses FFN_Linear_1 BOLT bridge.

Input shape: [B,S,768]  Output shape: [B,S,768]

Delegates to `run_bert_bolt_ffn_linear1_mpc_chunked` with `out_dim=768` so
that shapes with `n = B*S > 64` are handled transparently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from operators._cost_signature import OperatorCostSignature, bs_product, mpc_signature
from operators.linear_ffn1.method_mpc_bolt import (
    BertBoltFfnLinear1Config,
    CHUNK_MAX_N,
    run_bert_bolt_ffn_linear1_mpc_chunked,
)
from runtime.types import ExecutionContext


@dataclass
class BertBoltOutProjectionConfig:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    weight_seed: int = 3334


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_out_projection_mpc_bolt(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: BertBoltOutProjectionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or BertBoltOutProjectionConfig()
    _log(
        ctx,
        "[out_projection_mpc_bolt] lowered_to=FFN_Linear_1.method_mpc_bolt "
        "primitive=NonLinear::n_matrix_mul_iron (chunked)",
    )
    return run_bert_bolt_ffn_linear1_mpc_chunked(
        np.asarray(x, dtype=np.float64),
        out_dim=768,
        ctx=ctx,
        cfg=BertBoltFfnLinear1Config(
            ell=cfg.ell,
            scale=cfg.scale,
            nthreads=cfg.nthreads,
            address=cfg.address,
            port=cfg.port,
            weight_seed=cfg.weight_seed,
        ),
    )


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out = output_shape if output_shape is not None else in_shape
    n = bs_product(in_shape) if len(in_shape) >= 2 else 1
    return mpc_signature(
        "Out_Projection",
        input_shape=in_shape,
        output_shape=out,
        feasible=True,
        notes=(
            "Out_Projection via BOLT_FFN_LINEAR1_BRIDGE; "
            f"chunked in blocks of <= {CHUNK_MAX_N} tokens (n={n})."
        ),
        extras={
            "chunked": n > CHUNK_MAX_N,
            "num_chunks": (n + CHUNK_MAX_N - 1) // CHUNK_MAX_N,
            "chunk_size": CHUNK_MAX_N,
        },
    )
