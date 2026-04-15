from __future__ import annotations

from dataclasses import dataclass
import random
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from runtime.types import ExecutionContext


@dataclass
class BertBoltAttentionVMatMulConfig:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    bridge_binary: Path = Path(
        "/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_ATTN_V_MATMUL_MPC_BRIDGE"
    )


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)
    print(message)


def _port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _choose_port_block(block_size: int = 64, trials: int = 200) -> int:
    for _ in range(trials):
        start = random.randint(20000, 50000 - block_size - 1)
        if all(_port_available(start + i) for i in range(block_size)):
            return start
    raise RuntimeError("Failed to find free contiguous port block for SCI runtime.")


def _share_encode_fixed(x: np.ndarray, ell: int, scale: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << ell) - 1
    q = np.round(x.reshape(-1) * (1 << scale)).astype(np.int64)
    q_u = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(seed)
    sh0 = rng.integers(0, 1 << ell, size=q_u.size, dtype=np.uint64)
    sh1 = (q_u - sh0) & np.uint64(mask)
    return sh0, sh1


def _decode_recombine(sh0: np.ndarray, sh1: np.ndarray, ell: int, scale: int, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.uint64((1 << ell) - 1)
    c = (sh0 + sh1) & mask
    signed = c.astype(np.int64)
    signed = np.where(signed >= (1 << (ell - 1)), signed - (1 << ell), signed)
    return (signed.astype(np.float64) / float(1 << scale)).reshape(shape)


def _resolve_num_heads(hidden_size: int, ctx: ExecutionContext | None) -> tuple[int, int]:
    if ctx is None:
        num_heads = 1
    else:
        num_heads = int(ctx.params.get("attention_num_heads", ctx.params.get("num_heads", 1)))
    if num_heads <= 0:
        raise ValueError(f"attention_num_heads must be positive, got {num_heads}")
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden size must be divisible by attention_num_heads; hidden={hidden_size}, heads={num_heads}"
        )
    head_dim = hidden_size // num_heads
    return num_heads, head_dim


def _to_canonical_v(arr: np.ndarray, heads_hint: int, ctx: ExecutionContext | None) -> np.ndarray:
    # Canonical value layout in this integration: [B, H, S, D].
    if arr.ndim == 4:
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"V must be [B,S,H*D] or [B,H,S,D], got shape={arr.shape}")
    bsz, seq, hidden = arr.shape
    heads, head_dim = _resolve_num_heads(hidden, ctx)
    if heads_hint > 0 and heads_hint != heads:
        raise ValueError(f"attn heads ({heads_hint}) and V heads ({heads}) mismatch")
    return np.asarray(arr, dtype=np.float64).reshape(bsz, seq, heads, head_dim).transpose(0, 2, 1, 3)


def _to_canonical_attn(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        # Legacy compatibility when the graph behaves as single-head attention.
        return arr[:, np.newaxis, :, :]
    raise ValueError(f"attn_probs must be [B,S,S] or [B,H,S,S], got shape={arr.shape}")


def _extract_v_from_packed(packed: np.ndarray, heads_hint: int, ctx: ExecutionContext | None) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float64)
    if packed.ndim == 4 and packed.shape[0] == 3:
        return _to_canonical_v(packed[2], heads_hint, ctx)
    if packed.ndim == 5 and packed.shape[0] == 3:
        return np.asarray(packed[2], dtype=np.float64)
    if packed.ndim == 3 and packed.shape[-1] % 3 == 0:
        hidden = packed.shape[-1] // 3
        return _to_canonical_v(packed[..., 2 * hidden :], heads_hint, ctx)
    raise ValueError(f"Unsupported packed qkv_out shape for V extraction: {packed.shape}")


def _normalize_attn_v_inputs(inputs: list[np.ndarray], ctx: ExecutionContext | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize to canonical tensors:
    - Attn in [B,H,S,S]
    - V in [B,H,S,D]

    Supported forms:
    1) Current route: `[attn_probs, packed_qkv_out]`.
    2) Future pre-split: `[attn_probs, V]` (or `[attn_probs, Q, K, V]`, where V is the last tensor).
    """
    if len(inputs) < 2:
        raise ValueError("Attention_V_MatMul requires at least two inputs.")
    attn = _to_canonical_attn(inputs[0])
    heads_hint = int(attn.shape[1])

    if len(inputs) >= 4:
        v = _to_canonical_v(np.asarray(inputs[3]), heads_hint, ctx)
    elif len(inputs) >= 3:
        v = _to_canonical_v(np.asarray(inputs[2]), heads_hint, ctx)
    else:
        second = np.asarray(inputs[1])
        if (second.ndim == 4 and second.shape[0] == 3) or (second.ndim == 5 and second.shape[0] == 3):
            v = _extract_v_from_packed(second, heads_hint, ctx)
        elif second.ndim == 3 and second.shape[-1] % 3 == 0:
            v = _extract_v_from_packed(second, heads_hint, ctx)
        else:
            v = _to_canonical_v(second, heads_hint, ctx)

    if attn.shape[0] != v.shape[0] or attn.shape[1] != v.shape[1] or attn.shape[2] != v.shape[2]:
        raise ValueError(f"attn and V shape mismatch after normalization: attn={attn.shape}, v={v.shape}")
    if attn.shape[2] != attn.shape[3]:
        raise ValueError(f"attn_probs must be square in the last two dims, got shape={attn.shape}")
    return attn, v


def _format_context_output(context_bhsd: np.ndarray, ctx: ExecutionContext | None) -> np.ndarray:
    return_canonical = False if ctx is None else bool(ctx.params.get("attention_return_canonical", False))
    if return_canonical:
        return context_bhsd
    bsz, heads, seq, head_dim = context_bhsd.shape
    return context_bhsd.transpose(0, 2, 1, 3).reshape(bsz, seq, heads * head_dim)


def run_bert_bolt_attention_v_matmul_mpc(
    inputs: list[np.ndarray],
    ctx: ExecutionContext | None = None,
    cfg: BertBoltAttentionVMatMulConfig | None = None,
) -> np.ndarray:
    cfg = cfg or BertBoltAttentionVMatMulConfig()
    if not cfg.bridge_binary.exists():
        raise RuntimeError(f"Attention_V_MatMul bridge binary not found: {cfg.bridge_binary}")

    attn, v = _normalize_attn_v_inputs(inputs, ctx)
    bsz, heads, seq, _ = attn.shape
    _, _, _, head_dim = v.shape

    attn_batched = attn.reshape(bsz * heads, seq, seq)
    v_batched = v.reshape(bsz * heads, seq, head_dim)
    n = bsz * heads
    dim1, dim2, dim3 = seq, seq, head_dim
    out_size = n * dim1 * dim3

    attn0, attn1 = _share_encode_fixed(attn_batched, cfg.ell, cfg.scale, seed=107)
    v0, v1 = _share_encode_fixed(v_batched, cfg.ell, cfg.scale, seed=109)

    _log(
        ctx,
        "[attention_v_wrapper] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp "
        "function=NonLinear::n_matrix_mul_iron(...)",
    )
    last_error = None
    for attempt in range(1, 6):
        port = cfg.port if cfg.port is not None else _choose_port_block(block_size=64)
        _log(
            ctx,
            f"[attention_v_wrapper] attn_shape={list(attn.shape)} v_shape={list(v.shape)} n={n} "
            f"dim1={dim1} dim2={dim2} dim3={dim3} ell={cfg.ell} s={cfg.scale} "
            f"nthreads={cfg.nthreads} port={port} attempt={attempt}",
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p1_a = td_path / "party1_a.bin"
            p2_a = td_path / "party2_a.bin"
            p1_b = td_path / "party1_b.bin"
            p2_b = td_path / "party2_b.bin"
            p1_out = td_path / "party1_out.bin"
            p2_out = td_path / "party2_out.bin"
            attn0.tofile(p1_a)
            attn1.tofile(p2_a)
            v0.tofile(p1_b)
            v1.tofile(p2_b)

            cmd1 = [
                str(cfg.bridge_binary),
                "--party",
                "1",
                "--port",
                str(port),
                "--address",
                cfg.address,
                "--nthreads",
                str(cfg.nthreads),
                "--ell",
                str(cfg.ell),
                "--scale",
                str(cfg.scale),
                "--n",
                str(n),
                "--dim1",
                str(dim1),
                "--dim2",
                str(dim2),
                "--dim3",
                str(dim3),
                "--input_a",
                str(p1_a),
                "--input_b",
                str(p1_b),
                "--output",
                str(p1_out),
            ]
            cmd2 = [
                str(cfg.bridge_binary),
                "--party",
                "2",
                "--port",
                str(port),
                "--address",
                cfg.address,
                "--nthreads",
                str(cfg.nthreads),
                "--ell",
                str(cfg.ell),
                "--scale",
                str(cfg.scale),
                "--n",
                str(n),
                "--dim1",
                str(dim1),
                "--dim2",
                str(dim2),
                "--dim3",
                str(dim3),
                "--input_a",
                str(p2_a),
                "--input_b",
                str(p2_b),
                "--output",
                str(p2_out),
            ]

            p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.3)
            p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                o1, e1 = p1.communicate(timeout=180)
                o2, e2 = p2.communicate(timeout=180)
            except subprocess.TimeoutExpired as exc:
                p1.kill()
                p2.kill()
                if cfg.port is None:
                    last_error = RuntimeError(f"Attention_V bridge timeout; retrying with new port block: {exc}")
                    _log(ctx, "[attention_v_wrapper] timeout_retry=True")
                    continue
                raise RuntimeError(f"Bridge process timeout during MPC Attention_V_MatMul execution: {exc}") from exc

            ok = p1.returncode == 0 and p2.returncode == 0
            _log(ctx, f"[attention_v_wrapper] party_launch_success={ok} rc1={p1.returncode} rc2={p2.returncode}")
            _log(ctx, f"[attention_v_wrapper] party1_stdout={o1.strip()}")
            _log(ctx, f"[attention_v_wrapper] party2_stdout={o2.strip()}")
            if e1.strip():
                _log(ctx, f"[attention_v_wrapper] party1_stderr={e1.strip()}")
            if e2.strip():
                _log(ctx, f"[attention_v_wrapper] party2_stderr={e2.strip()}")
            if not ok:
                stderr_blob = f"{e1}\n{e2}".lower()
                if "address already in use" in stderr_blob and cfg.port is None:
                    last_error = RuntimeError("Attention_V bridge bind conflict; retrying with a new port block.")
                    _log(ctx, "[attention_v_wrapper] bind_conflict_retry=True")
                    continue
                raise RuntimeError("Bridge party execution failed; see logged stdout/stderr.")

            y1 = np.fromfile(p1_out, dtype=np.uint64)
            y2 = np.fromfile(p2_out, dtype=np.uint64)
            if y1.size != out_size or y2.size != out_size:
                raise RuntimeError(f"Output share size mismatch: y1={y1.size}, y2={y2.size}, expected={out_size}")

            context_bhsd = _decode_recombine(y1, y2, cfg.ell, cfg.scale, (bsz, heads, seq, head_dim))
            _log(ctx, "[attention_v_wrapper] recombine_success=True")
            return _format_context_output(context_bhsd, ctx)

    raise RuntimeError(f"Attention_V wrapper failed after retries. Last error: {last_error}")


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, mpc_signature


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out = output_shape if output_shape is not None else in_shape
    return mpc_signature(
        "Attention_V_MatMul",
        input_shape=in_shape,
        output_shape=out,
        feasible=True,
        notes="BOLT_ATTN_V_MATMUL_MPC_BRIDGE: NonLinear::n_matrix_mul_iron (SCI)",
    )

