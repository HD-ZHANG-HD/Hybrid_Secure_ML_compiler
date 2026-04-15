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
class BertBoltAttentionQkMatMulConfig:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    bridge_binary: Path = Path(
        "/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_QK_MATMUL_MPC_BRIDGE"
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


def _to_canonical_qk(arr: np.ndarray, ctx: ExecutionContext | None, tensor_name: str) -> np.ndarray:
    # Canonical attention layout in this integration: [B, H, S, D].
    if arr.ndim == 4:
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{tensor_name} must be [B,S,H*D] or [B,H,S,D], got shape={arr.shape}")
    bsz, seq, hidden = arr.shape
    heads, head_dim = _resolve_num_heads(hidden, ctx)
    return np.asarray(arr, dtype=np.float64).reshape(bsz, seq, heads, head_dim).transpose(0, 2, 1, 3)


def _normalize_qk_inputs(inputs: list[np.ndarray], ctx: ExecutionContext | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize to canonical Q/K tensors in [B,H,S,D].

    Supported forms:
    1) Packed qkv_out (compat path): single tensor `[3,B,S,H*D]` or `[B,S,3*H*D]`.
    2) Pre-split future path: explicit `Q,K` tensors (either `[B,S,H*D]` or `[B,H,S,D]`).
    """
    if len(inputs) >= 2:
        q = _to_canonical_qk(np.asarray(inputs[0]), ctx, "Q")
        k = _to_canonical_qk(np.asarray(inputs[1]), ctx, "K")
        return q, k

    packed = np.asarray(inputs[0], dtype=np.float64)
    if packed.ndim == 4 and packed.shape[0] == 3:
        q, k = packed[0], packed[1]
        return _to_canonical_qk(q, ctx, "Q"), _to_canonical_qk(k, ctx, "K")
    if packed.ndim == 5 and packed.shape[0] == 3:
        q, k = packed[0], packed[1]
        return np.asarray(q, dtype=np.float64), np.asarray(k, dtype=np.float64)
    if packed.ndim == 3:
        hidden3 = packed.shape[-1]
        if hidden3 % 3 != 0:
            raise ValueError(
                f"Packed qkv_out last dim must be divisible by 3 for [B,S,3*hidden], got shape={packed.shape}"
            )
        hidden = hidden3 // 3
        q = packed[..., :hidden]
        k = packed[..., hidden : 2 * hidden]
        return _to_canonical_qk(q, ctx, "Q"), _to_canonical_qk(k, ctx, "K")
    raise ValueError(f"Unsupported input form for Attention_QK_MatMul: shape={packed.shape}")


def run_bert_bolt_attention_qk_matmul_mpc(
    inputs: list[np.ndarray],
    ctx: ExecutionContext | None = None,
    cfg: BertBoltAttentionQkMatMulConfig | None = None,
) -> np.ndarray:
    cfg = cfg or BertBoltAttentionQkMatMulConfig()
    if not cfg.bridge_binary.exists():
        raise RuntimeError(f"Attention_QK_MatMul bridge binary not found: {cfg.bridge_binary}")

    q, k = _normalize_qk_inputs(inputs, ctx)
    if q.shape != k.shape:
        raise ValueError(f"Q and K must have identical shape [B,H,S,D], got Q={q.shape}, K={k.shape}")
    bsz, heads, seq, head_dim = q.shape

    q_batched = q.reshape(bsz * heads, seq, head_dim)
    k_t_batched = np.swapaxes(k, -1, -2).reshape(bsz * heads, head_dim, seq)
    n = bsz * heads
    dim1, dim2, dim3 = seq, head_dim, seq
    out_size = n * dim1 * dim3

    q0, q1 = _share_encode_fixed(q_batched, cfg.ell, cfg.scale, seed=101)
    k0, k1 = _share_encode_fixed(k_t_batched, cfg.ell, cfg.scale, seed=103)

    _log(
        ctx,
        "[attention_qk_wrapper] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp "
        "function=NonLinear::n_matrix_mul_iron(...)",
    )
    last_error = None
    for attempt in range(1, 6):
        port = cfg.port if cfg.port is not None else _choose_port_block(block_size=64)
        _log(
            ctx,
            f"[attention_qk_wrapper] q_shape={list(q.shape)} n={n} dim1={dim1} dim2={dim2} dim3={dim3} "
            f"ell={cfg.ell} s={cfg.scale} nthreads={cfg.nthreads} port={port} attempt={attempt}",
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p1_a = td_path / "party1_a.bin"
            p2_a = td_path / "party2_a.bin"
            p1_b = td_path / "party1_b.bin"
            p2_b = td_path / "party2_b.bin"
            p1_out = td_path / "party1_out.bin"
            p2_out = td_path / "party2_out.bin"
            q0.tofile(p1_a)
            q1.tofile(p2_a)
            k0.tofile(p1_b)
            k1.tofile(p2_b)

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
                    last_error = RuntimeError(f"Attention_QK bridge timeout; retrying with new port block: {exc}")
                    _log(ctx, "[attention_qk_wrapper] timeout_retry=True")
                    continue
                raise RuntimeError(f"Bridge process timeout during MPC Attention_QK_MatMul execution: {exc}") from exc

            ok = p1.returncode == 0 and p2.returncode == 0
            _log(ctx, f"[attention_qk_wrapper] party_launch_success={ok} rc1={p1.returncode} rc2={p2.returncode}")
            _log(ctx, f"[attention_qk_wrapper] party1_stdout={o1.strip()}")
            _log(ctx, f"[attention_qk_wrapper] party2_stdout={o2.strip()}")
            if e1.strip():
                _log(ctx, f"[attention_qk_wrapper] party1_stderr={e1.strip()}")
            if e2.strip():
                _log(ctx, f"[attention_qk_wrapper] party2_stderr={e2.strip()}")
            if not ok:
                stderr_blob = f"{e1}\n{e2}".lower()
                if "address already in use" in stderr_blob and cfg.port is None:
                    last_error = RuntimeError("Attention_QK bridge bind conflict; retrying with a new port block.")
                    _log(ctx, "[attention_qk_wrapper] bind_conflict_retry=True")
                    continue
                raise RuntimeError("Bridge party execution failed; see logged stdout/stderr.")

            y1 = np.fromfile(p1_out, dtype=np.uint64)
            y2 = np.fromfile(p2_out, dtype=np.uint64)
            if y1.size != out_size or y2.size != out_size:
                raise RuntimeError(f"Output share size mismatch: y1={y1.size}, y2={y2.size}, expected={out_size}")

            scores = _decode_recombine(y1, y2, cfg.ell, cfg.scale, (bsz, heads, seq, seq))
            _log(ctx, "[attention_qk_wrapper] recombine_success=True")
            return scores

    raise RuntimeError(f"Attention_QK wrapper failed after retries. Last error: {last_error}")


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, mpc_signature


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out = output_shape if output_shape is not None else in_shape
    return mpc_signature(
        "Attention_QK_MatMul",
        input_shape=in_shape,
        output_shape=out,
        feasible=True,
        notes="BOLT_QK_MATMUL_MPC_BRIDGE: NonLinear::n_matrix_mul_iron (SCI)",
    )

