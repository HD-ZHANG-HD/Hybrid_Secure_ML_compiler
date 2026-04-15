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
class BertBoltFfnLinear1Config:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    weight_seed: int = 1234
    bridge_binary: Path = Path(
        "/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_FFN_LINEAR1_BRIDGE"
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


def deterministic_ffn_linear1_params(h: int, out_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((h, out_dim))
    b = rng.standard_normal((out_dim,))
    return w.astype(np.float64), b.astype(np.float64)


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


def run_bert_bolt_ffn_linear1_mpc(
    x: np.ndarray,
    out_dim: int,
    ctx: ExecutionContext | None = None,
    cfg: BertBoltFfnLinear1Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or BertBoltFfnLinear1Config()
    if not cfg.bridge_binary.exists():
        raise RuntimeError(f"FFN_Linear_1 bridge binary not found: {cfg.bridge_binary}")
    if x.ndim != 3:
        raise ValueError(f"FFN_Linear_1 wrapper expects [B,S,H], got shape={x.shape}")
    bsz, seq, h = x.shape
    n = bsz * seq
    if n > 64:
        raise ValueError(f"Current bridge relies on n<=64 (NonLinear fpmath array), got n={n}")
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")

    w, bias = deterministic_ffn_linear1_params(h, out_dim, cfg.weight_seed)
    x2d = np.asarray(x, dtype=np.float64).reshape(n, h)
    w_rep = np.broadcast_to(w.reshape(1, h, out_dim), (n, h, out_dim)).copy()
    b_rep = np.broadcast_to(bias.reshape(1, out_dim), (n, out_dim)).copy()

    x0, x1 = _share_encode_fixed(x2d, cfg.ell, cfg.scale, seed=51)
    w0, w1 = _share_encode_fixed(w_rep, cfg.ell, cfg.scale, seed=53)
    b0, b1 = _share_encode_fixed(b_rep, cfg.ell, cfg.scale, seed=59)
    out_size = n * out_dim

    _log(
        ctx,
        "[ffn_linear1_wrapper] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp "
        "function=NonLinear::n_matrix_mul_iron(...)",
    )
    last_error = None
    for attempt in range(1, 6):
        port = cfg.port if cfg.port is not None else _choose_port_block(block_size=64)
        _log(
            ctx,
            f"[ffn_linear1_wrapper] input_shape=[{bsz},{seq},{h}] n={n} h={h} i={out_dim} "
            f"ell={cfg.ell} s={cfg.scale} nthreads={cfg.nthreads} port={port} attempt={attempt} seed={cfg.weight_seed}",
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p1_in = td_path / "party1_in.bin"
            p2_in = td_path / "party2_in.bin"
            p1_w = td_path / "party1_w.bin"
            p2_w = td_path / "party2_w.bin"
            p1_b = td_path / "party1_b.bin"
            p2_b = td_path / "party2_b.bin"
            p1_out = td_path / "party1_out.bin"
            p2_out = td_path / "party2_out.bin"
            x0.tofile(p1_in)
            x1.tofile(p2_in)
            w0.tofile(p1_w)
            w1.tofile(p2_w)
            b0.tofile(p1_b)
            b1.tofile(p2_b)

            cmd1 = [
                str(cfg.bridge_binary), "--party", "1", "--port", str(port), "--address", cfg.address,
                "--nthreads", str(cfg.nthreads), "--ell", str(cfg.ell), "--scale", str(cfg.scale),
                "--n", str(n), "--h", str(h), "--i", str(out_dim),
                "--input", str(p1_in), "--weight", str(p1_w), "--bias", str(p1_b), "--output", str(p1_out),
            ]
            cmd2 = [
                str(cfg.bridge_binary), "--party", "2", "--port", str(port), "--address", cfg.address,
                "--nthreads", str(cfg.nthreads), "--ell", str(cfg.ell), "--scale", str(cfg.scale),
                "--n", str(n), "--h", str(h), "--i", str(out_dim),
                "--input", str(p2_in), "--weight", str(p2_w), "--bias", str(p2_b), "--output", str(p2_out),
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
                    last_error = RuntimeError(f"FFN_Linear_1 bridge timeout; retrying with new port block: {exc}")
                    _log(ctx, "[ffn_linear1_wrapper] timeout_retry=True")
                    continue
                raise RuntimeError(f"Bridge process timeout during MPC FFN_Linear_1 execution: {exc}") from exc

            ok = p1.returncode == 0 and p2.returncode == 0
            _log(ctx, f"[ffn_linear1_wrapper] party_launch_success={ok} rc1={p1.returncode} rc2={p2.returncode}")
            _log(ctx, f"[ffn_linear1_wrapper] party1_stdout={o1.strip()}")
            _log(ctx, f"[ffn_linear1_wrapper] party2_stdout={o2.strip()}")
            if e1.strip():
                _log(ctx, f"[ffn_linear1_wrapper] party1_stderr={e1.strip()}")
            if e2.strip():
                _log(ctx, f"[ffn_linear1_wrapper] party2_stderr={e2.strip()}")
            if not ok:
                stderr_blob = f"{e1}\n{e2}".lower()
                if "address already in use" in stderr_blob and cfg.port is None:
                    last_error = RuntimeError("FFN_Linear_1 bridge bind conflict; retrying with a new port block.")
                    _log(ctx, "[ffn_linear1_wrapper] bind_conflict_retry=True")
                    continue
                raise RuntimeError("Bridge party execution failed; see logged stdout/stderr.")

            y1 = np.fromfile(p1_out, dtype=np.uint64)
            y2 = np.fromfile(p2_out, dtype=np.uint64)
            if y1.size != out_size or y2.size != out_size:
                raise RuntimeError(f"Output share size mismatch: y1={y1.size}, y2={y2.size}, expected={out_size}")
            y = _decode_recombine(y1, y2, cfg.ell, cfg.scale, (bsz, seq, out_dim))
            _log(ctx, "[ffn_linear1_wrapper] recombine_success=True")
            return y, w, bias

    raise RuntimeError(f"FFN_Linear_1 wrapper failed after retries. Last error: {last_error}")


# -- chunking wrapper ---------------------------------------------------------
#
# The BOLT_FFN_LINEAR1_BRIDGE path hard-limits `n = B*S <= 64` because the
# SCI bert_bolt NonLinear kernel statically allocates thread/pack arrays.
# The compiler/model-level shape for BERT-base is B=1, S=128, which violates
# the bridge's limit. This helper splits the flattened token dimension into
# chunks of at most 64 tokens, invokes the single-shot primitive per chunk,
# and reassembles the output. No new cryptographic primitive is introduced —
# the weights/bias are regenerated deterministically per-chunk from the same
# seed so the chunks share parameters and match the single-shot semantics.

CHUNK_MAX_N: int = 64


def run_bert_bolt_ffn_linear1_mpc_chunked(
    x: np.ndarray,
    out_dim: int,
    ctx: ExecutionContext | None = None,
    cfg: BertBoltFfnLinear1Config | None = None,
    chunk_size: int = CHUNK_MAX_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked FFN_Linear_1 MPC wrapper.

    - Splits `n = B*S` into blocks of `<= chunk_size` tokens.
    - Calls `run_bert_bolt_ffn_linear1_mpc` per chunk.
    - Reassembles and returns `(y, w, bias)` with `y` shape `[B,S,out_dim]`.

    Keeps `(w, bias)` identical across chunks because every chunk uses the
    same `cfg.weight_seed`.
    """
    cfg = cfg or BertBoltFfnLinear1Config()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"FFN_Linear_1 chunked wrapper expects [B,S,H], got {x.shape}")
    if chunk_size <= 0 or chunk_size > CHUNK_MAX_N:
        raise ValueError(
            f"chunk_size must be in [1,{CHUNK_MAX_N}]; got {chunk_size}"
        )
    bsz, seq, h = x.shape
    n = bsz * seq
    if n <= chunk_size:
        return run_bert_bolt_ffn_linear1_mpc(x, out_dim, ctx=ctx, cfg=cfg)

    x2d = x.reshape(n, h)
    pieces: list[np.ndarray] = []
    w_ref: np.ndarray | None = None
    b_ref: np.ndarray | None = None

    _log(
        ctx,
        f"[ffn_linear1_chunked] n={n} h={h} out={out_dim} chunk_size={chunk_size} "
        f"num_chunks={(n + chunk_size - 1) // chunk_size}",
    )

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = x2d[start:end].reshape(1, end - start, h)
        y_block, w, bias = run_bert_bolt_ffn_linear1_mpc(
            block, out_dim=out_dim, ctx=ctx, cfg=cfg
        )
        # chunks share seed so weights match; capture once and validate
        if w_ref is None:
            w_ref = w
            b_ref = bias
        else:
            if not np.array_equal(w, w_ref) or not np.array_equal(bias, b_ref):
                raise RuntimeError(
                    "FFN_Linear_1 chunked wrapper: per-chunk weights diverged; "
                    "check weight_seed determinism."
                )
        pieces.append(y_block.reshape(end - start, out_dim))

    y_flat = np.concatenate(pieces, axis=0)
    y = y_flat.reshape(bsz, seq, out_dim)
    assert w_ref is not None and b_ref is not None
    return y, w_ref, b_ref


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, bs_product, mpc_signature


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    """MPC FFN_Linear_1 is always feasible via the chunked wrapper."""
    del ctx
    in_shape = tuple(int(d) for d in input_shape)
    out = output_shape if output_shape is not None else in_shape
    n = bs_product(in_shape) if len(in_shape) >= 2 else 1
    return mpc_signature(
        "FFN_Linear_1",
        input_shape=in_shape,
        output_shape=out,
        feasible=True,
        notes=(
            "BOLT_FFN_LINEAR1_BRIDGE: NonLinear::n_matrix_mul_iron; "
            f"chunked in blocks of <= {CHUNK_MAX_N} tokens (n={n})."
        ),
        extras={
            "chunked": n > CHUNK_MAX_N,
            "num_chunks": (n + CHUNK_MAX_N - 1) // CHUNK_MAX_N,
            "chunk_size": CHUNK_MAX_N,
        },
    )

