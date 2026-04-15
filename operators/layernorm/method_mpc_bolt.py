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
class BertBoltLayerNormConfig:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    bridge_binary: Path = Path(
        "/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_LAYERNORM_BRIDGE"
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


def _resolve_affine_params(
    ctx: ExecutionContext | None, dim: int, array_size: int
) -> tuple[np.ndarray, np.ndarray, str]:
    weight = None if ctx is None else ctx.params.get("layernorm_weight")
    bias = None if ctx is None else ctx.params.get("layernorm_bias")
    mode = "default_affine"
    if weight is None:
        w2d = np.ones((dim, array_size), dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape == (array_size,):
            w2d = np.broadcast_to(w.reshape(1, array_size), (dim, array_size)).copy()
            mode = "broadcast_vector_affine"
        elif w.shape == (dim, array_size):
            w2d = w
            mode = "full_row_affine"
        else:
            raise ValueError(f"layernorm_weight shape must be [{array_size}] or [{dim},{array_size}], got {w.shape}")
    if bias is None:
        b2d = np.zeros((dim, array_size), dtype=np.float64)
    else:
        b = np.asarray(bias, dtype=np.float64)
        if b.shape == (array_size,):
            b2d = np.broadcast_to(b.reshape(1, array_size), (dim, array_size)).copy()
            mode = "broadcast_vector_affine" if mode == "default_affine" else mode
        elif b.shape == (dim, array_size):
            b2d = b
            mode = "full_row_affine" if mode == "default_affine" else mode
        else:
            raise ValueError(f"layernorm_bias shape must be [{array_size}] or [{dim},{array_size}], got {b.shape}")
    return w2d, b2d, mode


def run_bert_bolt_layernorm_mpc(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: BertBoltLayerNormConfig | None = None,
) -> np.ndarray:
    cfg = cfg or BertBoltLayerNormConfig()
    if not cfg.bridge_binary.exists():
        raise RuntimeError(f"LayerNorm bridge binary not found: {cfg.bridge_binary}")
    if x.ndim < 2:
        raise ValueError(f"LayerNorm wrapper expects ndim>=2 with last axis as feature width, got shape={x.shape}")

    original_shape = tuple(x.shape)
    dim = int(np.prod(original_shape[:-1]))
    array_size = int(original_shape[-1])
    x2d = np.asarray(x, dtype=np.float64).reshape(dim, array_size)
    w2d, b2d, affine_mode = _resolve_affine_params(ctx, dim, array_size)
    flat_size = dim * array_size
    in0, in1 = _share_encode_fixed(x2d, cfg.ell, cfg.scale, seed=31)
    w0, w1 = _share_encode_fixed(w2d, cfg.ell, cfg.scale, seed=37)
    b0, b1 = _share_encode_fixed(b2d, cfg.ell, cfg.scale, seed=41)

    _log(
        ctx,
        "[layernorm_wrapper] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp "
        "function=NonLinear::layer_norm(int nthreads, uint64_t* input, uint64_t* output, uint64_t* weight, uint64_t* bias, int dim, int array_size, int ell, int s)",
    )

    last_error = None
    for attempt in range(1, 6):
        port = cfg.port if cfg.port is not None else _choose_port_block(block_size=64)
        _log(
            ctx,
            f"[layernorm_wrapper] input_shape={list(original_shape)} dim={dim} array_size={array_size} flat_size={flat_size} "
            f"ell={cfg.ell} s={cfg.scale} nthreads={cfg.nthreads} port={port} attempt={attempt} affine_mode={affine_mode}",
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
            in0.tofile(p1_in)
            in1.tofile(p2_in)
            w0.tofile(p1_w)
            w1.tofile(p2_w)
            b0.tofile(p1_b)
            b1.tofile(p2_b)

            cmd1 = [
                str(cfg.bridge_binary), "--party", "1", "--port", str(port), "--address", cfg.address,
                "--nthreads", str(cfg.nthreads), "--ell", str(cfg.ell), "--scale", str(cfg.scale),
                "--dim", str(dim), "--array_size", str(array_size),
                "--input", str(p1_in), "--weight", str(p1_w), "--bias", str(p1_b), "--output", str(p1_out),
            ]
            cmd2 = [
                str(cfg.bridge_binary), "--party", "2", "--port", str(port), "--address", cfg.address,
                "--nthreads", str(cfg.nthreads), "--ell", str(cfg.ell), "--scale", str(cfg.scale),
                "--dim", str(dim), "--array_size", str(array_size),
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
                    last_error = RuntimeError(
                        f"LayerNorm bridge timeout on attempt={attempt}; retrying with a new port block. Detail: {exc}"
                    )
                    _log(ctx, "[layernorm_wrapper] timeout_retry=True")
                    continue
                raise RuntimeError(
                    f"Bridge process timeout during MPC LayerNorm execution: {exc}. "
                    "This indicates SCI runtime setup/connectivity failure."
                ) from exc

            ok = p1.returncode == 0 and p2.returncode == 0
            _log(ctx, f"[layernorm_wrapper] party_launch_success={ok} rc1={p1.returncode} rc2={p2.returncode}")
            _log(ctx, f"[layernorm_wrapper] party1_stdout={o1.strip()}")
            _log(ctx, f"[layernorm_wrapper] party2_stdout={o2.strip()}")
            if e1.strip():
                _log(ctx, f"[layernorm_wrapper] party1_stderr={e1.strip()}")
            if e2.strip():
                _log(ctx, f"[layernorm_wrapper] party2_stderr={e2.strip()}")
            if not ok:
                stderr_blob = f"{e1}\n{e2}".lower()
                if "address already in use" in stderr_blob and cfg.port is None:
                    last_error = RuntimeError("LayerNorm bridge bind conflict; retrying with a new port block.")
                    _log(ctx, "[layernorm_wrapper] bind_conflict_retry=True")
                    continue
                raise RuntimeError("Bridge party execution failed; see logged stdout/stderr.")

            y1 = np.fromfile(p1_out, dtype=np.uint64)
            y2 = np.fromfile(p2_out, dtype=np.uint64)
            if y1.size != flat_size or y2.size != flat_size:
                raise RuntimeError(f"Output share size mismatch: y1={y1.size}, y2={y2.size}, expected={flat_size}")
            y = _decode_recombine(y1, y2, cfg.ell, cfg.scale, original_shape)
            _log(ctx, "[layernorm_wrapper] recombine_success=True")
            return y

    raise RuntimeError(f"LayerNorm wrapper failed after retries. Last error: {last_error}")


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, mpc_signature


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    out = output_shape if output_shape is not None else input_shape
    return mpc_signature(
        "LayerNorm",
        input_shape=input_shape,
        output_shape=out,
        feasible=True,
        notes="BOLT_LAYERNORM_BRIDGE: NonLinear::layer_norm (SCI), full affine support",
    )

