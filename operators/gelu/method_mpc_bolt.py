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
class BertBoltGeluConfig:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    bridge_binary: Path = Path(
        "/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_GELU_BRIDGE"
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


def _share_encode_fixed(x: np.ndarray, ell: int, scale: int) -> tuple[np.ndarray, np.ndarray, int]:
    mask = (1 << ell) - 1
    flat = x.reshape(-1)
    q = np.round(flat * (1 << scale)).astype(np.int64)
    q_u64 = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(0)
    share_0 = rng.integers(0, 1 << ell, size=q_u64.size, dtype=np.uint64)
    share_1 = (q_u64 - share_0) & np.uint64(mask)
    return share_0, share_1, q_u64.size


def _decode_recombine(share_0: np.ndarray, share_1: np.ndarray, ell: int, scale: int, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.uint64((1 << ell) - 1)
    combined = (share_0 + share_1) & mask
    signed = combined.astype(np.int64)
    sign_cut = 1 << (ell - 1)
    signed = np.where(signed >= sign_cut, signed - (1 << ell), signed)
    return (signed.astype(np.float64) / float(1 << scale)).reshape(shape)


def run_bert_bolt_gelu_mpc(x: np.ndarray, ctx: ExecutionContext | None = None, cfg: BertBoltGeluConfig | None = None) -> np.ndarray:
    cfg = cfg or BertBoltGeluConfig()
    if not cfg.bridge_binary.exists():
        raise RuntimeError(f"GELU bridge binary not found: {cfg.bridge_binary}")
    if x.ndim != 3:
        raise ValueError(f"GeLU wrapper expects [B, S, H], got shape={x.shape}")

    b, s, h = x.shape
    share_0, share_1, size = _share_encode_fixed(np.asarray(x, dtype=np.float64), cfg.ell, cfg.scale)
    _log(
        ctx,
        "[gelu_wrapper] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp "
        "function=NonLinear::gelu(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s)",
    )
    port = cfg.port if cfg.port is not None else _choose_port_block(block_size=64)
    _log(
        ctx,
        f"[gelu_wrapper] shape=[{b},{s},{h}] flattened_size={size} "
        f"ell={cfg.ell} s={cfg.scale} nthreads={cfg.nthreads} port={port}",
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in0 = td_path / "party1_in.bin"
        in1 = td_path / "party2_in.bin"
        out0 = td_path / "party1_out.bin"
        out1 = td_path / "party2_out.bin"
        share_0.tofile(in0)
        share_1.tofile(in1)

        cmd0 = [
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
            "--size",
            str(size),
            "--input",
            str(in0),
            "--output",
            str(out0),
        ]
        cmd1 = [
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
            "--size",
            str(size),
            "--input",
            str(in1),
            "--output",
            str(out1),
        ]
        p0 = subprocess.Popen(cmd0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Let ALICE bind/listen first; Bob then connects to same port block.
        time.sleep(0.3)
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out_p0, err_p0 = p0.communicate(timeout=120)
            out_p1, err_p1 = p1.communicate(timeout=120)
        except subprocess.TimeoutExpired as exc:
            p0.kill()
            p1.kill()
            raise RuntimeError(
                f"Bridge process timeout during MPC GeLU execution: {exc}. "
                "This indicates SCI runtime setup/connectivity failure."
            ) from exc

        launched_ok = p0.returncode == 0 and p1.returncode == 0
        _log(ctx, f"[gelu_wrapper] party_launch_success={launched_ok} rc1={p0.returncode} rc2={p1.returncode}")
        _log(ctx, f"[gelu_wrapper] party1_stdout={out_p0.strip()}")
        _log(ctx, f"[gelu_wrapper] party2_stdout={out_p1.strip()}")
        if err_p0.strip():
            _log(ctx, f"[gelu_wrapper] party1_stderr={err_p0.strip()}")
        if err_p1.strip():
            _log(ctx, f"[gelu_wrapper] party2_stderr={err_p1.strip()}")
        if not launched_ok:
            raise RuntimeError("Bridge party execution failed; see logged stdout/stderr.")

        y0 = np.fromfile(out0, dtype=np.uint64)
        y1 = np.fromfile(out1, dtype=np.uint64)
        if y0.size != size or y1.size != size:
            raise RuntimeError(f"Output share size mismatch: y0={y0.size}, y1={y1.size}, expected={size}")

    y = _decode_recombine(y0, y1, cfg.ell, cfg.scale, (b, s, h))
    _log(ctx, "[gelu_wrapper] recombine_success=True")
    return y


# -- cost signature -----------------------------------------------------------

from operators._cost_signature import OperatorCostSignature, mpc_signature


def cost_signature(input_shape, output_shape=None, ctx=None) -> OperatorCostSignature:
    del ctx
    out = output_shape if output_shape is not None else input_shape
    return mpc_signature(
        "GeLU",
        input_shape=input_shape,
        output_shape=out,
        feasible=True,
        notes="BOLT_GELU_BRIDGE: NonLinear::gelu (SCI)",
    )

