"""Section A: FFN_Linear_1 / FFN_Linear_2 MPC chunking wrapper.

The BOLT_FFN_LINEAR1_BRIDGE enforces `n = B*S <= 64`. BERT-base runs with
`n = 128`, so the framework adds a chunking wrapper
(`run_bert_bolt_ffn_linear1_mpc_chunked`) that splits the flattened token
dimension into `<= 64` blocks, invokes the single-shot primitive per
block, and reassembles the result.

We cannot run the real SCI bridge in a test environment — the binary may
be missing and spawning two parallel subprocesses is expensive and flaky —
so these tests monkey-patch the single-shot function with a deterministic
numpy reference that matches the signature. This isolates and exercises:

- correct chunk count
- correct per-chunk shape
- deterministic weight/bias propagation across chunks (same seed)
- output reassembly back to [B,S,out_dim]
- chunk_size validation
- passthrough when n <= chunk_size
"""

from __future__ import annotations

import numpy as np
import pytest

from operators.linear_ffn1 import method_mpc_bolt as ffn1_mod


# -- test helpers -------------------------------------------------------------


def _fake_single_shot(x, out_dim, ctx=None, cfg=None):
    """Pure-numpy stand-in for the real SCI bridge call.

    Matches `run_bert_bolt_ffn_linear1_mpc`'s signature and returns a
    deterministic (y, w, bias) triple so the chunking reassembly can be
    checked against a single-shot reference computed over the whole tensor.
    """
    cfg = cfg or ffn1_mod.BertBoltFfnLinear1Config()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("fake_single_shot expects [B,S,H]")
    bsz, seq, h = x.shape
    n = bsz * seq
    if n > ffn1_mod.CHUNK_MAX_N:
        raise ValueError(f"fake bridge refuses n > {ffn1_mod.CHUNK_MAX_N}; got {n}")
    w, bias = ffn1_mod.deterministic_ffn_linear1_params(h, out_dim, cfg.weight_seed)
    y = x.reshape(n, h) @ w + bias  # [n, out_dim]
    return y.reshape(bsz, seq, out_dim), w, bias


@pytest.fixture
def patched_single_shot(monkeypatch):
    monkeypatch.setattr(ffn1_mod, "run_bert_bolt_ffn_linear1_mpc", _fake_single_shot)
    yield


# -- tests --------------------------------------------------------------------


def test_chunked_wrapper_matches_single_shot_reference(patched_single_shot):
    """Chunked result must equal the whole-tensor reference computation.

    Uses a deterministic (non-random) input so the test stays reproducible
    and fast; random inputs would otherwise pull in BLAS threading paths
    that can deadlock under pytest's stdout capture.
    """
    B, S, H = 1, 128, 8
    out_dim = 4
    # Construct deterministic inputs without np.random to avoid pulling in
    # BLAS thread pools during the reference matmul.
    x = np.arange(B * S * H, dtype=np.float64).reshape(B, S, H) / 1000.0

    cfg = ffn1_mod.BertBoltFfnLinear1Config()
    w, bias = ffn1_mod.deterministic_ffn_linear1_params(H, out_dim, cfg.weight_seed)
    ref = x.reshape(B * S, H) @ w + bias
    ref = ref.reshape(B, S, out_dim)

    y, w_out, b_out = ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(
        x, out_dim=out_dim, cfg=cfg
    )
    assert y.shape == (B, S, out_dim)
    assert float(np.abs(y - ref).max()) < 1e-10
    assert np.array_equal(w, w_out)
    assert np.array_equal(bias, b_out)


def test_chunked_wrapper_num_chunks(patched_single_shot):
    """n=128, chunk_size=64 -> 2 chunks; n=130 -> 3 chunks."""
    B, H = 1, 768
    out_dim = 32

    y128, _, _ = ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(
        np.zeros((B, 128, H)), out_dim=out_dim
    )
    assert y128.shape == (1, 128, out_dim)

    y130, _, _ = ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(
        np.zeros((B, 130, H)), out_dim=out_dim
    )
    assert y130.shape == (1, 130, out_dim)


def test_chunked_wrapper_passthrough_under_chunk_size(patched_single_shot):
    """When n <= CHUNK_MAX_N the wrapper calls the single-shot bridge once."""
    B, S, H = 1, 16, 768
    out_dim = 8
    x = np.ones((B, S, H))
    y, _, _ = ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(x, out_dim=out_dim)
    assert y.shape == (B, S, out_dim)


def test_chunked_wrapper_rejects_bad_chunk_size(patched_single_shot):
    x = np.zeros((1, 128, 768))
    with pytest.raises(ValueError, match="chunk_size"):
        ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(x, out_dim=8, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(x, out_dim=8, chunk_size=65)


def test_chunked_wrapper_rejects_rank_other_than_3(patched_single_shot):
    with pytest.raises(ValueError, match=r"\[B,S,H\]"):
        ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(
            np.zeros((128, 768)), out_dim=8
        )


def test_chunked_wrapper_detects_seed_divergence(monkeypatch):
    """If the per-chunk bridge ever returns inconsistent weights, the
    reassembly path must fail loudly — it would otherwise produce a
    silently wrong y."""
    counter = {"i": 0}

    def diverging_single_shot(x, out_dim, ctx=None, cfg=None):
        cfg = cfg or ffn1_mod.BertBoltFfnLinear1Config()
        x = np.asarray(x, dtype=np.float64)
        bsz, seq, h = x.shape
        seed = cfg.weight_seed + counter["i"]  # different every call
        counter["i"] += 1
        w, bias = ffn1_mod.deterministic_ffn_linear1_params(h, out_dim, seed)
        y = x.reshape(bsz * seq, h) @ w + bias
        return y.reshape(bsz, seq, out_dim), w, bias

    monkeypatch.setattr(ffn1_mod, "run_bert_bolt_ffn_linear1_mpc", diverging_single_shot)

    with pytest.raises(RuntimeError, match="weights diverged"):
        ffn1_mod.run_bert_bolt_ffn_linear1_mpc_chunked(
            np.zeros((1, 128, 768)), out_dim=8
        )


def test_ffn_linear_2_wrapper_also_uses_chunked(monkeypatch):
    """FFN_Linear_2 delegates to the chunked wrapper, not the single-shot."""
    from operators.ffn_linear_2 import method_mpc_bolt as ffn2_mod

    calls = {"chunked": 0}

    def fake_chunked(x, out_dim, ctx=None, cfg=None):
        calls["chunked"] += 1
        cfg = cfg or ffn1_mod.BertBoltFfnLinear1Config()
        x = np.asarray(x, dtype=np.float64)
        w, bias = ffn1_mod.deterministic_ffn_linear1_params(
            x.shape[-1], out_dim, cfg.weight_seed
        )
        y = x.reshape(-1, x.shape[-1]) @ w + bias
        return y.reshape(*x.shape[:-1], out_dim), w, bias

    monkeypatch.setattr(
        ffn2_mod, "run_bert_bolt_ffn_linear1_mpc_chunked", fake_chunked
    )
    y, _, _ = ffn2_mod.run_bert_bolt_ffn_linear2_mpc(
        np.zeros((1, 128, 3072)), out_dim=768
    )
    assert calls["chunked"] == 1
    assert y.shape == (1, 128, 768)
