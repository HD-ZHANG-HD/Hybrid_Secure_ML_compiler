"""Section 0 + A: tests for the OperatorCostSignature dataclass and helpers.

Covers:
- construction via `he_signature` / `mpc_signature`
- immutability (frozen dataclass)
- `with_feasible` produces a new instance with the new verdict
- `bs_product` handles rank-0/1/N shapes
- `BootstrapUnsupportedError` is a RuntimeError subclass
"""

from __future__ import annotations

import pytest

from operators._cost_signature import (
    BootstrapUnsupportedError,
    OperatorCostSignature,
    bs_product,
    he_signature,
    mpc_signature,
)


def test_he_signature_populates_fields():
    sig = he_signature(
        "GeLU",
        input_shape=(1, 128, 768),
        output_shape=(1, 128, 768),
        level_delta=4,
        bootstrap_supported=False,
        feasible=True,
        notes="test",
    )
    assert sig.op_type == "GeLU"
    assert sig.domain == "HE"
    assert sig.input_shape == (1, 128, 768)
    assert sig.output_shape == (1, 128, 768)
    assert sig.he_level_delta == 4
    assert sig.he_min_level_required == 4
    assert sig.he_bootstrap_supported is False
    assert sig.feasible is True
    assert sig.notes == "test"


def test_mpc_signature_zero_depth():
    sig = mpc_signature(
        "Softmax",
        input_shape=(1, 128, 128),
        output_shape=(1, 128, 128),
    )
    assert sig.domain == "MPC"
    assert sig.he_level_delta == 0
    assert sig.he_min_level_required == 0
    assert sig.he_bootstrap_supported is False
    assert sig.feasible is True


def test_signature_is_frozen():
    sig = mpc_signature("GeLU", input_shape=(1,), output_shape=(1,))
    with pytest.raises(Exception):  # FrozenInstanceError
        sig.feasible = False  # type: ignore[misc]


def test_with_feasible_returns_new_instance():
    sig = he_signature(
        "LayerNorm",
        input_shape=(1, 128, 768),
        output_shape=(1, 128, 768),
        level_delta=3,
    )
    blocked = sig.with_feasible(False, "shape too large")
    assert sig.feasible is True
    assert blocked.feasible is False
    assert blocked.notes == "shape too large"
    # original unchanged
    assert sig.notes != "shape too large"


def test_with_feasible_preserves_other_fields():
    sig = he_signature(
        "Softmax",
        input_shape=(2, 4, 4),
        output_shape=(2, 4, 4),
        level_delta=8,
        bootstrap_supported=False,
        extras={"poly": 4096},
    )
    blocked = sig.with_feasible(False, "infeasible")
    assert blocked.he_level_delta == 8
    assert blocked.extras == {"poly": 4096}
    assert blocked.op_type == "Softmax"


def test_bs_product_handles_small_shapes():
    assert bs_product(()) == 1
    assert bs_product((5,)) == 1
    assert bs_product((2, 3)) == 6
    assert bs_product((2, 3, 4)) == 6  # only first two dims count
    assert bs_product((1, 128, 768)) == 128


def test_bootstrap_unsupported_is_runtime_error():
    assert issubclass(BootstrapUnsupportedError, RuntimeError)


def test_he_signature_coerces_tuple():
    """Lists are accepted; stored as tuples."""
    sig = he_signature(
        "Foo", input_shape=[1, 2, 3], output_shape=[1, 2, 3], level_delta=1
    )
    assert sig.input_shape == (1, 2, 3)
    assert isinstance(sig.input_shape, tuple)


def test_mpc_signature_extras_defaults_empty():
    sig = mpc_signature("GeLU", input_shape=(1,), output_shape=(1,))
    assert sig.extras == {}
