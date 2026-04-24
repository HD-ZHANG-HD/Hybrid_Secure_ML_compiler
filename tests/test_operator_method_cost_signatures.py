"""Section A: every kept operator method_*.py exposes `cost_signature()`.

For each method module:
- `cost_signature(input_shape, output_shape, ctx)` is callable
- It returns an `OperatorCostSignature`
- HE methods report the correct hard feasibility verdict for BERT-base
  shapes (BERT-base = B=1, S=128, H=768)
- MPC methods are always feasible
- δ_i matches the seed values from the todo.md decision table
"""

from __future__ import annotations

import pytest

from operators._cost_signature import OperatorCostSignature
from runtime.types import ExecutionContext

BERT_INPUT = (1, 128, 768)


# -- HE methods ---------------------------------------------------------------


def test_he_gelu_cost_signature():
    from operators.gelu.method_he_nexus import cost_signature

    sig = cost_signature(BERT_INPUT, BERT_INPUT)
    assert isinstance(sig, OperatorCostSignature)
    assert sig.domain == "HE"
    assert sig.op_type == "GeLU"
    assert sig.he_level_delta == 4
    assert sig.feasible is True  # shape-agnostic
    assert sig.he_bootstrap_supported is False


def test_he_softmax_cost_signature():
    from operators.softmax.method_he_nexus import cost_signature

    sig = cost_signature((1, 12, 128, 128))
    assert sig.domain == "HE"
    assert sig.he_level_delta == 8
    assert sig.feasible is True


def test_he_layernorm_blocked_at_bert_base():
    from operators.layernorm.method_he_nexus import cost_signature

    sig = cost_signature(BERT_INPUT)
    assert sig.feasible is False
    assert "B*S" in sig.notes
    assert sig.he_level_delta == 3


def test_he_layernorm_feasible_at_demo_shape():
    from operators.layernorm.method_he_nexus import cost_signature

    sig = cost_signature((1, 16, 768))
    assert sig.feasible is True


def test_he_layernorm_blocked_by_wrong_hidden():
    from operators.layernorm.method_he_nexus import cost_signature

    sig = cost_signature((1, 4, 512))
    assert sig.feasible is False


def test_he_layernorm_affine_blocks_feasibility():
    from operators.layernorm.method_he_nexus import cost_signature

    ctx = ExecutionContext(params={"layernorm_weight": [1.0]})
    sig = cost_signature((1, 8, 768), ctx=ctx)
    assert sig.feasible is False
    assert "affine" in sig.notes.lower()


def test_he_ffn_linear1_feasibility():
    from operators.linear_ffn1.method_he_nexus import cost_signature

    feasible = cost_signature((1, 128, 768))
    assert feasible.feasible is True
    assert feasible.he_level_delta == 1
    assert feasible.extras["poly_modulus_degree"] == 4096

    blocked = cost_signature((1, 128, 512))
    assert blocked.feasible is False

    blocked2 = cost_signature((16, 512, 768))  # B*S = 8192 > 4096
    assert blocked2.feasible is False


def test_he_ffn_linear2_cost_signature():
    from operators.ffn_linear_2.method_he_nexus import cost_signature

    sig = cost_signature((1, 128, 3072), (1, 128, 768))
    assert sig.feasible is True
    assert sig.he_level_delta == 1


def test_he_attention_qk_feasibility_matrix():
    from operators.attention_qk_matmul.method_he_nexus import cost_signature

    # Honours the packed [3,B,S,768] contract, BERT-base
    good = cost_signature((3, 1, 128, 768))
    assert good.feasible is True
    assert good.he_level_delta == 1

    # S > 128 is outside the NEXUS contract
    assert cost_signature((3, 1, 256, 768)).feasible is False
    # Not packed
    assert cost_signature((1, 128, 768)).feasible is False
    # Wrong hidden
    assert cost_signature((3, 1, 128, 512)).feasible is False


def test_he_attention_v_feasibility_matrix():
    from operators.attention_v_matmul.method_he_nexus import cost_signature

    assert cost_signature((3, 1, 128, 768)).feasible is True
    assert cost_signature((3, 9, 128, 768)).feasible is False  # B=9 > 8


# -- MPC methods --------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "operators.gelu.method_mpc_bolt",
        "operators.softmax.method_mpc_bolt",
        "operators.layernorm.method_mpc_bolt",
        "operators.linear_ffn1.method_mpc_bolt",
        "operators.ffn_linear_2.method_mpc_bolt",
        "operators.attention_qk_matmul.method_mpc",
        "operators.attention_v_matmul.method_mpc",
    ],
)
def test_mpc_methods_expose_feasible_cost_signature(module_path: str):
    import importlib

    mod = importlib.import_module(module_path)
    assert hasattr(mod, "cost_signature"), f"{module_path} missing cost_signature()"
    sig = mod.cost_signature((1, 128, 768))
    assert sig.domain == "MPC"
    assert sig.feasible is True
    assert sig.he_level_delta == 0
    assert isinstance(sig, OperatorCostSignature)


def test_mpc_ffn_linear1_cost_signature_marks_chunked_shape():
    from operators.linear_ffn1.method_mpc_bolt import CHUNK_MAX_N, cost_signature

    # B*S = 128 > 64 -> should be chunked
    sig = cost_signature((1, 128, 768))
    assert sig.extras["chunked"] is True
    assert sig.extras["num_chunks"] == 2
    assert sig.extras["chunk_size"] == CHUNK_MAX_N

    # B*S = 4 -> single-shot
    small = cost_signature((1, 4, 768))
    assert small.extras["chunked"] is False
    assert small.extras["num_chunks"] == 1


def test_residual_add_cost_signature_both_domains():
    from operators.residual_add.method_runtime_default import cost_signature

    he_sig = cost_signature((1, 128, 768), domain="HE")
    mpc_sig = cost_signature((1, 128, 768), domain="MPC")
    assert he_sig.domain == "HE"
    assert mpc_sig.domain == "MPC"
    # Residual_Add has zero multiplicative depth
    assert he_sig.he_level_delta == 0
    assert mpc_sig.he_level_delta == 0
    assert he_sig.feasible and mpc_sig.feasible


# -- bootstrap hooks ----------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "operators.gelu.method_he_nexus",
        "operators.softmax.method_he_nexus",
        "operators.layernorm.method_he_nexus",
        "operators.linear_ffn1.method_he_nexus",
        "operators.ffn_linear_2.method_he_nexus",
        "operators.attention_qk_matmul.method_he_nexus",
        "operators.attention_v_matmul.method_he_nexus",
    ],
)
def test_he_methods_raise_bootstrap_unsupported(module_path: str):
    """None of the NEXUS-backed HE methods support in-place bootstrap."""
    import importlib

    from operators._cost_signature import BootstrapUnsupportedError

    mod = importlib.import_module(module_path)
    assert hasattr(mod, "bootstrap"), f"{module_path} missing bootstrap()"
    with pytest.raises(BootstrapUnsupportedError):
        mod.bootstrap(None)
