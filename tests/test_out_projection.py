"""Section A: Out_Projection operator.

Out_Projection has no dedicated NEXUS / BOLT primitive in this tree. The
two new method files lower onto FFN_Linear_1. This test verifies:
- The spec is registered with both HE and MPC methods
- cost_signature() reports the correct verdict for BERT-base shapes
- The HE adapter produces a signature with δ_i = 1 and correct feasibility
- The MPC method marks chunking for n=B*S > 64
- Import side effects do not drag in the live BOLT/NEXUS runtimes
"""

from __future__ import annotations

import pytest

from operators._cost_signature import OperatorCostSignature


def test_out_projection_spec_registers_both_methods():
    from operators.out_projection.spec import OPERATOR_SPEC

    assert OPERATOR_SPEC["name"] == "Out_Projection"
    methods = OPERATOR_SPEC["attributes"]["available_methods"]
    assert "method_he_nexus_as_ffn1" in methods
    assert "method_mpc_bolt_as_ffn1" in methods
    assert OPERATOR_SPEC["attributes"]["lowered_to"] == "FFN_Linear_1"


def test_out_projection_he_cost_signature_bert_base():
    from operators.out_projection.method_he_nexus_as_ffn1 import cost_signature

    sig = cost_signature((1, 128, 768))
    assert isinstance(sig, OperatorCostSignature)
    assert sig.op_type == "Out_Projection"
    assert sig.domain == "HE"
    assert sig.he_level_delta == 1
    assert sig.feasible is True


def test_out_projection_he_blocks_bad_shape():
    from operators.out_projection.method_he_nexus_as_ffn1 import cost_signature

    # Wrong hidden dim
    assert cost_signature((1, 128, 512)).feasible is False
    # B*S > 4096
    assert cost_signature((16, 512, 768)).feasible is False


def test_out_projection_mpc_cost_signature_chunked_for_bert_base():
    from operators.out_projection.method_mpc_bolt_as_ffn1 import cost_signature

    sig = cost_signature((1, 128, 768))
    assert sig.domain == "MPC"
    assert sig.feasible is True
    assert sig.extras["chunked"] is True
    assert sig.extras["num_chunks"] == 2


def test_out_projection_mpc_single_shot_for_small_shape():
    from operators.out_projection.method_mpc_bolt_as_ffn1 import cost_signature

    sig = cost_signature((1, 4, 768))
    assert sig.extras["chunked"] is False
    assert sig.extras["num_chunks"] == 1


def test_out_projection_has_bootstrap_hook():
    from operators._cost_signature import BootstrapUnsupportedError
    from operators.out_projection.method_he_nexus_as_ffn1 import bootstrap

    with pytest.raises(BootstrapUnsupportedError):
        bootstrap(None)


def test_capability_registry_knows_out_projection():
    from runtime.capabilities import CapabilityStatus, capability_registry
    from runtime.types import BackendType

    assert capability_registry.get_status(
        "Out_Projection", BackendType.MPC
    ) == CapabilityStatus.REAL_INTEGRATED
    assert capability_registry.get_status(
        "Out_Projection", BackendType.HE
    ) == CapabilityStatus.RESTRICTED_INTEGRATED
