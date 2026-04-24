"""Section A: Embedding/Linear_QKV are folded out of the compiler DAG.

Verifies:
- BERT_OPERATOR_SEQUENCE keeps Embedding and Linear_QKV as semantic entries
  with new roles ("pre_compile" / "fused_into_next").
- bert_executed_operator_sequence() returns only the 9 compiler vertices
  that have real HE/MPC methods in this tree.
- Out_Projection appears in the executed sequence.
- capability_registry reflects the new roles.
"""

from __future__ import annotations

from runtime.capabilities import CapabilityStatus, capability_registry
from runtime.operator_specs import (
    BERT_OPERATOR_SEQUENCE,
    bert_executed_operator_sequence,
)
from runtime.types import BackendType


EXPECTED_EXECUTED = [
    "Attention_QK_MatMul",
    "Softmax",
    "Attention_V_MatMul",
    "Out_Projection",
    "Residual_Add",
    "LayerNorm",
    "FFN_Linear_1",
    "GeLU",
    "FFN_Linear_2",
]


def test_bert_operator_sequence_keeps_embedding_and_linear_qkv():
    names = [spec.name for spec in BERT_OPERATOR_SEQUENCE]
    assert "Embedding" in names
    assert "Linear_QKV" in names


def test_embedding_role_is_pre_compile():
    spec = next(s for s in BERT_OPERATOR_SEQUENCE if s.name == "Embedding")
    assert spec.role == "pre_compile"


def test_linear_qkv_role_is_fused_into_next():
    spec = next(s for s in BERT_OPERATOR_SEQUENCE if s.name == "Linear_QKV")
    assert spec.role == "fused_into_next"
    assert spec.fused_into == "Attention_QK_MatMul"


def test_executed_sequence_matches_expected_order():
    executed = [s.name for s in bert_executed_operator_sequence()]
    assert executed == EXPECTED_EXECUTED


def test_executed_sequence_drops_embedding_and_linear_qkv():
    executed = [s.name for s in bert_executed_operator_sequence()]
    assert "Embedding" not in executed
    assert "Linear_QKV" not in executed


def test_capability_registry_marks_embedding_pre_compile():
    for backend in (BackendType.MPC, BackendType.HE):
        assert capability_registry.get_status("Embedding", backend) == CapabilityStatus.PRE_COMPILE


def test_capability_registry_marks_linear_qkv_fused():
    for backend in (BackendType.MPC, BackendType.HE):
        assert capability_registry.get_status("Linear_QKV", backend) == CapabilityStatus.FUSED_INTO_NEXT


def test_capability_registry_marks_out_projection_real_mpc_restricted_he():
    assert capability_registry.get_status("Out_Projection", BackendType.MPC) == CapabilityStatus.REAL_INTEGRATED
    assert capability_registry.get_status("Out_Projection", BackendType.HE) == CapabilityStatus.RESTRICTED_INTEGRATED


def test_every_executed_op_has_some_real_backend():
    """Section A invariant: the executed DAG must be runnable end-to-end."""
    for spec in bert_executed_operator_sequence():
        statuses = {
            backend: capability_registry.get_status(spec.name, backend)
            for backend in (BackendType.MPC, BackendType.HE)
        }
        real = {CapabilityStatus.REAL_INTEGRATED, CapabilityStatus.RESTRICTED_INTEGRATED}
        assert any(s in real for s in statuses.values()), (spec.name, statuses)
