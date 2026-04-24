"""Section D: solver feasibility gate + strict l >= δ enforcement.

The state-expanded solver used to price HE transitions through for any
shape, relying on the capability checker's (shape-coarse) `is_method_valid`
for feasibility. Section D adds:

- A per-node HE feasibility gate driven by `cost_signature().feasible`.
  For e.g. LayerNorm at BERT-base shapes (B*S > 16), the HE transition
  must be pruned entirely, not charged as finite cost.
- Strict `l >= δ_i` enforcement for the direct HE-execute transition
  (only emit when the level budget actually covers the op).
- Bootstrap transitions are only offered when the method advertises
  in-place bootstrap support. Otherwise the solver must pay the
  HE->MPC->HE detour.

These tests hit `_enumerate_chain_transitions` directly so we can observe
the exact transition set without running the full Dijkstra loop.
"""

from __future__ import annotations

from typing import List

import pytest

from compiler.min_cut.profiler_db import BenchmarkRecord, ConversionRecord, ProfilerDB
from compiler.state_expanded_opt.cost_model import StateExpandedCostModel
from compiler.state_expanded_opt.solver import (
    _enumerate_chain_transitions,
    _he_feasible,
    _he_method_bootstrap_supported,
)
from compiler.state_expanded_opt.state_space import State
from ir.types import OperatorNode


# -- helpers ------------------------------------------------------------------


def _both_domain_db() -> ProfilerDB:
    """Tiny DB with HE+MPC records for LayerNorm and GeLU.

    Shapes are chosen so LayerNorm can be tested at both feasible and
    infeasible HE shapes (B*S=8 vs 128).
    """
    records: List[BenchmarkRecord] = []
    for op in ("LayerNorm", "GeLU"):
        for shape in ((1, 8, 768), (1, 128, 768)):
            records.append(
                BenchmarkRecord(
                    op_type=op,
                    domain="HE",
                    method="method_he_nexus",
                    input_shape=shape,
                    output_shape=shape,
                    local_compute_ms=20.0,
                    comm_bytes=0,
                    comm_rounds=0,
                    total_latency_ms=20.0,
                    metadata={"he_level_delta": 3 if op == "LayerNorm" else 4},
                )
            )
            records.append(
                BenchmarkRecord(
                    op_type=op,
                    domain="MPC",
                    method="method_mpc_bolt",
                    input_shape=shape,
                    output_shape=shape,
                    local_compute_ms=5.0,
                    comm_bytes=0,
                    comm_rounds=0,
                    total_latency_ms=5.0,
                    metadata={},
                )
            )
    conv = [
        ConversionRecord(
            from_domain="HE",
            to_domain="MPC",
            method="method_default",
            layout_family="generic",
            tensor_shape=shape,
            local_compute_ms=1.0,
            comm_bytes=0,
            comm_rounds=0,
            total_latency_ms=1.0,
            metadata={},
        )
        for shape in ((1, 8, 768), (1, 128, 768))
    ] + [
        ConversionRecord(
            from_domain="MPC",
            to_domain="HE",
            method="method_default",
            layout_family="generic",
            tensor_shape=shape,
            local_compute_ms=1.0,
            comm_bytes=0,
            comm_rounds=0,
            total_latency_ms=1.0,
            metadata={},
        )
        for shape in ((1, 8, 768), (1, 128, 768))
    ]
    return ProfilerDB(records=records, conversion_records=conv)


def _node(op: str, shape) -> OperatorNode:
    return OperatorNode(node_id=f"{op}_0", op_type=op, input_shape=shape, output_shape=shape)


def _kinds(transitions):
    out = set()
    for _state, actions, _cost in transitions:
        for a in actions:
            out.add(a.kind)
    return out


# -- _he_feasible gate --------------------------------------------------------


def test_he_feasible_blocks_layernorm_at_bert_shape():
    model = StateExpandedCostModel(db=_both_domain_db())
    assert _he_feasible(_node("LayerNorm", (1, 128, 768)), model) is False
    assert _he_feasible(_node("LayerNorm", (1, 8, 768)), model) is True


def test_he_feasible_allows_gelu_any_shape():
    model = StateExpandedCostModel(db=_both_domain_db())
    assert _he_feasible(_node("GeLU", (1, 128, 768)), model) is True
    assert _he_feasible(_node("GeLU", (1, 8, 768)), model) is True


def test_he_method_bootstrap_unsupported_for_every_nexus_op():
    """None of the NEXUS-backed HE methods advertise in-place bootstrap."""
    for op, shape in [
        ("GeLU", (1, 128, 768)),
        ("Softmax", (1, 12, 128, 128)),
        ("LayerNorm", (1, 8, 768)),
        ("FFN_Linear_1", (1, 128, 768)),
    ]:
        assert _he_method_bootstrap_supported(_node(op, shape)) is False, op


# -- _enumerate_chain_transitions --------------------------------------------


def test_chain_transitions_he_to_he_gelu_direct_when_budget_allows():
    """GeLU δ=4; starting with level_bucket=8, direct execute is allowed."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="HE", level_bucket=8)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("GeLU", (1, 8, 768)), model, step_id=1
    )
    kinds = _kinds(transitions)
    assert "execute_he" in kinds
    # No bootstrap (NEXUS GeLU does not support in-place BS).
    assert "bootstrap" not in kinds


def test_chain_transitions_he_drops_direct_execute_when_budget_too_low():
    """GeLU δ=4; level_bucket=2 means no direct execute transition."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="HE", level_bucket=2)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("GeLU", (1, 8, 768)), model, step_id=1
    )
    # Direct execute must NOT be present.
    for next_state, actions, _ in transitions:
        if next_state.domain == "HE":
            for a in actions:
                assert a.kind != "execute_he" or a.reason != "execute", (
                    "Direct execute_he emitted despite l < delta"
                )


def test_chain_transitions_he_does_not_emit_bootstrap_when_unsupported():
    """NEXUS GeLU: bootstrap unsupported → no bootstrap transition."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="HE", level_bucket=2)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("GeLU", (1, 8, 768)), model, step_id=1
    )
    for _, actions, _ in transitions:
        for a in actions:
            assert a.kind != "bootstrap"


def test_chain_transitions_he_always_offers_mpc_detour_when_infeasible():
    """LayerNorm@(1,128,768): HE infeasible → only HE→MPC detour is emitted."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="HE", level_bucket=3)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("LayerNorm", (1, 128, 768)), model, step_id=1
    )
    # No HE execution at all.
    for _, actions, _ in transitions:
        for a in actions:
            assert a.kind != "execute_he", "HE execution emitted for infeasible shape"
    # Must have at least one detour ending in MPC.
    assert any(next_state.domain == "MPC" for next_state, _, _ in transitions)


def test_chain_transitions_mpc_starting_state_blocks_convert_up_when_infeasible():
    """From MPC, convert->HE->execute must also be dropped when HE infeasible."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="MPC", level_bucket=None)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("LayerNorm", (1, 128, 768)), model, step_id=1
    )
    for next_state, actions, _ in transitions:
        # From MPC starting state, only MPC-ending transitions allowed for
        # infeasible HE ops.
        assert next_state.domain == "MPC"
        for a in actions:
            assert a.kind != "execute_he"


def test_chain_transitions_layernorm_feasible_at_demo_shape():
    """At a feasible shape, HE execution is allowed from HE state."""
    model = StateExpandedCostModel(db=_both_domain_db())
    state = State(position=1, domain="HE", level_bucket=5)
    transitions, _ = _enumerate_chain_transitions(
        state, _node("LayerNorm", (1, 8, 768)), model, step_id=1
    )
    kinds = _kinds(transitions)
    assert "execute_he" in kinds


# -- cost-model feasible() surfaces the same verdict -------------------------


def test_cost_model_feasible_is_consistent_with_solver_gate():
    model = StateExpandedCostModel(db=_both_domain_db())
    bert = _node("LayerNorm", (1, 128, 768))
    demo = _node("LayerNorm", (1, 8, 768))
    assert model.feasible(bert, "HE") is False
    assert model.feasible(demo, "HE") is True
    # MPC side is always feasible
    assert model.feasible(bert, "MPC") is True
