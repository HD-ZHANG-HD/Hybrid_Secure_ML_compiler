"""Section B: unified HardwareAwareCostModel + NetworkProfile.

Covers:
- `NetworkProfile.comm_ms` implements the paper's linear
  `bytes/bandwidth + rounds*RTT` model and returns 0 for empty comm.
- `default_method_for` returns the right canonical method per backend.
- `cost_signature_for_method` dispatches into every registered method
  module and returns the feasibility verdict.
- `level_delta_for_method` replaces the static `DEFAULT_LEVEL_DELTAS`
  — δ_i is a property of the method+shape, not the op_type alone.
- `HardwareAwareCostModel.operator_cost` scales MPC communication
  according to the active `NetworkProfile`.
- `HardwareAwareCostModel.feasibility` surfaces the method's verdict
  up to the compiler so infeasible HE branches can be pruned.
- `StateExpandedCostModel.level_delta` prefers the method-driven
  value over the static fallback dict.
"""

from __future__ import annotations

from compiler.cost_model import (
    HardwareAwareCostModel,
    METHOD_MODULE_MAP,
    NetworkProfile,
    cost_signature_for_method,
    default_method_for,
    level_delta_for_method,
)
from compiler.min_cut.profiler_db import BenchmarkRecord, ConversionRecord, ProfilerDB
from compiler.state_expanded_opt.cost_model import (
    DEFAULT_LEVEL_DELTAS,
    StateExpandedCostModel,
)
from ir.types import OperatorNode


# -- NetworkProfile -----------------------------------------------------------


def test_network_profile_defaults_are_lan():
    p = NetworkProfile()
    assert p.bandwidth_mbps == 1000.0
    assert p.rtt_ms == 1.0


def test_network_profile_lan_and_wan_helpers():
    lan = NetworkProfile.lan()
    wan = NetworkProfile.wan()
    assert lan.name == "lan"
    assert wan.name == "wan"
    assert lan.bandwidth_mbps == 1000.0
    assert wan.bandwidth_mbps == 100.0
    assert wan.rtt_ms == 40.0


def test_network_profile_comm_ms_zero_comm():
    p = NetworkProfile.lan()
    assert p.comm_ms(0, 0) == 0.0


def test_network_profile_comm_ms_scales_linearly_with_bandwidth():
    lan = NetworkProfile(bandwidth_mbps=1000.0, rtt_ms=0.0)
    wan = NetworkProfile(bandwidth_mbps=100.0, rtt_ms=0.0)
    lan_ms = lan.comm_ms(1_000_000, 0)
    wan_ms = wan.comm_ms(1_000_000, 0)
    # WAN is 10x slower → 10x more ms for the same bytes
    assert abs(wan_ms - 10 * lan_ms) < 1e-6


def test_network_profile_comm_ms_adds_rtt_per_round():
    p = NetworkProfile(bandwidth_mbps=1_000_000.0, rtt_ms=5.0)
    # bandwidth huge, so byte cost ~0; 7 rounds => 35 ms
    assert abs(p.comm_ms(1024, 7) - 35.0) < 1e-4


def test_network_profile_is_frozen_dataclass():
    import pytest

    p = NetworkProfile()
    with pytest.raises(Exception):
        p.bandwidth_mbps = 10.0  # type: ignore[misc]


# -- default_method_for -------------------------------------------------------


def test_default_method_for_he_and_mpc():
    assert default_method_for("GeLU", "HE") == "method_he_nexus"
    assert default_method_for("GeLU", "MPC") == "method_mpc_bolt"
    assert default_method_for("Attention_QK_MatMul", "HE") == "method_he_nexus"
    assert default_method_for("Attention_QK_MatMul", "MPC") == "method_mpc"
    assert default_method_for("Out_Projection", "HE") == "method_he_nexus_as_ffn1"
    assert default_method_for("Out_Projection", "MPC") == "method_mpc_bolt_as_ffn1"


def test_default_method_for_unknown_op_returns_none():
    assert default_method_for("NonexistentOp", "HE") is None


# -- cost_signature_for_method -----------------------------------------------


def test_method_module_map_covers_every_executed_op():
    from runtime.operator_specs import bert_executed_operator_sequence
    from runtime.types import BackendType

    missing = []
    for spec in bert_executed_operator_sequence():
        for backend in ("HE", "MPC"):
            method = default_method_for(spec.name, backend)
            if method is None:
                continue
            if (spec.name, method) not in METHOD_MODULE_MAP:
                missing.append((spec.name, backend, method))
    assert missing == [], f"unmapped methods: {missing}"


def test_cost_signature_for_method_dispatches_to_he_module():
    sig = cost_signature_for_method("GeLU", "method_he_nexus", (1, 128, 768))
    assert sig is not None
    assert sig.domain == "HE"
    assert sig.he_level_delta == 4
    assert sig.feasible is True


def test_cost_signature_for_method_returns_none_for_unmapped():
    assert cost_signature_for_method("Bogus", "method_bogus", (1,)) is None


def test_cost_signature_for_method_handles_residual_add_domain_kwarg():
    he_sig = cost_signature_for_method(
        "Residual_Add", "method_runtime_default", (1, 128, 768)
    )
    assert he_sig is not None
    assert he_sig.domain == "HE"
    assert he_sig.he_level_delta == 0


# -- level_delta_for_method ---------------------------------------------------


def test_level_delta_for_method_matches_seed_values():
    expected = {
        ("GeLU", "method_he_nexus"): 4,
        ("Softmax", "method_he_nexus"): 8,
        ("LayerNorm", "method_he_nexus"): 3,
        ("FFN_Linear_1", "method_he_nexus"): 1,
        ("FFN_Linear_2", "method_he_nexus"): 1,
        ("Attention_QK_MatMul", "method_he_nexus"): 1,
        ("Attention_V_MatMul", "method_he_nexus"): 1,
        ("Residual_Add", "method_runtime_default"): 0,
    }
    # Use a feasible shape for every method (LayerNorm uses demo shape)
    shapes = {
        ("GeLU", "method_he_nexus"): (1, 8, 768),
        ("Softmax", "method_he_nexus"): (1, 12, 4, 4),
        ("LayerNorm", "method_he_nexus"): (1, 8, 768),
        ("FFN_Linear_1", "method_he_nexus"): (1, 8, 768),
        ("FFN_Linear_2", "method_he_nexus"): (1, 8, 3072),
        ("Attention_QK_MatMul", "method_he_nexus"): (3, 1, 128, 768),
        ("Attention_V_MatMul", "method_he_nexus"): (3, 1, 128, 768),
        ("Residual_Add", "method_runtime_default"): (1, 8, 768),
    }
    for key, want in expected.items():
        op, method = key
        got = level_delta_for_method(op, method, shapes[key])
        assert got == want, f"{key}: want {want}, got {got}"


def test_level_delta_for_mpc_method_is_zero():
    assert level_delta_for_method("GeLU", "method_mpc_bolt", (1, 128, 768)) == 0


def test_level_delta_for_unmapped_is_none():
    assert level_delta_for_method("Nope", "method_none", (1,)) is None


# -- HardwareAwareCostModel ---------------------------------------------------


def _small_db() -> ProfilerDB:
    records = [
        BenchmarkRecord(
            op_type="GeLU",
            domain="MPC",
            method="method_mpc_bolt",
            input_shape=(1, 128, 768),
            output_shape=(1, 128, 768),
            local_compute_ms=5.0,
            comm_bytes=1_000_000,
            comm_rounds=4,
            total_latency_ms=10.0,
            metadata={},
        ),
        BenchmarkRecord(
            op_type="GeLU",
            domain="HE",
            method="method_he_nexus",
            input_shape=(1, 128, 768),
            output_shape=(1, 128, 768),
            local_compute_ms=20.0,
            comm_bytes=0,
            comm_rounds=0,
            total_latency_ms=20.0,
            metadata={"he_level_delta": 4, "he_bootstrap_ms": 12.5},
        ),
    ]
    conversions = [
        ConversionRecord(
            from_domain="MPC",
            to_domain="HE",
            method="method_default",
            layout_family="generic",
            tensor_shape=(1, 128, 768),
            local_compute_ms=1.0,
            comm_bytes=500_000,
            comm_rounds=2,
            total_latency_ms=2.0,
            metadata={},
        ),
    ]
    return ProfilerDB(records=records, conversion_records=conversions)


def test_hardware_aware_model_lan_vs_wan_mpc_scaling():
    db = _small_db()
    lan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.lan())
    wan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.wan())

    lan_cost = lan_model.operator_cost(
        "GeLU", "MPC", (1, 128, 768), (1, 128, 768), method="method_mpc_bolt"
    )
    wan_cost = wan_model.operator_cost(
        "GeLU", "MPC", (1, 128, 768), (1, 128, 768), method="method_mpc_bolt"
    )
    # WAN must be strictly slower for an MPC op with non-zero comm.
    assert wan_cost.latency_ms > lan_cost.latency_ms
    # Both use the profile-scaled path.
    assert lan_cost.strategy_used == "network_profile_scaled"
    assert wan_cost.strategy_used == "network_profile_scaled"


def test_hardware_aware_model_he_cost_not_network_scaled():
    """HE cost is unchanged by the network profile (no comm)."""
    db = _small_db()
    lan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.lan())
    wan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.wan())

    lan_cost = lan_model.operator_cost(
        "GeLU", "HE", (1, 128, 768), (1, 128, 768), method="method_he_nexus"
    )
    wan_cost = wan_model.operator_cost(
        "GeLU", "HE", (1, 128, 768), (1, 128, 768), method="method_he_nexus"
    )
    assert lan_cost.latency_ms == wan_cost.latency_ms


def test_hardware_aware_model_conversion_cost_network_scaled():
    db = _small_db()
    lan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.lan())
    wan_model = HardwareAwareCostModel(db=db, network=NetworkProfile.wan())

    lan_cc = lan_model.conversion_cost((1, 128, 768), "MPC", "HE")
    wan_cc = wan_model.conversion_cost((1, 128, 768), "MPC", "HE")
    assert wan_cc.latency_ms > lan_cc.latency_ms


def test_hardware_aware_model_bootstrap_cost_reads_metadata():
    db = _small_db()
    model = HardwareAwareCostModel(db=db)
    bs = model.bootstrap_cost("GeLU", (1, 128, 768), (1, 128, 768))
    assert bs.latency_ms == 12.5
    assert bs.strategy_used == "record_metadata"


def test_hardware_aware_model_level_delta_uses_cost_signature():
    db = _small_db()
    model = HardwareAwareCostModel(db=db)
    # δ_i for LayerNorm HE comes from cost_signature() = 3, not from the
    # static dict (which would report 1).
    d = model.level_delta("LayerNorm", "method_he_nexus", (1, 8, 768), (1, 8, 768))
    assert d == 3


def test_hardware_aware_model_feasibility_surface():
    db = _small_db()
    model = HardwareAwareCostModel(db=db)
    ok, _ = model.feasibility(
        "LayerNorm", "method_he_nexus", (1, 8, 768), (1, 8, 768)
    )
    bad, reason = model.feasibility(
        "LayerNorm", "method_he_nexus", (1, 128, 768), (1, 128, 768)
    )
    assert ok is True
    assert bad is False
    assert "B*S" in reason


# -- StateExpandedCostModel integration ---------------------------------------


def test_state_expanded_level_delta_prefers_cost_signature_over_static_dict():
    """`DEFAULT_LEVEL_DELTAS['GeLU']` is 2, but the method says 4."""
    assert DEFAULT_LEVEL_DELTAS["GeLU"] == 2  # legacy seed
    db = _small_db()
    model = StateExpandedCostModel(db=db)
    node = OperatorNode(
        node_id="n0",
        op_type="GeLU",
        input_shape=(1, 128, 768),
        output_shape=(1, 128, 768),
    )
    # Dynamic method-driven value takes precedence.
    assert model.level_delta(node, "HE", method="method_he_nexus") == 4


def test_state_expanded_level_delta_returns_zero_for_mpc_domain():
    db = _small_db()
    model = StateExpandedCostModel(db=db)
    node = OperatorNode(
        node_id="n0",
        op_type="GeLU",
        input_shape=(1, 128, 768),
        output_shape=(1, 128, 768),
    )
    assert model.level_delta(node, "MPC") == 0


def test_state_expanded_feasible_flag_blocks_infeasible_layernorm():
    db = _small_db()
    model = StateExpandedCostModel(db=db)
    bert_node = OperatorNode(
        node_id="n0",
        op_type="LayerNorm",
        input_shape=(1, 128, 768),
        output_shape=(1, 128, 768),
    )
    demo_node = OperatorNode(
        node_id="n1",
        op_type="LayerNorm",
        input_shape=(1, 8, 768),
        output_shape=(1, 8, 768),
    )
    assert model.feasible(bert_node, "HE") is False
    assert model.feasible(demo_node, "HE") is True
    assert model.feasible(bert_node, "MPC") is True


def test_state_expanded_feasible_flag_default_true_for_unknown_ops():
    db = _small_db()
    model = StateExpandedCostModel(db=db)
    node = OperatorNode(
        node_id="nx",
        op_type="Bogus",
        input_shape=(1,),
        output_shape=(1,),
    )
    assert model.feasible(node, "HE") is True
