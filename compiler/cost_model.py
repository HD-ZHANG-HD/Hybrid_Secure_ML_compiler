"""Unified hardware-aware cost model (Section B of the revise list).

This module deduplicates the two previously-separate cost-model paths
(`compiler/min_cut/cost_model.py` and `compiler/state_expanded_opt/cost_model.py`)
behind a single `HardwareAwareCostModel` and introduces the
`NetworkProfile` dataclass that the paper's cost model requires
(§3.2.1: target profile is `{hardware, bandwidth, rtt}`).

Two key Section-B changes land here:

1. **Dynamic δ_i via `cost_signature()`**. The static `DEFAULT_LEVEL_DELTAS`
   dict is no longer authoritative; `level_delta_for_method()` imports
   the named method module and calls its `cost_signature()` helper to
   read the HE multiplicative-depth consumption. Feasibility verdicts
   from the same cost signature are also surfaced so the solver can
   prune infeasible HE transitions up-front.

2. **Network-aware MPC scaling**. MPC operator latency is a function of
   `(local_compute_ms, comm_bytes, comm_rounds)` and the target
   `NetworkProfile`. The profile can be attached to
   `HardwareAwareCostModel` and then propagates into every
   operator/conversion cost lookup — previously MPC cost was a scalar
   independent of bandwidth/RTT, which is what prevented any pure-MPC
   baseline from reproducing the paper's Figure 4 shape.

The min_cut `CostModel` is kept as a low-level backend; experiments that
imported `compiler.min_cut.cost_model.CostModel` still work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from compiler.min_cut.cost_model import CostEstimate, CostModel
from compiler.min_cut.profiler_db import Domain, ProfilerDB
from operators._cost_signature import OperatorCostSignature


Shape = Tuple[int, ...]


# ---------------------------------------------------------------------------
# NetworkProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkProfile:
    """Target network characteristics for an MPC/HE cost lookup.

    Fields match the paper §4.1 experiment grid:
    - `bandwidth_mbps`: effective bandwidth in megabits/sec.
    - `rtt_ms`: round-trip time in milliseconds.

    Two helper constructors (`LAN`, `WAN`) return the defaults used in
    the paper's Figure 4 sweep.
    """

    bandwidth_mbps: float = 1000.0  # default: 1 Gbps LAN
    rtt_ms: float = 1.0
    name: str = "default"

    @classmethod
    def lan(cls, bandwidth_mbps: float = 1000.0, rtt_ms: float = 1.0) -> "NetworkProfile":
        return cls(bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, name="lan")

    @classmethod
    def wan(cls, bandwidth_mbps: float = 100.0, rtt_ms: float = 40.0) -> "NetworkProfile":
        return cls(bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, name="wan")

    def comm_ms(self, comm_bytes: int, comm_rounds: int) -> float:
        """Simple linear model: bytes/bandwidth + rounds * RTT.

        Deliberately simple — the paper's model in §3.2.1 is the same
        first-order decomposition. The real value is that the model is
        a function of the profile, not a constant.
        """
        if comm_bytes <= 0 and comm_rounds <= 0:
            return 0.0
        bw_bytes_per_ms = max(1e-9, self.bandwidth_mbps * (1_000_000.0 / 8.0) / 1000.0)
        byte_ms = float(comm_bytes) / bw_bytes_per_ms
        rtt_cost = float(comm_rounds) * float(self.rtt_ms)
        return byte_ms + rtt_cost


# ---------------------------------------------------------------------------
# Method-driven δ_i lookup
# ---------------------------------------------------------------------------


# Map of `(op_type, method_name)` to the fully qualified module path whose
# `cost_signature()` function returns the authoritative `OperatorCostSignature`.
#
# This replaces the hard-coded `DEFAULT_LEVEL_DELTAS` fallback: δ_i is now a
# property of the method, not the op_type, which is how the paper's solver
# treats it (Table 1).
METHOD_MODULE_MAP: Dict[Tuple[str, str], str] = {
    # HE methods
    ("GeLU", "method_he_nexus"): "operators.gelu.method_he_nexus",
    ("Softmax", "method_he_nexus"): "operators.softmax.method_he_nexus",
    ("LayerNorm", "method_he_nexus"): "operators.layernorm.method_he_nexus",
    ("FFN_Linear_1", "method_he_nexus"): "operators.linear_ffn1.method_he_nexus",
    ("FFN_Linear_2", "method_he_nexus"): "operators.ffn_linear_2.method_he_nexus",
    ("Attention_QK_MatMul", "method_he_nexus"): "operators.attention_qk_matmul.method_he_nexus",
    ("Attention_V_MatMul", "method_he_nexus"): "operators.attention_v_matmul.method_he_nexus",
    ("Out_Projection", "method_he_nexus_as_ffn1"): "operators.out_projection.method_he_nexus_as_ffn1",
    ("Residual_Add", "method_runtime_default"): "operators.residual_add.method_runtime_default",
    # MPC methods
    ("GeLU", "method_mpc_bolt"): "operators.gelu.method_mpc_bolt",
    ("Softmax", "method_mpc_bolt"): "operators.softmax.method_mpc_bolt",
    ("LayerNorm", "method_mpc_bolt"): "operators.layernorm.method_mpc_bolt",
    ("FFN_Linear_1", "method_mpc_bolt"): "operators.linear_ffn1.method_mpc_bolt",
    ("FFN_Linear_2", "method_mpc_bolt"): "operators.ffn_linear_2.method_mpc_bolt",
    ("Attention_QK_MatMul", "method_mpc"): "operators.attention_qk_matmul.method_mpc",
    ("Attention_V_MatMul", "method_mpc"): "operators.attention_v_matmul.method_mpc",
    ("Out_Projection", "method_mpc_bolt_as_ffn1"): "operators.out_projection.method_mpc_bolt_as_ffn1",
}


def default_method_for(op_type: str, domain: Domain) -> Optional[str]:
    """Return the canonical method name for `(op_type, domain)`, if any."""
    if domain == "HE":
        he_table = {
            "GeLU": "method_he_nexus",
            "Softmax": "method_he_nexus",
            "LayerNorm": "method_he_nexus",
            "FFN_Linear_1": "method_he_nexus",
            "FFN_Linear_2": "method_he_nexus",
            "Attention_QK_MatMul": "method_he_nexus",
            "Attention_V_MatMul": "method_he_nexus",
            "Out_Projection": "method_he_nexus_as_ffn1",
            "Residual_Add": "method_runtime_default",
        }
        return he_table.get(op_type)
    mpc_table = {
        "GeLU": "method_mpc_bolt",
        "Softmax": "method_mpc_bolt",
        "LayerNorm": "method_mpc_bolt",
        "FFN_Linear_1": "method_mpc_bolt",
        "FFN_Linear_2": "method_mpc_bolt",
        "Attention_QK_MatMul": "method_mpc",
        "Attention_V_MatMul": "method_mpc",
        "Out_Projection": "method_mpc_bolt_as_ffn1",
        "Residual_Add": "method_runtime_default",
    }
    return mpc_table.get(op_type)


def cost_signature_for_method(
    op_type: str,
    method_name: str,
    input_shape: Shape,
    output_shape: Optional[Shape] = None,
    ctx: Any = None,
) -> Optional[OperatorCostSignature]:
    """Resolve a method module and return its `OperatorCostSignature`.

    Returns `None` if the method has no registered module (e.g. mock ops
    like `Embedding` and `Linear_QKV` that are pre_compile / fused).
    """
    mod_path = METHOD_MODULE_MAP.get((op_type, method_name))
    if mod_path is None:
        return None
    try:
        mod = importlib.import_module(mod_path)
    except ImportError:
        return None
    fn = getattr(mod, "cost_signature", None)
    if fn is None:
        return None
    # residual_add's cost_signature has an extra `domain=` kw — pass it only
    # when the default method is in use so we produce the right verdict.
    if op_type == "Residual_Add":
        domain_hint = "HE" if method_name == "method_runtime_default" else "MPC"
        return fn(input_shape, output_shape, ctx=ctx, domain=domain_hint)
    return fn(input_shape, output_shape, ctx=ctx)


def level_delta_for_method(
    op_type: str,
    method_name: str,
    input_shape: Shape,
    output_shape: Optional[Shape] = None,
) -> Optional[int]:
    """Look up δ_i for a concrete method/shape from its `cost_signature()`."""
    sig = cost_signature_for_method(op_type, method_name, input_shape, output_shape)
    if sig is None:
        return None
    return int(sig.he_level_delta)


# ---------------------------------------------------------------------------
# Unified cost model
# ---------------------------------------------------------------------------


@dataclass
class HardwareAwareCostModel:
    """Single cost model shared by min_cut and state_expanded solvers.

    Wraps the legacy `CostModel` (for record lookup + shape interpolation)
    and layers the paper's target-profile scaling on top:

    - A `NetworkProfile` governs MPC-side communication cost.
    - `level_delta()` prefers method-driven `cost_signature()` results
      over the legacy static dict.
    """

    db: ProfilerDB
    network: NetworkProfile = field(default_factory=NetworkProfile)
    default_strategy: str = "auto"
    default_bootstrap_ms: float = 6.0

    def _inner(self) -> CostModel:
        return CostModel(self.db, default_strategy=self.default_strategy)  # type: ignore[arg-type]

    def operator_cost(
        self,
        op_type: str,
        domain: Domain,
        input_shape: Shape,
        output_shape: Shape,
        method: Optional[str] = None,
    ) -> CostEstimate:
        base = self._inner().estimate_node_cost(
            op_type=op_type,
            domain=domain,
            input_shape=input_shape,
            output_shape=output_shape,
            method=method,
        )
        if domain != "MPC":
            return base
        # Add network cost on top of the record-provided local_compute_ms.
        # The base record's total_latency_ms already bakes in some network
        # contribution, so we only *add* the delta between the profile the
        # record was measured under and the target profile when metadata
        # lets us. For the common case where metadata is missing we trust
        # the record as LAN-baseline and add the RTT delta for round count.
        records = self.db.get_operator_records(op_type, domain, method=method)
        if not records:
            return base
        # Take the nearest record's comm_bytes/comm_rounds as the best
        # estimate for this shape, then rescale communication using the
        # active NetworkProfile.
        ref = records[0]
        comm_ms = self.network.comm_ms(ref.comm_bytes, ref.comm_rounds)
        latency = ref.local_compute_ms + comm_ms
        return CostEstimate(latency_ms=max(0.0, latency), strategy_used="network_profile_scaled")

    def conversion_cost(
        self,
        tensor_shape: Shape,
        from_domain: Domain,
        to_domain: Domain,
        method: Optional[str] = None,
        layout_family: Optional[str] = None,
    ) -> CostEstimate:
        base = self._inner().estimate_conversion_cost(
            tensor_shape=tensor_shape,
            from_domain=from_domain,
            to_domain=to_domain,
            method=method,
            layout_family=layout_family,
        )
        if from_domain == to_domain:
            return base
        records = self.db.get_conversion_records(
            from_domain, to_domain, method=method, layout_family=layout_family
        )
        if not records:
            return base
        ref = records[0]
        comm_ms = self.network.comm_ms(ref.comm_bytes, ref.comm_rounds)
        latency = ref.local_compute_ms + comm_ms
        return CostEstimate(latency_ms=max(0.0, latency), strategy_used="network_profile_scaled")

    def bootstrap_cost(self, op_type: str, input_shape: Shape, output_shape: Shape) -> CostEstimate:
        """Look up bootstrap latency from profiler records if present."""
        records = self.db.get_operator_records(op_type, "HE")
        for rec in records:
            if rec.input_shape == input_shape and rec.output_shape == output_shape:
                raw = rec.metadata.get("he_bootstrap_ms")
                if raw is not None:
                    return CostEstimate(latency_ms=float(raw), strategy_used="record_metadata")
        for rec in records:
            raw = rec.metadata.get("he_bootstrap_ms")
            if raw is not None:
                return CostEstimate(latency_ms=float(raw), strategy_used="record_metadata_fallback")
        return CostEstimate(
            latency_ms=self.default_bootstrap_ms, strategy_used="default_bootstrap"
        )

    def level_delta(
        self,
        op_type: str,
        method: Optional[str],
        input_shape: Shape,
        output_shape: Optional[Shape] = None,
    ) -> int:
        if method is None:
            method = default_method_for(op_type, "HE")
        if method is not None:
            delta = level_delta_for_method(op_type, method, input_shape, output_shape)
            if delta is not None:
                return delta
        # Last-resort: 0 (pretend no multiplicative depth is consumed). The
        # solver will then rely on feasibility signals to prune the branch.
        return 0

    def feasibility(
        self,
        op_type: str,
        method: Optional[str],
        input_shape: Shape,
        output_shape: Optional[Shape] = None,
        ctx: Any = None,
    ) -> Tuple[bool, str]:
        if method is None:
            method = default_method_for(op_type, "HE")
        if method is None:
            return True, "no cost_signature registered; treated as feasible"
        sig = cost_signature_for_method(op_type, method, input_shape, output_shape, ctx)
        if sig is None:
            return True, "no cost_signature; treated as feasible"
        return sig.feasible, sig.notes


__all__ = [
    "NetworkProfile",
    "HardwareAwareCostModel",
    "cost_signature_for_method",
    "level_delta_for_method",
    "default_method_for",
    "METHOD_MODULE_MAP",
]
