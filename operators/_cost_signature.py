"""Per-method cost signatures consumed by the compiler cost model.

The paper's state-expanded solver needs three kinds of signal from every
operator method:

1. HE multiplicative-depth consumption `he_level_delta` (δ_i in paper §3.2.2).
2. A hard feasibility gate per-shape (`feasible`) so HE transitions get
   pruned for shapes the NEXUS primitive cannot run — e.g. LayerNorm when
   B*S > 16.
3. Hardware-profile-independent metadata used when ranking transitions
   (domain, op type, input/output shapes, bootstrap support).

Every kept method module exposes a top-level ``cost_signature(input_shape,
output_shape, ctx=None) -> OperatorCostSignature`` helper. Methods that
never run HE (plain MPC wrappers) return ``he_level_delta = 0`` and
``he_min_level_required = 0``; methods that run HE without bootstrap
support set ``he_bootstrap_supported = False`` so the solver falls back to
HE->MPC->HE whenever the level budget is exhausted.

This module is intentionally dependency-free (only stdlib + typing) so the
compiler cost model can import it without pulling in NEXUS/SCI adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class OperatorCostSignature:
    op_type: str
    domain: str
    input_shape: Shape
    output_shape: Shape
    he_level_delta: int = 0
    he_min_level_required: int = 0
    he_bootstrap_supported: bool = False
    feasible: bool = True
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_feasible(self, feasible: bool, reason: str = "") -> "OperatorCostSignature":
        return OperatorCostSignature(
            op_type=self.op_type,
            domain=self.domain,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            he_level_delta=self.he_level_delta,
            he_min_level_required=self.he_min_level_required,
            he_bootstrap_supported=self.he_bootstrap_supported,
            feasible=feasible,
            notes=(reason or self.notes),
            extras=dict(self.extras),
        )


def _as_shape(shape: Sequence[int] | None) -> Shape:
    if shape is None:
        return ()
    return tuple(int(d) for d in shape)


def he_signature(
    op_type: str,
    *,
    input_shape: Sequence[int] | None,
    output_shape: Sequence[int] | None,
    level_delta: int,
    bootstrap_supported: bool = False,
    feasible: bool = True,
    notes: str = "",
    extras: Dict[str, Any] | None = None,
) -> OperatorCostSignature:
    return OperatorCostSignature(
        op_type=op_type,
        domain="HE",
        input_shape=_as_shape(input_shape),
        output_shape=_as_shape(output_shape),
        he_level_delta=int(level_delta),
        he_min_level_required=int(level_delta),
        he_bootstrap_supported=bool(bootstrap_supported),
        feasible=bool(feasible),
        notes=notes,
        extras=dict(extras or {}),
    )


def mpc_signature(
    op_type: str,
    *,
    input_shape: Sequence[int] | None,
    output_shape: Sequence[int] | None,
    feasible: bool = True,
    notes: str = "",
    extras: Dict[str, Any] | None = None,
) -> OperatorCostSignature:
    return OperatorCostSignature(
        op_type=op_type,
        domain="MPC",
        input_shape=_as_shape(input_shape),
        output_shape=_as_shape(output_shape),
        he_level_delta=0,
        he_min_level_required=0,
        he_bootstrap_supported=False,
        feasible=bool(feasible),
        notes=notes,
        extras=dict(extras or {}),
    )


class BootstrapUnsupportedError(RuntimeError):
    """Raised when a HE method cannot bootstrap in place.

    The paper's solver catches this and falls back to the `HE -> MPC -> HE`
    detour described in §3.2.2. Raising it (vs. silently returning the
    input) keeps the fallback path explicit.
    """


def bs_product(shape: Sequence[int]) -> int:
    """Return `B*S` for a `[B,S,...]` tensor, or 1 if shape is rank<2."""
    shape = _as_shape(shape)
    if len(shape) < 2:
        return 1
    return int(shape[0]) * int(shape[1])
