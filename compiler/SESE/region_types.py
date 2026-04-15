from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from compiler.min_cut.profiler_db import Domain

from .region_analysis import SESEBlock


@dataclass(frozen=True)
class BoundaryState:
    domain: Domain
    level_bucket: int | None = None

    def label(self) -> str:
        if self.domain == "MPC":
            return "MPC"
        return f"HE:{self.level_bucket}"

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "level_bucket": self.level_bucket,
            "label": self.label(),
        }


@dataclass(frozen=True)
class SummaryAction:
    kind: str
    node_id: str
    op_type: str
    latency_ms: float
    from_state: BoundaryState
    to_state: BoundaryState
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "op_type": self.op_type,
            "latency_ms": self.latency_ms,
            "from_state": self.from_state.as_dict(),
            "to_state": self.to_state.as_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SummaryEntry:
    input_state: BoundaryState
    output_state: BoundaryState
    total_cost_ms: float
    actions: Tuple[SummaryAction, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "input_state": self.input_state.as_dict(),
            "output_state": self.output_state.as_dict(),
            "total_cost_ms": self.total_cost_ms,
            "actions": [action.as_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class BlockSummary:
    block: SESEBlock
    boundary_states: Tuple[BoundaryState, ...]
    supported: bool
    summary_entries: Tuple[SummaryEntry, ...]
    unsupported_reason: str | None = None

    def entries_by_input(self) -> Dict[str, List[SummaryEntry]]:
        grouped: Dict[str, List[SummaryEntry]] = {}
        for entry in self.summary_entries:
            grouped.setdefault(entry.input_state.label(), []).append(entry)
        return grouped

    def as_dict(self) -> dict[str, object]:
        return {
            "block": self.block.as_dict(),
            "boundary_states": [state.as_dict() for state in self.boundary_states],
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "summary_entries": [entry.as_dict() for entry in self.summary_entries],
        }
