from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from compiler.min_cut.profiler_db import Domain


ActionKind = Literal["execute_he", "execute_mpc", "convert_he_to_mpc", "convert_mpc_to_he", "bootstrap_he"]


@dataclass(frozen=True)
class State:
    position: int
    domain: Domain
    level_bucket: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "domain": self.domain,
            "level_bucket": self.level_bucket,
        }


@dataclass(frozen=True)
class ActionRecord:
    step_id: str
    kind: ActionKind | str
    node_id: str
    op_type: str
    estimated_latency_ms: float
    reason: str
    from_state: dict[str, object]
    to_state: dict[str, object]
    from_node: str | None = None
    to_node: str | None = None
    from_domain: Domain | None = None
    to_domain: Domain | None = None
    tensor_shape: tuple[int, ...] | None = None
    level_before: int | None = None
    level_after: int | None = None


@dataclass(frozen=True)
class NodeExecution:
    node_id: str
    domain: Domain
    level_before: int | None
    level_after: int | None
    incremental_cost_ms: float
