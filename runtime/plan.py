from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from .types import BackendType


@dataclass(frozen=True)
class OperatorStep:
    type: Literal["operator"] = "operator"
    op_type: str = ""
    method: str = "method_default"
    backend: BackendType = BackendType.MPC
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversionStep:
    type: Literal["conversion"] = "conversion"
    from_domain: BackendType = BackendType.MPC
    to_domain: BackendType = BackendType.HE
    tensor: str = ""
    method: str = "method_default"
    output_tensor: str | None = None


@dataclass(frozen=True)
class BootstrapStep:
    type: Literal["bootstrap"] = "bootstrap"
    backend: BackendType = BackendType.HE
    tensor: str = ""
    method: str = "method_default"
    output_tensor: str | None = None


ExecutionStep = OperatorStep | ConversionStep | BootstrapStep


@dataclass(frozen=True)
class ExecutionPlan:
    steps: List[ExecutionStep]
