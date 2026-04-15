from __future__ import annotations

from enum import Enum
from typing import Dict

from .operator_specs import BERT_OPERATOR_SEQUENCE
from .types import BackendType


class CapabilityStatus(str, Enum):
    REAL_INTEGRATED = "real-integrated"
    RESTRICTED_INTEGRATED = "restricted-integrated"
    MOCK = "mock"
    UNSUPPORTED = "unsupported"
    PRE_COMPILE = "pre-compile"         # handled client-side, outside the DAG
    FUSED_INTO_NEXT = "fused-into-next"  # absorbed by a downstream operator


class BackendCapabilityRegistry:
    def __init__(self) -> None:
        self._status: Dict[str, Dict[BackendType, CapabilityStatus]] = {}
        for spec in BERT_OPERATOR_SEQUENCE:
            self._status[spec.name] = {
                BackendType.MPC: CapabilityStatus.MOCK,
                BackendType.HE: CapabilityStatus.MOCK,
                BackendType.HYBRID: CapabilityStatus.MOCK,
            }

    def set_status(self, op_name: str, backend: BackendType, status: CapabilityStatus) -> None:
        if op_name not in self._status:
            self._status[op_name] = {}
        self._status[op_name][backend] = status

    def get_status(self, op_name: str, backend: BackendType) -> CapabilityStatus:
        return self._status.get(op_name, {}).get(backend, CapabilityStatus.UNSUPPORTED)

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        return {
            op: {backend.value: status.value for backend, status in backend_map.items()}
            for op, backend_map in self._status.items()
        }


capability_registry = BackendCapabilityRegistry()
capability_registry.set_status("GeLU", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Softmax", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("LayerNorm", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("FFN_Linear_1", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Attention_QK_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Attention_V_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Residual_Add", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("FFN_Linear_2", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("GeLU", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Softmax", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Residual_Add", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("LayerNorm", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("FFN_Linear_1", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("FFN_Linear_2", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("Attention_QK_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("Attention_V_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)

# Out_Projection is now a real operator with HE and MPC methods that reuse
# the FFN_Linear_1 primitives; it is feasible for BERT-base shapes via the
# MPC chunking wrapper, and is restricted-integrated in HE (same constraints
# as FFN_Linear_1).
capability_registry.set_status("Out_Projection", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Out_Projection", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)

# Embedding and Linear_QKV are not runtime-executed operators. Embedding is
# a plaintext client-side step; Linear_QKV is fused into Attention_QK_MatMul
# via the packed [3,B,S,768] qkv input. Mark them accordingly so the cost
# model and plan builder do not allocate any real latency to them.
capability_registry.set_status("Embedding", BackendType.MPC, CapabilityStatus.PRE_COMPILE)
capability_registry.set_status("Embedding", BackendType.HE, CapabilityStatus.PRE_COMPILE)
capability_registry.set_status("Linear_QKV", BackendType.MPC, CapabilityStatus.FUSED_INTO_NEXT)
capability_registry.set_status("Linear_QKV", BackendType.HE, CapabilityStatus.FUSED_INTO_NEXT)
