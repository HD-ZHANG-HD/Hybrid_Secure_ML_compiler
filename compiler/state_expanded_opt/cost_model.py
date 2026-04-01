from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from compiler.min_cut.cost_model import CostEstimate
from compiler.min_cut.profiler_db import Domain, ProfilerDB
from ir.types import OperatorNode


DEFAULT_LEVEL_DELTAS: Dict[str, int] = {
    "Embedding": 0,
    "Linear_QKV": 1,
    "Attention_QK_MatMul": 1,
    "Softmax": 2,
    "Attention_V_MatMul": 1,
    "Out_Projection": 1,
    "Residual_Add": 0,
    "LayerNorm": 1,
    "GeLU": 2,
    "FFN_Linear_1": 1,
    "FFN_Linear_2": 1,
}


@dataclass(frozen=True)
class OperatorCost:
    latency_ms: float
    strategy_used: str
    method: str | None


class StateExpandedCostModel:
    def __init__(
        self,
        db: ProfilerDB,
        default_level_bucket: int = 3,
        default_bootstrap_ms: float = 6.0,
        fallback_level_deltas: Dict[str, int] | None = None,
    ) -> None:
        self.db = db
        self.default_level_bucket = int(default_level_bucket)
        self.default_bootstrap_ms = float(default_bootstrap_ms)
        self.fallback_level_deltas = dict(DEFAULT_LEVEL_DELTAS)
        if fallback_level_deltas:
            self.fallback_level_deltas.update(fallback_level_deltas)

    def operator_cost(self, node: OperatorNode, domain: Domain, method: str | None = None) -> OperatorCost:
        estimate = self._estimate_node_cost(node, domain, method=method)
        return OperatorCost(latency_ms=estimate.latency_ms, strategy_used=estimate.strategy_used, method=method)

    def conversion_cost(self, tensor_shape: tuple[int, ...], from_domain: Domain, to_domain: Domain) -> CostEstimate:
        from compiler.min_cut.cost_model import CostModel

        return CostModel(self.db, default_strategy="auto").estimate_conversion_cost(
            tensor_shape=tensor_shape,
            from_domain=from_domain,
            to_domain=to_domain,
        )

    def bootstrap_cost(self, node: OperatorNode) -> CostEstimate:
        records = self.db.get_operator_records(node.op_type, "HE")
        for rec in records:
            if rec.input_shape == node.input_shape and rec.output_shape == node.output_shape:
                raw = rec.metadata.get("he_bootstrap_ms")
                if raw is not None:
                    return CostEstimate(latency_ms=float(raw), strategy_used="record_metadata")
        for rec in records:
            raw = rec.metadata.get("he_bootstrap_ms")
            if raw is not None:
                return CostEstimate(latency_ms=float(raw), strategy_used="record_metadata_fallback")
        return CostEstimate(latency_ms=self.default_bootstrap_ms, strategy_used="default_bootstrap")

    def level_delta(self, node: OperatorNode, domain: Domain, method: str | None = None) -> int:
        if domain != "HE":
            return 0
        record = self.db.find_exact_operator_record(node.op_type, "HE", node.input_shape, node.output_shape, method=method)
        if record is not None and "he_level_delta" in record.metadata:
            return int(record.metadata["he_level_delta"])
        for rec in self.db.get_operator_records(node.op_type, "HE", method=method):
            if "he_level_delta" in rec.metadata:
                return int(rec.metadata["he_level_delta"])
        if node.op_type not in self.fallback_level_deltas:
            raise ValueError(f"Missing HE level delta for op={node.op_type}")
        return int(self.fallback_level_deltas[node.op_type])

    def max_level(self) -> int:
        return self.default_level_bucket

    def _estimate_node_cost(self, node: OperatorNode, domain: Domain, method: str | None) -> CostEstimate:
        from compiler.min_cut.cost_model import CostModel

        return CostModel(self.db, default_strategy="auto").estimate_node_cost(
            op_type=node.op_type,
            domain=domain,
            input_shape=node.input_shape,
            output_shape=node.output_shape,
            method=method,
        )
