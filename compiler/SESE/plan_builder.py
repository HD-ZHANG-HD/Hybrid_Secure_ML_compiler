"""Lower a `GlobalSolveResult` to a linear compiler-plan dict (Section E).

The output format matches the shape produced by
`compiler/state_expanded_opt/plan_builder.py:build_execution_plan` so the
existing runtime adapter infrastructure can be reused:

```
{
    "graph_id": str,
    "strategy": "sese_global",
    "total_cost_ms": float,
    "start_state": dict | None,
    "goal_state": dict | None,
    "block_order": list[str],
    "block_decisions": list[dict],
    "steps": list[dict],  # flattened action stream
}
```

Each element of `steps` is one of:

- `{"kind": "execute_he" | "execute_mpc", "node_id": ..., "op_type": ..., ...}`
- `{"kind": "conversion", "from_node": ..., "to_node": ..., "from_domain": ..., "to_domain": ..., "tensor_shape": ...}`
- `{"kind": "bootstrap", "node_id": ..., "op_type": ..., ...}`

SESE action kinds are normalized onto those three so the runtime adapter
only has to know about a single action vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .global_solver import GlobalSolveResult


_EXECUTE_KINDS = {"execute_he", "execute_mpc", "merge_execute"}
_CONVERSION_KINDS = {"conversion", "convert_he_to_mpc", "convert_mpc_to_he"}
_BOOTSTRAP_KINDS = {"bootstrap"}


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(action.get("kind", ""))
    if kind in _EXECUTE_KINDS:
        # `merge_execute` is an SESE bookkeeping name; the runtime
        # interprets it exactly like a regular execute_*.
        from_state = action.get("from_state") or {}
        to_state = action.get("to_state") or {}
        domain = str(from_state.get("domain") or to_state.get("domain") or "")
        kind_out = "execute_he" if domain == "HE" else "execute_mpc"
        return {
            "kind": kind_out,
            "node_id": action.get("node_id"),
            "op_type": action.get("op_type"),
            "estimated_latency_ms": float(action.get("latency_ms", 0.0)),
            "reason": action.get("reason", ""),
            "from_state": from_state,
            "to_state": to_state,
            "domain": domain,
        }
    if kind in _CONVERSION_KINDS:
        from_state = action.get("from_state") or {}
        to_state = action.get("to_state") or {}
        edge = action.get("edge") or {}
        return {
            "kind": "conversion",
            "from_node": edge.get("src"),
            "to_node": edge.get("dst"),
            "from_domain": from_state.get("domain"),
            "to_domain": to_state.get("domain"),
            "tensor_shape": list(edge.get("tensor_shape") or []),
            "estimated_latency_ms": float(action.get("latency_ms", 0.0)),
            "reason": action.get("reason", ""),
        }
    if kind in _BOOTSTRAP_KINDS:
        return {
            "kind": "bootstrap",
            "node_id": action.get("node_id"),
            "op_type": action.get("op_type"),
            "estimated_latency_ms": float(action.get("latency_ms", 0.0)),
            "reason": action.get("reason", ""),
        }
    raise ValueError(f"Unsupported SESE action kind: {kind}")


def build_execution_plan_from_sese(result: GlobalSolveResult) -> Dict[str, Any]:
    """Flatten a `GlobalSolveResult` into the runtime adapter's plan dict."""
    if not result.supported:
        raise ValueError(
            f"SESE global_solver reported unsupported graph: {result.unsupported_reason!r}"
        )

    steps: List[Dict[str, Any]] = []
    for decision in result.block_decisions:
        for action in decision.actions:
            steps.append(_normalize_action(dict(action)))

    return {
        "graph_id": result.graph_id,
        "strategy": "sese_global",
        "total_cost_ms": float(result.total_cost_ms or 0.0),
        "start_state": None if result.start_state is None else result.start_state.as_dict(),
        "goal_state": None if result.goal_state is None else result.goal_state.as_dict(),
        "block_order": list(result.block_order),
        "block_decisions": [decision.as_dict() for decision in result.block_decisions],
        "steps": steps,
    }


__all__ = ["build_execution_plan_from_sese"]
