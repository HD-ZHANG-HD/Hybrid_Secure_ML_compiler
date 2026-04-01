from __future__ import annotations

from typing import Dict, List

from .solver import SolverResult


def build_execution_plan(solver_result: SolverResult) -> Dict[str, object]:
    steps: List[Dict[str, object]] = []
    for action in solver_result.actions:
        step = {
            "step_id": action.step_id,
            "kind": action.kind if action.kind != "bootstrap" else "bootstrap",
            "node_id": action.node_id,
            "op_type": action.op_type,
            "estimated_latency_ms": action.estimated_latency_ms,
            "reason": action.reason,
            "from_state": action.from_state,
            "to_state": action.to_state,
            "level_before": action.level_before,
            "level_after": action.level_after,
        }
        if action.kind == "conversion":
            step.update(
                {
                    "from_node": action.from_node,
                    "to_node": action.to_node,
                    "from_domain": action.from_domain,
                    "to_domain": action.to_domain,
                    "tensor_shape": list(action.tensor_shape or ()),
                }
            )
        elif action.kind in {"execute_he", "execute_mpc"}:
            step["domain"] = "HE" if action.kind == "execute_he" else "MPC"
            step["input_shape"] = None
        steps.append(step)

    return {
        "graph_id": solver_result.graph_id,
        "strategy": solver_result.strategy,
        "best_path": solver_result.best_path,
        "start_state": solver_result.start_state,
        "goal_state": solver_result.goal_state,
        "state_trace": solver_result.state_trace,
        "decision_trace": solver_result.decision_trace,
        "cost_breakdown": solver_result.cost_breakdown,
        "final_assignment": solver_result.final_assignment,
        "stage_summaries": solver_result.stage_summaries,
        "steps": steps,
    }
