from __future__ import annotations

import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from compiler.SESE.region_analysis import analyze_sese_regions
from compiler.SESE.global_solver import solve_block_graph_dag, solve_block_graph_linear
from compiler.SESE.summary_builder import build_block_summaries
from compiler.min_cut.profiler_db import ProfilerDB
from compiler.state_expanded_opt.cost_model import StateExpandedCostModel
from compiler.state_expanded_opt.graph_model import load_graph_json


def run_demo() -> None:
    here = Path(__file__).resolve().parent
    profiler_db = ProfilerDB.from_json(here.parent / "state_expanded_opt" / "test" / "profiler_with_budget.json")
    cost_model = StateExpandedCostModel(profiler_db)
    cases = {
        "chain_budget": load_graph_json(here.parent / "state_expanded_opt" / "test" / "graph_chain_budget.json"),
        "residual_stage_local": load_graph_json(here.parent / "state_expanded_opt" / "test" / "graph_residual_stage_local.json"),
        "fork_supported": load_graph_json(here / "test" / "graph_fork_supported.json"),
    }
    for name, graph in cases.items():
        result = analyze_sese_regions(graph)
        summaries = build_block_summaries(graph, result, cost_model)
        global_result = solve_block_graph_linear(result, summaries)
        dag_result = solve_block_graph_dag(result, summaries, graph, cost_model)
        print(f"\n===== {name} =====")
        payload = result.as_dict()
        payload["block_summaries"] = {block_id: summary.as_dict() for block_id, summary in summaries.items()}
        payload["global_solution"] = global_result.as_dict()
        payload["global_solution_dag"] = dag_result.as_dict()
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run_demo()
