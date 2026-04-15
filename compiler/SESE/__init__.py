"""SESE-localized state-expanded compiler prototype plan."""

from .region_analysis import (
    RegionAnalysisResult,
    SESEBlock,
    analyze_sese_regions,
)
from .global_solver import GlobalSolveResult, solve_block_graph_linear
from .global_solver import solve_block_graph_dag
from .region_types import BlockSummary, BoundaryState, SummaryEntry
from .summary_builder import build_block_summaries, enumerate_boundary_states

__all__ = [
    "BlockSummary",
    "BoundaryState",
    "GlobalSolveResult",
    "RegionAnalysisResult",
    "SESEBlock",
    "SummaryEntry",
    "analyze_sese_regions",
    "build_block_summaries",
    "enumerate_boundary_states",
    "solve_block_graph_dag",
    "solve_block_graph_linear",
]
