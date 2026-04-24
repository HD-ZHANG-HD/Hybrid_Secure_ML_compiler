"""Section C: torch.fx frontend + unified OperatorGraph type.

The frontend takes a `torch.nn.Module`, runs `torch.fx.symbolic_trace`,
propagates tensor shapes, and collapses the fx graph onto the
compiler's canonical 9-operator vocabulary. These tests verify the
frontend against *stub* torch modules — the real
MPCFormer BertModel is not imported because it pulls in
HuggingFace/transformers weights and is environment-sensitive.

Coverage:
- Module classification (LayerNorm / GELU / Softmax / Linear variants)
- Linear disambiguation (FFN_Linear_1 via GeLU consumer, FFN_Linear_2
  via GeLU producer, Out_Projection via shape-ratio fallback)
- Shape propagation populates `input_shape` / `output_shape`
- `Residual_Add` emitted for `operator.add`
- Raw matmul routing: QK vs V MatMul based on softmax position
- `graph_executed_subset` drops `Unknown` nodes and rewires edges
- `trace_module_to_graph` raises `ValueError` when `fail_on_unknown=True`
- Frontend graceful-skip when torch is missing (importorskip)

The Section C graph-type unification is also verified: `min_cut`'s
`OperatorNode` / `OperatorGraph` / `DataEdge` are now the same objects
as `ir.types`'.
"""

from __future__ import annotations

import pytest

# Skip the whole module when torch is not available in the environment.
torch = pytest.importorskip("torch")
nn = torch.nn

from ir.frontend_torchfx import (  # noqa: E402
    TorchFxFrontendUnavailable,
    TorchFxTraceConfig,
    graph_executed_subset,
    trace_module_to_graph,
)
from ir.types import DataEdge, OperatorGraph, OperatorNode  # noqa: E402


# -- helpers ------------------------------------------------------------------


def _find(graph: OperatorGraph, node_id: str) -> OperatorNode:
    for node in graph.nodes:
        if node.node_id == node_id:
            return node
    raise AssertionError(f"node {node_id!r} not in graph")


def _op_types(graph: OperatorGraph) -> list[str]:
    return [node.op_type for node in graph.nodes]


# -- graph type unification (Section C prerequisite) ------------------------


def test_min_cut_graph_types_are_aliases_of_ir_types():
    """After Section C unification these are literally the same classes."""
    from compiler.min_cut.domain_assignment import (
        DataEdge as MinCutEdge,
        OperatorGraph as MinCutGraph,
        OperatorNode as MinCutNode,
    )

    assert MinCutNode is OperatorNode
    assert MinCutGraph is OperatorGraph
    assert MinCutEdge is DataEdge


def test_min_cut_positional_constructor_still_works():
    """Existing positional-arg callers must not break."""
    from compiler.min_cut.domain_assignment import OperatorNode as MCNode

    node = MCNode("n0", "GeLU", (1, 4, 8), (1, 4, 8))
    assert node.node_id == "n0"
    assert node.op_type == "GeLU"
    # New `attributes` field defaults to an empty dict.
    assert node.attributes == {}


# -- classification -----------------------------------------------------------


class _MiniFFN(nn.Module):
    def __init__(self, hidden: int = 8, mult: int = 4) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.ffn1 = nn.Linear(hidden, hidden * mult)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(hidden * mult, hidden)

    def forward(self, x):
        h = self.ln(x)
        h = self.ffn1(h)
        h = self.gelu(h)
        h = self.ffn2(h)
        return h + x


def test_trace_mini_ffn_emits_canonical_op_types():
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    op_types = _op_types(graph)
    assert "LayerNorm" in op_types
    assert "FFN_Linear_1" in op_types
    assert "GeLU" in op_types
    assert "FFN_Linear_2" in op_types
    assert "Residual_Add" in op_types


def test_trace_mini_ffn_populates_shapes():
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    assert _find(graph, "ln").input_shape == (1, 4, 8)
    assert _find(graph, "ln").output_shape == (1, 4, 8)
    assert _find(graph, "ffn1").input_shape == (1, 4, 8)
    assert _find(graph, "ffn1").output_shape == (1, 4, 32)
    assert _find(graph, "ffn2").input_shape == (1, 4, 32)
    assert _find(graph, "ffn2").output_shape == (1, 4, 8)


def test_trace_mini_ffn_builds_forward_edges():
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    edges = {(e.src, e.dst) for e in graph.edges}
    assert ("ln", "ffn1") in edges
    assert ("ffn1", "gelu") in edges
    assert ("gelu", "ffn2") in edges
    assert ("ffn2", "add") in edges


def test_linear_disambiguation_by_gelu_neighbour():
    """ffn1 is upstream of GeLU → FFN_Linear_1; ffn2 is downstream → FFN_Linear_2."""
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    assert _find(graph, "ffn1").op_type == "FFN_Linear_1"
    assert _find(graph, "ffn2").op_type == "FFN_Linear_2"


# -- attention routing --------------------------------------------------------


class _MiniAttn(nn.Module):
    """Synthetic attention: matmul -> softmax -> matmul -> linear -> residual."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        q = self.proj(x)
        scores = torch.matmul(q, q.transpose(-1, -2))
        probs = self.softmax(scores)
        context = torch.matmul(probs, q)
        proj = self.out_proj(context)
        residual = proj + x
        return self.ln(residual)


def test_trace_mini_attn_routes_matmuls_to_qk_and_v():
    graph = trace_module_to_graph(_MiniAttn(), torch.zeros(1, 4, 8))
    # The two matmuls should route differently based on softmax position
    matmul = _find(graph, "matmul")
    matmul_1 = _find(graph, "matmul_1")
    assert matmul.op_type == "Attention_QK_MatMul"
    assert matmul_1.op_type == "Attention_V_MatMul"


def test_trace_mini_attn_emits_softmax_layernorm_residual():
    graph = trace_module_to_graph(_MiniAttn(), torch.zeros(1, 4, 8))
    op_types = _op_types(graph)
    assert "Softmax" in op_types
    assert "LayerNorm" in op_types
    assert "Residual_Add" in op_types


def test_graph_executed_subset_drops_unknown_transpose():
    """The `.transpose(-1,-2)` call becomes `Unknown`; pruning should
    remove it and rewire the edge qkv -> matmul directly."""
    graph = trace_module_to_graph(_MiniAttn(), torch.zeros(1, 4, 8))
    assert "Unknown" in _op_types(graph)  # transpose or similar
    pruned = graph_executed_subset(graph)
    assert "Unknown" not in _op_types(pruned)
    # The matmul still has an edge coming from the qkv producer after rewire.
    edges = {(e.src, e.dst) for e in pruned.edges}
    # producer of matmul in pruned graph should resolve to an emitted node
    matmul_predecessors = {src for src, dst in edges if dst == "matmul"}
    assert matmul_predecessors, "matmul has no predecessors after pruning"


# -- error handling + config --------------------------------------------------


class _WithReshape(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        return self.ln(x).reshape(1, 4, 8)


def test_fail_on_unknown_raises():
    with pytest.raises(ValueError, match="could not classify"):
        trace_module_to_graph(
            _WithReshape(),
            torch.zeros(1, 4, 8),
            cfg=TorchFxTraceConfig(fail_on_unknown=True),
        )


def test_default_config_emits_unknown_instead_of_raising():
    graph = trace_module_to_graph(_WithReshape(), torch.zeros(1, 4, 8))
    # LayerNorm recognised, reshape unknown
    assert "LayerNorm" in _op_types(graph)
    assert "Unknown" in _op_types(graph)


def test_graph_id_respected():
    cfg = TorchFxTraceConfig(graph_id="mini_test")
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8), cfg=cfg)
    assert graph.graph_id == "mini_test"


# -- shape-ratio fallback for Linear disambiguation --------------------------


class _StandaloneLinear(nn.Module):
    """Linear with no GeLU / Attention context → exercise shape-ratio fallback."""

    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_f)
        self.lin = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.lin(self.ln(x))


def test_standalone_linear_square_becomes_out_projection():
    graph = trace_module_to_graph(_StandaloneLinear(8, 8), torch.zeros(1, 4, 8))
    assert _find(graph, "lin").op_type == "Out_Projection"


def test_standalone_linear_expand_becomes_ffn_linear_1():
    graph = trace_module_to_graph(_StandaloneLinear(8, 32), torch.zeros(1, 4, 8))
    assert _find(graph, "lin").op_type == "FFN_Linear_1"


def test_standalone_linear_contract_becomes_ffn_linear_2():
    graph = trace_module_to_graph(_StandaloneLinear(32, 8), torch.zeros(1, 4, 32))
    assert _find(graph, "lin").op_type == "FFN_Linear_2"


# -- frontend is shape-propagated end-to-end ---------------------------------


def test_every_emitted_node_has_a_shape():
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    for node in graph.nodes:
        assert node.input_shape != (), node.node_id
        assert node.output_shape != (), node.node_id


def test_source_fx_node_is_recorded_in_attributes():
    graph = trace_module_to_graph(_MiniFFN(), torch.zeros(1, 4, 8))
    for node in graph.nodes:
        assert "source_fx_node" in node.attributes
