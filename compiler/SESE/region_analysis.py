from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple

from ir.types import OperatorGraph


SYNTHETIC_ENTRY = "__sese_entry__"
SYNTHETIC_EXIT = "__sese_exit__"


@dataclass(frozen=True)
class SESEBlock:
    block_id: str
    kind: str
    entry: str
    exit: str
    nodes: Tuple[str, ...]
    internal_nodes: Tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "entry": self.entry,
            "exit": self.exit,
            "nodes": list(self.nodes),
            "internal_nodes": list(self.internal_nodes),
        }


@dataclass(frozen=True)
class RegionAnalysisResult:
    graph_id: str
    topological_order: Tuple[str, ...]
    dominators: Dict[str, Tuple[str, ...]]
    post_dominators: Dict[str, Tuple[str, ...]]
    blocks: Tuple[SESEBlock, ...]
    block_edges: Tuple[Tuple[str, str], ...]
    block_topological_order: Tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "topological_order": list(self.topological_order),
            "dominators": {node: list(values) for node, values in self.dominators.items()},
            "post_dominators": {node: list(values) for node, values in self.post_dominators.items()},
            "blocks": [block.as_dict() for block in self.blocks],
            "block_edges": [list(edge) for edge in self.block_edges],
            "block_topological_order": list(self.block_topological_order),
        }


class RegionGraphView:
    def __init__(self, graph: OperatorGraph) -> None:
        self.graph = graph
        self.node_ids: Tuple[str, ...] = tuple(node.node_id for node in graph.nodes)
        self.node_set: Set[str] = set(self.node_ids)
        self.predecessors: Dict[str, List[str]] = defaultdict(list)
        self.successors: Dict[str, List[str]] = defaultdict(list)

        for edge in graph.edges:
            if edge.src not in self.node_set or edge.dst not in self.node_set:
                raise ValueError(f"Edge refers to unknown node: {edge.src} -> {edge.dst}")
            self.predecessors[edge.dst].append(edge.src)
            self.successors[edge.src].append(edge.dst)

        self.source_nodes: Tuple[str, ...] = tuple(node_id for node_id in self.node_ids if not self.predecessors[node_id])
        self.sink_nodes: Tuple[str, ...] = tuple(node_id for node_id in self.node_ids if not self.successors[node_id])
        self.topological_order: Tuple[str, ...] = self._topological_order()
        self.synthetic_predecessors, self.synthetic_successors = self._build_augmented_adjacency()

    def _topological_order(self) -> Tuple[str, ...]:
        indegree: MutableMapping[str, int] = {node_id: len(self.predecessors[node_id]) for node_id in self.node_ids}
        queue = deque(node_id for node_id in self.node_ids if indegree[node_id] == 0)
        order: List[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for succ in self.successors[node_id]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.node_ids):
            raise ValueError("SESE region analysis requires OperatorGraph to be a DAG")
        return tuple(order)

    def _build_augmented_adjacency(self) -> tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
        preds: Dict[str, List[str]] = {node_id: list(values) for node_id, values in self.predecessors.items()}
        succs: Dict[str, List[str]] = {node_id: list(values) for node_id, values in self.successors.items()}

        preds[SYNTHETIC_ENTRY] = []
        succs[SYNTHETIC_ENTRY] = list(self.source_nodes)
        for source in self.source_nodes:
            preds.setdefault(source, []).append(SYNTHETIC_ENTRY)

        succs[SYNTHETIC_EXIT] = []
        preds[SYNTHETIC_EXIT] = list(self.sink_nodes)
        for sink in self.sink_nodes:
            succs.setdefault(sink, []).append(SYNTHETIC_EXIT)

        for node_id in self.node_ids:
            preds.setdefault(node_id, [])
            succs.setdefault(node_id, [])

        return (
            {node: tuple(values) for node, values in preds.items()},
            {node: tuple(values) for node, values in succs.items()},
        )

    def merge_nodes(self) -> List[str]:
        return [node_id for node_id in self.topological_order if len(self.predecessors[node_id]) > 1]


def _compute_dominators(
    nodes: Iterable[str],
    predecessors: Mapping[str, Tuple[str, ...]],
    start_node: str,
) -> Dict[str, Set[str]]:
    node_list = list(nodes)
    node_set = set(node_list)
    dominators: Dict[str, Set[str]] = {}
    for node in node_list:
        if node == start_node:
            dominators[node] = {node}
        else:
            dominators[node] = set(node_set)

    changed = True
    while changed:
        changed = False
        for node in node_list:
            if node == start_node:
                continue
            preds = predecessors[node]
            if not preds:
                new_value = {node}
            else:
                intersection = set(dominators[preds[0]])
                for pred in preds[1:]:
                    intersection &= dominators[pred]
                new_value = intersection | {node}
            if new_value != dominators[node]:
                dominators[node] = new_value
                changed = True
    return dominators


def _dominator_depths(topological_nodes: Iterable[str], dominators: Mapping[str, Set[str]]) -> Dict[str, int]:
    return {node: len(dominators[node]) for node in topological_nodes}


def _ordered_tuple(values: Iterable[str], rank: Mapping[str, int]) -> Tuple[str, ...]:
    return tuple(sorted(values, key=lambda item: rank[item]))


def _extract_residual_blocks(
    view: RegionGraphView,
    dominators: Mapping[str, Set[str]],
    post_dominators: Mapping[str, Set[str]],
) -> List[SESEBlock]:
    topo_rank = {node_id: idx for idx, node_id in enumerate(view.topological_order)}
    dom_depth = _dominator_depths((SYNTHETIC_ENTRY, *view.topological_order), dominators)
    blocks: List[SESEBlock] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for merge_node in view.merge_nodes():
        preds = view.predecessors[merge_node]
        common_dominators = set(dominators[preds[0]])
        for pred in preds[1:]:
            common_dominators &= dominators[pred]
        candidates = [node for node in common_dominators if node not in {SYNTHETIC_ENTRY, merge_node} and merge_node in post_dominators[node]]
        if not candidates:
            continue
        entry = max(candidates, key=lambda node: (dom_depth[node], -topo_rank.get(node, -1)))
        pair = (entry, merge_node)
        if pair in seen_pairs:
            continue
        region_nodes = [
            node_id
            for node_id in view.topological_order
            if entry in dominators[node_id] and merge_node in post_dominators[node_id]
        ]
        if len(region_nodes) < 3:
            continue
        seen_pairs.add(pair)
        blocks.append(
            SESEBlock(
                block_id=f"residual::{entry}::{merge_node}",
                kind="residual",
                entry=entry,
                exit=merge_node,
                nodes=tuple(region_nodes),
                internal_nodes=tuple(node_id for node_id in region_nodes if node_id not in {entry, merge_node}),
            )
        )

    blocks.sort(key=lambda block: (len(block.nodes), topo_rank[block.entry], topo_rank[block.exit]))
    return blocks


def _pack_chain_blocks(view: RegionGraphView, claimed_nodes: Set[str]) -> List[SESEBlock]:
    blocks: List[SESEBlock] = []
    topo_rank = {node_id: idx for idx, node_id in enumerate(view.topological_order)}
    visited: Set[str] = set()

    for node_id in view.topological_order:
        if node_id in claimed_nodes or node_id in visited:
            continue
        chain = [node_id]
        visited.add(node_id)
        cursor = node_id
        while True:
            uncovered_successors = [
                succ
                for succ in view.successors[cursor]
                if succ not in claimed_nodes
                and succ not in visited
                and len(view.predecessors[succ]) == 1
                and len(view.successors[cursor]) == 1
            ]
            if len(uncovered_successors) != 1:
                break
            next_node = uncovered_successors[0]
            if len(view.predecessors[next_node]) != 1:
                break
            chain.append(next_node)
            visited.add(next_node)
            cursor = next_node

        if len(chain) == 1:
            kind = "atomic"
        else:
            kind = "chain"
        blocks.append(
            SESEBlock(
                block_id=f"{kind}::{chain[0]}::{chain[-1]}",
                kind=kind,
                entry=chain[0],
                exit=chain[-1],
                nodes=tuple(chain),
                internal_nodes=tuple(chain[1:-1]),
            )
        )

    blocks.sort(key=lambda block: (topo_rank[block.entry], topo_rank[block.exit]))
    return blocks


def _build_block_graph(
    graph: OperatorGraph,
    blocks: Iterable[SESEBlock],
) -> tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
    node_to_block: Dict[str, str] = {}
    ordered_blocks = list(blocks)
    for block in ordered_blocks:
        for node_id in block.nodes:
            if node_id in node_to_block:
                raise ValueError(f"Node {node_id} assigned to multiple SESE blocks")
            node_to_block[node_id] = block.block_id

    edges: Set[Tuple[str, str]] = set()
    successors: Dict[str, Set[str]] = {block.block_id: set() for block in ordered_blocks}
    indegree: Dict[str, int] = {block.block_id: 0 for block in ordered_blocks}

    for edge in graph.edges:
        src_block = node_to_block[edge.src]
        dst_block = node_to_block[edge.dst]
        if src_block == dst_block:
            continue
        pair = (src_block, dst_block)
        if pair in edges:
            continue
        edges.add(pair)
        successors[src_block].add(dst_block)
        indegree[dst_block] += 1

    queue = deque(block.block_id for block in ordered_blocks if indegree[block.block_id] == 0)
    topo_order: List[str] = []
    while queue:
        block_id = queue.popleft()
        topo_order.append(block_id)
        for succ in sorted(successors[block_id]):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != len(ordered_blocks):
        raise ValueError("Extracted SESE block graph must be acyclic")
    return tuple(sorted(edges)), tuple(topo_order)


def analyze_sese_regions(graph: OperatorGraph) -> RegionAnalysisResult:
    view = RegionGraphView(graph)
    augmented_nodes = (SYNTHETIC_ENTRY, *view.topological_order, SYNTHETIC_EXIT)
    dominators = _compute_dominators(augmented_nodes, view.synthetic_predecessors, SYNTHETIC_ENTRY)
    post_dominators = _compute_dominators(
        reversed(augmented_nodes),
        view.synthetic_successors,
        SYNTHETIC_EXIT,
    )

    residual_blocks = _extract_residual_blocks(view, dominators, post_dominators)
    claimed_nodes: Set[str] = set()
    for block in residual_blocks:
        claimed_nodes.update(block.nodes)
    chain_blocks = _pack_chain_blocks(view, claimed_nodes)
    all_blocks = tuple(residual_blocks + chain_blocks)
    block_edges, block_topo = _build_block_graph(graph, all_blocks)

    topo_rank = {node_id: idx for idx, node_id in enumerate((SYNTHETIC_ENTRY, *view.topological_order, SYNTHETIC_EXIT))}
    ordered_dominators = {
        node_id: _ordered_tuple(values, topo_rank)
        for node_id, values in dominators.items()
    }
    ordered_post_dominators = {
        node_id: _ordered_tuple(values, topo_rank)
        for node_id, values in post_dominators.items()
    }

    return RegionAnalysisResult(
        graph_id=graph.graph_id,
        topological_order=view.topological_order,
        dominators=ordered_dominators,
        post_dominators=ordered_post_dominators,
        blocks=all_blocks,
        block_edges=block_edges,
        block_topological_order=block_topo,
    )
