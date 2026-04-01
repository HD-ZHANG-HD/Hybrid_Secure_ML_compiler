from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode


def _as_shape(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


def load_graph_json(path: str | Path) -> OperatorGraph:
    payload = json.loads(Path(path).read_text())
    nodes = [
        OperatorNode(
            node_id=str(node["node_id"]),
            op_type=str(node["op_type"]),
            input_shape=_as_shape(node["input_shape"]),
            output_shape=_as_shape(node["output_shape"]),
            attributes=dict(node.get("attributes", {})),
        )
        for node in payload["nodes"]
    ]
    edges = [
        DataEdge(
            src=str(edge["src"]),
            dst=str(edge["dst"]),
            tensor_shape=_as_shape(edge["tensor_shape"]),
        )
        for edge in payload["edges"]
    ]
    return OperatorGraph(graph_id=str(payload.get("graph_id", "graph")), nodes=nodes, edges=edges)


class GraphView:
    def __init__(self, graph: OperatorGraph) -> None:
        self.graph = graph
        self.node_map: Dict[str, OperatorNode] = {node.node_id: node for node in graph.nodes}
        self.incoming: Dict[str, List[DataEdge]] = defaultdict(list)
        self.outgoing: Dict[str, List[DataEdge]] = defaultdict(list)
        self.indegree: Dict[str, int] = {node.node_id: 0 for node in graph.nodes}
        for edge in graph.edges:
            self.incoming[edge.dst].append(edge)
            self.outgoing[edge.src].append(edge)
            self.indegree[edge.dst] += 1
        self.topological_order = self._topological_order()
        self.stage_by_node = self._stage_numbers()

    def _topological_order(self) -> List[str]:
        indeg = dict(self.indegree)
        q = deque(node.node_id for node in self.graph.nodes if indeg[node.node_id] == 0)
        order: List[str] = []
        while q:
            node_id = q.popleft()
            order.append(node_id)
            for edge in self.outgoing.get(node_id, []):
                indeg[edge.dst] -= 1
                if indeg[edge.dst] == 0:
                    q.append(edge.dst)
        if len(order) != len(self.graph.nodes):
            raise ValueError("OperatorGraph must be a DAG for state-expanded optimization")
        return order

    def _stage_numbers(self) -> Dict[str, int]:
        stage_by_node: Dict[str, int] = {}
        for node_id in self.topological_order:
            preds = self.incoming.get(node_id, [])
            if not preds:
                stage_by_node[node_id] = 0
                continue
            stage_by_node[node_id] = max(stage_by_node[edge.src] for edge in preds) + 1
        return stage_by_node

    def source_nodes(self) -> List[str]:
        return [node.node_id for node in self.graph.nodes if not self.incoming.get(node.node_id)]

    def sink_nodes(self) -> List[str]:
        return [node.node_id for node in self.graph.nodes if not self.outgoing.get(node.node_id)]

    def merge_nodes(self) -> List[str]:
        return [node.node_id for node in self.graph.nodes if len(self.incoming.get(node.node_id, [])) > 1]

    def is_chain(self) -> bool:
        for node in self.graph.nodes:
            if len(self.incoming.get(node.node_id, [])) > 1:
                return False
            if len(self.outgoing.get(node.node_id, [])) > 1:
                return False
        return True

    def stages(self) -> List[List[str]]:
        grouped: Dict[int, List[str]] = defaultdict(list)
        for node_id in self.topological_order:
            grouped[self.stage_by_node[node_id]].append(node_id)
        return [grouped[idx] for idx in sorted(grouped)]
