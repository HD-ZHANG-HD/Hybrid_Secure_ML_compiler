from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from runtime.plan import BootstrapStep, ConversionStep, ExecutionPlan, OperatorStep
from runtime.types import BackendType

from backends.layout.bert_edge_packing import supports_bert_edge_conversion_shape
from compiler.min_cut.profiler_db import ProfilerDB
from compiler.min_cut.runtime_plan_adapter import resolve_method_name
from ir.types import OperatorGraph

from .cost_model import StateExpandedCostModel
from .plan_builder import build_execution_plan
from .solver import SolverResult, solve_state_expanded


CONVERSION_METHOD_DEFAULT = "method_default"
CONVERSION_METHOD_SCI_RESTRICTED = "method_sci_restricted"


def resolve_conversion_method(tensor_shape: tuple[int, ...]) -> str:
    if supports_bert_edge_conversion_shape(tensor_shape):
        return CONVERSION_METHOD_SCI_RESTRICTED
    return CONVERSION_METHOD_DEFAULT


def _incoming_edge_map(graph: OperatorGraph):
    incoming = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.dst].append(edge)
    return incoming


def _source_nodes(graph: OperatorGraph) -> List[str]:
    incoming = {edge.dst for edge in graph.edges}
    return [node.node_id for node in graph.nodes if node.node_id not in incoming]


def compiler_plan_to_runtime_plan(
    graph: OperatorGraph,
    compiler_plan: Dict[str, object],
    external_inputs: Dict[str, List[str]] | None = None,
) -> ExecutionPlan:
    node_map = {node.node_id: node for node in graph.nodes}
    incoming = _incoming_edge_map(graph)
    provided_inputs = dict(external_inputs or {})
    for node_id in _source_nodes(graph):
        provided_inputs.setdefault(node_id, ["input"])

    runtime_steps: List[OperatorStep | ConversionStep | BootstrapStep] = []
    node_output_tensor: Dict[str, str] = {}
    conversion_output_tensor: Dict[tuple[str, str], str] = {}
    bootstrap_output_tensor: Dict[str, str] = {}

    for raw_step in compiler_plan["steps"]:  # type: ignore[index]
        step = dict(raw_step)
        kind = str(step["kind"])
        if kind == "conversion":
            from_node = str(step["from_node"])
            to_node = str(step["to_node"])
            source_tensor = node_output_tensor.get(from_node, f"{from_node}__out")
            output_tensor = f"{from_node}__to__{to_node}"
            runtime_steps.append(
                ConversionStep(
                    from_domain=BackendType(str(step["from_domain"])),
                    to_domain=BackendType(str(step["to_domain"])),
                    tensor=source_tensor,
                    method=resolve_conversion_method(tuple(step.get("tensor_shape", []))),
                    output_tensor=output_tensor,
                )
            )
            conversion_output_tensor[(from_node, to_node)] = output_tensor
            continue
        if kind == "bootstrap":
            node_id = str(step["node_id"])
            source_name = ""
            if incoming[node_id]:
                source_edge = incoming[node_id][0]
                source_name = conversion_output_tensor.get(
                    (source_edge.src, node_id),
                    node_output_tensor.get(source_edge.src, f"{source_edge.src}__out"),
                )
            else:
                source_name = provided_inputs[node_id][0]
            output_tensor = f"{node_id}__bootstrap"
            runtime_steps.append(
                BootstrapStep(
                    backend=BackendType.HE,
                    tensor=source_name,
                    method="method_default",
                    output_tensor=output_tensor,
                )
            )
            bootstrap_output_tensor[node_id] = output_tensor
            continue
        if kind not in {"execute_he", "execute_mpc"}:
            raise ValueError(f"Unsupported compiler plan step kind: {kind}")

        node_id = str(step["node_id"])
        node = node_map[node_id]
        domain = "HE" if kind == "execute_he" else "MPC"
        if incoming[node_id]:
            input_tensor_names: List[str] = []
            for edge in incoming[node_id]:
                if edge.src == incoming[node_id][0].src and node_id in bootstrap_output_tensor:
                    input_tensor_names.append(bootstrap_output_tensor[node_id])
                    continue
                input_tensor_names.append(
                    conversion_output_tensor.get((edge.src, node_id), node_output_tensor.get(edge.src, f"{edge.src}__out"))
                )
        else:
            input_tensor_names = list(provided_inputs[node_id])
            if node_id in bootstrap_output_tensor:
                input_tensor_names = [bootstrap_output_tensor[node_id]]
        output_tensor = f"{node_id}__out"
        runtime_steps.append(
            OperatorStep(
                op_type=node.op_type,
                method=resolve_method_name(node.op_type, domain),  # type: ignore[arg-type]
                backend=BackendType(domain),
                inputs=input_tensor_names,
                outputs=[output_tensor],
            )
        )
        node_output_tensor[node_id] = output_tensor

    return ExecutionPlan(steps=runtime_steps)


def compile_graph_state_expanded(
    graph: OperatorGraph,
    profiler_db: ProfilerDB,
    external_inputs: Dict[str, List[str]] | None = None,
    max_level_bucket: int = 3,
    default_bootstrap_ms: float = 6.0,
) -> tuple[SolverResult, Dict[str, object], ExecutionPlan]:
    cost_model = StateExpandedCostModel(
        db=profiler_db,
        default_level_bucket=max_level_bucket,
        default_bootstrap_ms=default_bootstrap_ms,
    )
    solver_result = solve_state_expanded(graph, cost_model)
    compiler_plan = build_execution_plan(solver_result)
    runtime_plan = compiler_plan_to_runtime_plan(graph, compiler_plan, external_inputs=external_inputs)
    return solver_result, compiler_plan, runtime_plan
