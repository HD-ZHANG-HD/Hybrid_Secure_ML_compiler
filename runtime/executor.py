
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .conversion import ConversionManager, conversion_manager
from .operator_registry import OperatorRegistry
from .plan import BootstrapStep, ConversionStep, ExecutionPlan, OperatorStep
from .profiling.collector import ProfilingCollector
from .profiling.network_model import NetworkConfig, NetworkModel
from .profiling.schema import ConversionProfileRecord, OperatorProfileRecord
from .types import BackendType, ExecutionContext, TensorValue


def _require_tensor(tensors: Dict[str, TensorValue], name: str) -> TensorValue:
    if name not in tensors:
        raise KeyError(f"Missing tensor in execution context: {name}")
    return tensors[name]


def _shape_of(value) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(value).shape)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n)


def _resolve_collector(ctx: ExecutionContext) -> ProfilingCollector | None:
    collector = ctx.profiling_collector or ctx.params.get("profiling_collector")
    return collector if isinstance(collector, ProfilingCollector) else None


def _resolve_network_config(ctx: ExecutionContext) -> NetworkConfig:
    cfg = ctx.network_config or ctx.params.get("network_config")
    if isinstance(cfg, NetworkConfig):
        return cfg
    return NetworkConfig(
        bandwidth_bytes_per_sec=float(ctx.params.get("bandwidth_bytes_per_sec", 125_000_000.0)),
        rtt_ms=float(ctx.params.get("rtt_ms", 1.0)),
    )


def _lookup_comm_override(ctx: ExecutionContext, category: str, key: tuple[str, ...]) -> tuple[int, int] | None:
    overrides = ctx.params.get(category, {})
    if not isinstance(overrides, dict):
        return None
    payload = overrides.get("|".join(key))
    if payload is None and len(key) > 1:
        payload = overrides.get("|".join(key[:-1]))
    if not isinstance(payload, dict):
        return None
    return int(payload.get("comm_bytes", 0)), int(payload.get("comm_rounds", 0))


def _estimate_operator_comm(step: OperatorStep, output_shape: tuple[int, ...], ctx: ExecutionContext) -> tuple[int, int]:
    if step.backend == BackendType.HE:
        return 0, 0
    override = _lookup_comm_override(
        ctx,
        "profiling_operator_comm",
        (step.op_type, step.backend.value, step.method),
    )
    if override is not None:
        return override
    default_rounds = int(ctx.params.get("profiling_default_mpc_rounds", 2))
    return _numel(output_shape) * 8, default_rounds


def _estimate_conversion_comm(step: ConversionStep, source_shape: tuple[int, ...], source: TensorValue, ctx: ExecutionContext) -> tuple[int, int, str]:
    layout_family = str(source.meta.get("layout_family", "generic"))
    override = _lookup_comm_override(
        ctx,
        "profiling_conversion_comm",
        (
            f"{step.from_domain.value}_to_{step.to_domain.value}",
            step.method,
            layout_family,
        ),
    )
    if override is not None:
        return override[0], override[1], layout_family
    default_rounds = int(ctx.params.get("profiling_default_conversion_rounds", 2))
    return _numel(source_shape) * 16, default_rounds, layout_family


def execute(
    plan: ExecutionPlan,
    tensors: Dict[str, TensorValue],
    ctx: ExecutionContext | None = None,
    registry: OperatorRegistry | None = None,
    conversion_mgr: ConversionManager | None = None,
) -> Dict[str, TensorValue]:
    ctx = ctx or ExecutionContext()
    if registry is None:
        raise ValueError("runtime.execute requires an OperatorRegistry")

    resolved_conversion_manager = conversion_mgr or conversion_manager
    collector = _resolve_collector(ctx)
    network_config = _resolve_network_config(ctx)

    for step in plan.steps:
        if isinstance(step, OperatorStep):
            inputs = [_require_tensor(tensors, name) for name in step.inputs]
            for tensor_name, tensor in zip(step.inputs, inputs):
                if tensor.domain != step.backend:
                    raise ValueError(
                        "Execution plan domain mismatch: "
                        f"step {step.op_type}@{step.backend.value} expected {tensor_name} "
                        f"in {step.backend.value}, found {tensor.domain.value}"
                    )
            fn = registry.get(step.op_type, step.backend, method_name=step.method)
            ctx.trace.append(f"EXECUTE {step.op_type}@{step.backend.value}/{step.method}")
            start = time.perf_counter()
            out = fn(inputs, ctx)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            output_names = step.outputs or [f"{step.op_type.lower()}_out"]
            if len(output_names) != 1:
                raise ValueError(f"Operator step currently expects exactly one output name: {step.op_type}")
            tensors[output_names[0]] = out
            if collector is not None:
                output_shape = _shape_of(out.data)
                comm_bytes, comm_rounds = _estimate_operator_comm(step, output_shape, ctx)
                total_latency_ms = (
                    elapsed_ms
                    if step.backend == BackendType.HE
                    else NetworkModel.estimate_latency(elapsed_ms, comm_bytes, comm_rounds, network_config)
                )
                collector.add_operator_record(
                    OperatorProfileRecord(
                        op_type=step.op_type,
                        backend=step.backend.value,
                        method=step.method,
                        input_shape=_shape_of(inputs[0].data) if inputs else tuple(),
                        output_shape=output_shape,
                        local_compute_ms=elapsed_ms,
                        comm_bytes=comm_bytes,
                        comm_rounds=comm_rounds,
                        total_latency_ms=total_latency_ms,
                        metadata={"network": network_config.describe()},
                    )
                )
            continue

        if isinstance(step, ConversionStep):
            source = _require_tensor(tensors, step.tensor)
            if source.domain != step.from_domain:
                raise ValueError(
                    "Execution plan conversion mismatch: "
                    f"{step.tensor} expected {step.from_domain.value}, found {source.domain.value}"
                )
            start = time.perf_counter()
            converted = resolved_conversion_manager.convert(
                source,
                target=step.to_domain,
                ctx=ctx,
                method_name=step.method,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            tensors[step.output_tensor or step.tensor] = converted
            if collector is not None:
                source_shape = _shape_of(source.data)
                comm_bytes, comm_rounds, layout_family = _estimate_conversion_comm(step, source_shape, source, ctx)
                collector.add_conversion_record(
                    ConversionProfileRecord(
                        direction=f"{step.from_domain.value}_to_{step.to_domain.value}",
                        method=step.method,
                        layout_family=layout_family,
                        tensor_shape=source_shape,
                        local_compute_ms=elapsed_ms,
                        comm_bytes=comm_bytes,
                        comm_rounds=comm_rounds,
                        total_latency_ms=NetworkModel.estimate_latency(
                            elapsed_ms,
                            comm_bytes,
                            comm_rounds,
                            network_config,
                        ),
                        metadata={"network": network_config.describe()},
                    )
                )
            continue

        if isinstance(step, BootstrapStep):
            source = _require_tensor(tensors, step.tensor)
            if source.domain != step.backend:
                raise ValueError(
                    "Execution plan bootstrap mismatch: "
                    f"{step.tensor} expected {step.backend.value}, found {source.domain.value}"
                )
            # NEXUS-backed HE methods currently do not expose an in-place
            # bootstrap primitive; the framework's plan builder therefore
            # only emits BootstrapStep for methods that advertise it.
            # At runtime we forward the tensor unchanged — semantically the
            # level budget is reset, and the downstream operator step
            # carries the execution. Any method that wires up a real
            # ciphertext bootstrap primitive should replace this block
            # with a method-level `bootstrap()` dispatch.
            start = time.perf_counter()
            refreshed = TensorValue(source.data, source.domain, dict(source.meta))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            tensors[step.output_tensor or step.tensor] = refreshed
            ctx.trace.append(
                f"BOOTSTRAP {step.tensor}@{step.backend.value}/{step.method} elapsed={elapsed_ms:.3f}ms"
            )
            continue

        raise TypeError(f"Unsupported execution step: {type(step)!r}")

    tensors["__trace__"] = TensorValue(list(ctx.trace), BackendType.HYBRID)
    return tensors
