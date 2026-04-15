from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_names: List[str]
    output_name: str
    # "pre_compile" = plaintext/client-side step, dropped from DAG
    # "fused_into_next" = folded into a downstream operator; kept as semantic vertex but cost = 0
    # "executed" = normal runtime vertex (default)
    role: str = "executed"
    fused_into: str = ""  # target op name when role == "fused_into_next"


# Canonical BERT-base operator sequence.
#
# - Embedding is a pre-compile / client-side step and not part of the
#   compiler's operator DAG. It is kept here for trace/backwards-compat.
# - Linear_QKV is folded into Attention_QK_MatMul (the packed [3,B,S,768]
#   input the NEXUS attention adapter expects). It is kept as a semantic
#   vertex with `cost = 0` in the cost model but the compiler prunes it
#   from the execution DAG.
# - Everything else is a normal compiler vertex.
BERT_OPERATOR_SEQUENCE: List[OperatorSpec] = [
    OperatorSpec("Embedding", ["input"], "embedding_out", role="pre_compile"),
    OperatorSpec(
        "Linear_QKV",
        ["embedding_out"],
        "qkv_out",
        role="fused_into_next",
        fused_into="Attention_QK_MatMul",
    ),
    OperatorSpec("Attention_QK_MatMul", ["qkv_out"], "qk_scores"),
    OperatorSpec("Softmax", ["qk_scores"], "attn_probs"),
    OperatorSpec("Attention_V_MatMul", ["attn_probs", "qkv_out"], "context"),
    OperatorSpec("Out_Projection", ["context"], "attn_proj"),
    OperatorSpec("Residual_Add", ["attn_proj", "embedding_out"], "attn_residual"),
    OperatorSpec("LayerNorm", ["attn_residual"], "attn_norm"),
    OperatorSpec("FFN_Linear_1", ["attn_norm"], "ffn_hidden"),
    OperatorSpec("GeLU", ["ffn_hidden"], "ffn_activated"),
    OperatorSpec("FFN_Linear_2", ["ffn_activated"], "ffn_out"),
]


def bert_executed_operator_sequence() -> List[OperatorSpec]:
    """BERT sequence with pre-compile and fused-into-next ops dropped.

    This is the compiler-facing DAG: exactly the operator vertices for
    which a real HE/MPC method exists in this tree.
    """
    return [spec for spec in BERT_OPERATOR_SEQUENCE if spec.role == "executed"]
