"""Operator metadata for Out_Projection."""

OPERATOR_NAME = "Out_Projection"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["context"],
    "outputs": ["attn_proj"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt_as_ffn1",
        "available_methods": [
            "method_mpc_bolt_as_ffn1",
            "method_he_nexus_as_ffn1",
        ],
        "lowered_to": "FFN_Linear_1",
        "note": "Out_Projection has no dedicated primitive; reuses FFN_Linear_1 adapters",
    },
}
