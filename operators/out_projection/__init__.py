"""Out_Projection operator — lowered onto FFN_Linear_1 primitives.

No dedicated NEXUS / BOLT primitive exists for Out_Projection in this tree.
Both HE and MPC methods reuse the existing FFN_Linear_1 adapters with
`hidden_size=768, out_dim=768`; trace visibility is preserved via a
distinct op name so the compiler and profiling tools can still distinguish
the semantic operator.
"""
