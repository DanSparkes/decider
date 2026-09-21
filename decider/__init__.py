"""decider: one-pass typed decisions with calibrated probabilities.

Public entry points are re-exported here. Optional surfaces (e.g. the MCP server under the ``[mcp]`` extra) are
re-exported lazily so importing ``decider`` never pulls in optional dependencies.
"""


def mcp_serve(*args, **kwargs):
    """Build a Model Context Protocol server exposing decider as typed tools (requires the ``[mcp]`` extra)."""
    from decider.mcp_server import mcp_serve as _mcp_serve

    return _mcp_serve(*args, **kwargs)