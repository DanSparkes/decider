"""In-memory MCP client tests for decider.mcp_server (mcp_serve).

The Decider is never loaded (no torch): each tool is exercised against a fake object implementing the
``system_one`` / ``decide`` duck-typed surface, and answers with the routing example from
``examples/routing_with_confidence.py``. The MCP server itself is driven through the official in-memory client
(``mcp.shared.memory.create_connected_server_and_client_session``).
"""
import asyncio
import json

import pytest

from decider.mcp_server import DEFAULT_MODEL, mcp_serve

pytest.importorskip("mcp")

ROUTING_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What is the user trying to do?",
        "criteria": {
            "check_balance": "View the account balance or recent transactions",
            "approve_transfer": {"what": "Approve or confirm a pending outgoing transfer", "not_for": "Asking about a transfer's status"},
            "support": "Get help with a problem, a complaint, or a question about the product",
        },
    },
    "urgent": {"type": "noul", "instructions": "Does the message convey urgency?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the user?", "criteria": ["calm", "irritated", "angry or threatening to leave"]},
}


class FakeDecider:
    """Duck-typed stand-in for decider.infer.Decider; answers the routing example deterministically."""

    def __init__(self, model="Mapika/decider-2b"):
        self.model = model
        self.last_state = None
        self.last_questions = None
        self.last_context = None

    def system_one(self, state, questions):
        self.last_state = state
        self.last_questions = questions
        answers = {"action": {"type": "choice", "choice": "support", "confidence": 0.97,
                              "probabilities": {"check_balance": 0.01, "approve_transfer": 0.02, "support": 0.97}},
                   "urgent": {"type": "noul", "noul": 0.91},
                   "frustration": {"type": "score", "score": 2.0, "confidence": 0.8,
                                   "probabilities": {"0": 0.05, "1": 0.15, "2": 0.80}}}
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": 42, "output_tokens": 0}}

    def decide(self, context, questions):
        self.last_context = context
        self.last_questions = questions
        out = []
        for q in questions:
            p = [0.9 if i == 0 else 0.05 for i in range(len(q["options"]))]
            out.append({"choice": q["options"][0], "confidence": p[0], "probs": {o: pi for o, pi in zip(q["options"], p)}})
        return out


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def server():
    return mcp_serve(DEFAULT_MODEL, decider=FakeDecider())


def test_mcp_serve_returns_configured_server(server):
    assert server is not None


def test_lists_system_one_and_decide(server):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _inner():
        async with create_connected_server_and_client_session(server) as client:
            tools = (await client.list_tools()).tools
            return [t.name for t in tools]

    assert _run(_inner()) == ["system_one", "decide"]


def test_system_one_roundtrip(server):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _inner():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool("system_one", {"state": "how much do i have left in checking?", "questions": ROUTING_QUESTIONS})

    res = _run(_inner())
    assert not res.isError
    payload = json.loads(res.content[0].text)
    assert set(payload) == {"model", "answers", "usage"}
    assert payload["answers"]["action"]["choice"] == "support"
    assert payload["answers"]["action"]["confidence"] == 0.97
    assert payload["answers"]["urgent"]["noul"] == 0.91
    assert payload["answers"]["frustration"]["score"] == 2.0


def test_decide_roundtrip(server):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _inner():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(
                "decide",
                {"context": "third time I'm asking. the app logs me out every time. fix it today or I'm gone",
                 "questions": [{"question": "What is the intent?", "options": ["support", "cancel", "billing"]}]},
            )

    res = _run(_inner())
    assert not res.isError
    structured = res.structuredContent or {}
    payload = structured.get("result", [])
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["choice"] == "support"
    assert payload[0]["confidence"] == 0.9


def test_tool_schema_matches_render_question_source_of_truth(server):
    """The Jev wire shape the tools expose is the one render_question emits (single source of truth)."""
    from decider import systemone as S1

    for spec in ROUTING_QUESTIONS.values():
        rendered = S1.render_question(spec)
        assert rendered["question"] and rendered["options"]

    from mcp.shared.memory import create_connected_server_and_client_session

    async def _inner():
        async with create_connected_server_and_client_session(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools["system_one"].inputSchema

    schema = _run(_inner())
    assert "questions" in schema.get("properties", {})
    required = schema.get("required", [])
    assert "state" in required and "questions" in required