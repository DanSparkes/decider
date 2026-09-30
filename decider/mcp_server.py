"""Model Context Protocol server exposing decider as typed tools (the ``/v1/systemone`` and ``/decide`` surfaces).

    from decider.mcp_server import mcp_serve
    server = mcp_serve(model="Mapika/decider-2b")          # a configured low-level MCP Server

    # stdio transport (the CLI):            uv run decider mcp --model Mapika/decider-2b
    # run it with the in-memory client:     from mcp.shared.memory import create_connected_server_and_client_session
    #                                       async with create_connected_server_and_client_session(mcp_serve(...)) as c: ...

Tools:
  - ``system_one(state, questions)``  the Jev/TypeSafe wire format (``decider.systemone``): a state plus a dict of
    typed questions (Choice up to 255 options, Score 2..10 levels, Noul). Returns ``{"answers": {id: answer}}``.
  - ``decide(context, questions)``    the plain typed-decide form: a context string plus a list of
    ``{"question", "options"}`` dicts. Returns a list of one result per question.

Every question spec is validated by ``decider.systemone.render_question``, which is kept as the single source of
truth for the Jev wire shape: adding a type, an option set or a level legend there is what both the HTTP server
(``decider.serve``), this MCP server and the evaluation harness see.
"""

import argparse
import logging

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Mapika/decider-2b"


_trace_state: dict = {"writer": None, "side": None, "ready": False}


def _trace_log(*, model: str, layout: str, state_hash: str, question_ids: list,
               n_opts: list, prompt_token_count: int, output: list,
               state=None, questions=None):
    """Mirror decider/serve.py: one canonical JSONL record per call, plus a
    .states.jsonl sidecar (DECIDER_TRACE_STATES, default on) so decider.replay
    can rescore the same prompts against different weights."""
    st = _trace_state
    if not st["ready"]:
        import os
        path = os.environ.get("DECIDER_TRACE", "")
        if path:
            from decider.trace import TraceWriter
            st["writer"] = TraceWriter(path)
            if os.environ.get("DECIDER_TRACE_STATES", "1") == "1":
                _side_p = st["writer"].path.with_suffix(".states.jsonl")
                st["side"] = open(_side_p, "a", encoding="utf-8")  # noqa: SIM115
        st["ready"] = True
    w = st["writer"]
    if w is None:
        return None
    rid = w.log(model=model, layout=layout, state_hash=state_hash,
                question_ids=question_ids, n_opts=n_opts,
                prompt_token_count=prompt_token_count, output=output)
    w.flush()
    side = st["side"]
    if side is not None and state is not None and questions is not None:
        import json
        side.write(json.dumps({"request_id": rid, "state": state, "questions": questions}, ensure_ascii=False) + "\n")
        side.flush()
    return rid


def mcp_serve(model: str = DEFAULT_MODEL, *, decider=None, name: str = "decider", **load_kwargs):
    """Build an MCP Server exposing ``system_one`` and ``decide`` as typed tools.

    The Decider is constructed lazily on the first tool call (from ``model``), so importing this module and calling
    ``mcp_serve`` never touches torch. Pass an already-built Decider (or a duck-typed stand-in) as ``decider`` to
    skip loading entirely — the in-memory tests use a fake object that answers with fixed routing outputs.
    """
    from mcp.server.fastmcp import FastMCP

    box = {"decider": decider, "model": model, "load": load_kwargs}

    def _get_decider():
        if box["decider"] is None:
            from decider.infer import Decider

            box["decider"] = Decider(box["model"], **box["load"])
        return box["decider"]

    app = FastMCP(name, instructions="decider: one-pass typed decisions with calibrated probabilities")

    @app.tool(
        description=(
            "Answer Jev-shaped typed questions over a state. `state` is a string or a JSON value; `questions` is a "
            "dict of {id: {type, instructions, criteria}} where type is 'choice' (criteria: {name: description}), "
            " 'score' (criteria: ordered list of level descriptions) or 'noul' (criteria: {true, false}). Returns "
            "{answers: {id: {type, choice|score|noul, confidence, probabilities}}}."
        )
    )
    def system_one(state: dict | str | list, questions: dict[str, object]) -> dict:
        """Jev/TypeSafe wire-format questions over a state (decider.systemone)."""
        out = _get_decider().system_one(state, questions)
        from decider.trace import hash_state, make_output_records
        _trace_log(model=str(out.get("model", model)), layout="schema_first", state_hash=hash_state(state),
                   question_ids=list(questions.keys()),
                   n_opts=[len(q.get("criteria") or []) for q in questions.values()],
                   prompt_token_count=0, output=make_output_records(out["answers"]),
                   state=state, questions=questions)
        return out

    @app.tool(
        description=(
            "Plain typed decisions from a context string and a list of {question, options} dicts. Returns one result "
            "per question: {choice, confidence, probs: {option: p}}. The list is delivered as structuredContent "
            "(and as one text block per decision)."
        )
    )
    def decide(context: str, questions: list[dict[str, object]]) -> list[dict[str, object]]:
        """Typed decisions in the plain decide format (decider.infer.Decider.decide)."""
        out = _get_decider().decide(context, questions)
        from decider.trace import hash_state
        recs = []
        for ans in out:
            rec = {"type": ans.get("type", "choice"),
                   "probabilities": dict(ans.get("probabilities", ans.get("probs", {})))}
            for k in ("choice", "score", "noul"):
                if k in ans:
                    rec[k] = ans[k]
            recs.append(rec)
        _trace_log(model=model, layout="state_first", state_hash=hash_state(context),
                   question_ids=[str(q.get("question", i)) for i, q in enumerate(questions)],
                   n_opts=[len(q.get("options") or []) for q in questions],
                   prompt_token_count=0, output=recs)
        return out

    return app._mcp_server


def main(argv=None) -> int:
    """CLI entry point (console script ``decider``): ``decider mcp --model ...`` runs the server on stdio."""
    p = argparse.ArgumentParser(prog="decider", description="decider CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    mcp = sub.add_parser("mcp", help="run the Model Context Protocol server (stdio transport)")
    mcp.add_argument("--model", default=DEFAULT_MODEL, help="model path or HF repo id (default %(default)s)")
    mcp.add_argument("--device", default=None, help="device: cuda or cpu (default: cuda if available)")
    mcp.add_argument("--temperature", type=float, default=None, help="override the fitted temperature")
    mcp.add_argument("--no-graphs", action="store_true", help="disable CUDA graphs (use eager mode)")
    mcp.add_argument(
        "--backend",
        default=None,
        choices=("mapika", "laya"),
        help="decision backend: mapika (torch Decider, default) or laya (laya-mlx, Apple silicon)",
    )
    args = p.parse_args(argv)
    if args.cmd == "mcp":
        import asyncio

        import mcp

        kwargs = {}
        if args.device is None:
            try:
                import torch

                kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover - torch is a core dependency, but be kind to the test env
                kwargs["device"] = "cpu"
        else:
            kwargs["device"] = args.device
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.no_graphs:
            kwargs["use_graphs"] = False
        backend = args.backend or __import__("os").environ.get("DECIDER_BACKEND", "mapika")
        if backend == "laya":
            from decider.laya_backend import LayaBackend

            server = mcp_serve(args.model, decider=LayaBackend(), **kwargs)
        else:
            server = mcp_serve(args.model, **kwargs)

        async def _run_stdio():
            from mcp import stdio_server

            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(_run_stdio())
        return 0
    p.error(f"unknown subcommand {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())