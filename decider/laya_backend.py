"""Optional laya-mlx backend for the decider MCP server (branch: merge-5).

laya_mlx (convaiinnovations/laya, ``typed-decisions`` subfolder) implements the
same typed-decision wire format natively in MLX: on Apple silicon it runs each
decision in tens of milliseconds with ~1/3 the resident memory of the torch
Decider (Mapika/decider-2b). Opt in with ``--backend laya`` (or env
``DECIDER_BACKEND=laya``); the torch Decider stays the default and the instant
revert path. laya's answers carry one extra ``action`` key per entry — kept
as-is because consumers only read the canonical fields.
"""
from __future__ import annotations


class LayaBackend:
    """Duck-typed Decider stand-in backed by laya_mlx (lazy model load)."""

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        subfolder: str = "typed-decisions",
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self._agent = None

    def _load(self):
        if self._agent is None:
            from laya_mlx import load

            self._agent = load(self.model_id, subfolder=self.subfolder)
        return self._agent

    def system_one(self, state, questions):
        return self._load().system_one(state, questions)

    def decide(self, context, questions):
        spec = {
            str(i): {
                "type": "choice",
                "instructions": q.get("question", ""),
                "criteria": {o: "" for o in (q.get("options") or [])},
            }
            for i, q in enumerate(questions)
        }
        out = self._load().system_one(context, spec)
        answers = out.get("answers", {})
        recs: list[dict] = []
        for i, _ in enumerate(questions):
            ans = answers.get(str(i), {})
            rec = {
                "type": ans.get("type", "choice"),
                "choice": ans.get("choice", ""),
                "confidence": ans.get("confidence", 0.0),
                "probs": ans.get("probabilities", {}),
            }
            if "score" in ans:
                rec["score"] = ans["score"]
            if "noul" in ans:
                rec["noul"] = ans["noul"]
            recs.append(rec)
        return recs


if __name__ == "__main__":  # quick manual probe
    b = LayaBackend()
    print(
        b.system_one(
            "The kitchen light is off and the door is closed.",
            {
                "light": {
                    "type": "choice",
                    "instructions": "Is the kitchen light on?",
                    "criteria": {"on": "the light is lit", "off": "the light is off"},
                }
            },
        )
    )
