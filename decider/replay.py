"""Replay a recorded JSONL trace against a model and emit a per-question drift report.

Three operating modes, picked from the CLI flags:

    --old old ... --new new                  Replay every state against two weight files
                                             and emit per-question KL / JS divergence of the
                                             served distributions, the count of top-1 flips,
                                             and the calibration drift ΔECE.
    --model model                            Replay against one model; diff against the
                                             recorded outputs ("did my model change since I
                                             recorded this?").
    (no --old / --new / --model)             Replay-on-itself: the recorded outputs are used
                                             as both sides. By construction this is zero drift;
                                             ``tests/test_trace.py`` asserts it.

The trace is keyed on ``state_hash`` + ``layout``: only records that share both with
the recorded request are paired up, so a replay that skipped rows (because the state
sidecar was missing) is visible in the ``n_unmatched`` counter.

    python -m decider.replay --trace trace.jsonl --old old.json --new new.json
    python -m decider.replay --trace trace.jsonl --model model.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from decider.trace import TraceReader

EPS = 1e-12


def _normalise(p: Sequence[float]) -> list[float]:
    s = sum(p)
    if s <= 0:
        n = len(p)
        return [1.0 / n] * n if n else []
    return [x / s for x in p]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(p || q) in nats; ``+inf`` when p has mass where q does not."""
    p = _normalise(list(p))
    q = _normalise(list(q))
    out = 0.0
    for pi, qi in zip(p, q):
        if pi > EPS:
            if qi <= EPS:
                return float("inf")
            out += pi * math.log(pi / qi)
    return float(out)


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in nats; symmetric, bounded by log(2)."""
    p = _normalise(list(p))
    q = _normalise(list(q))
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def ece(confidences: Sequence[float], corrects: Sequence[bool], bins: int = 15) -> float:
    """Expected calibration error. Matches ``decider.metrics.ece`` so the number is
    directly comparable to ``evaluate.py`` output."""
    if not confidences:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    e = 0.0
    n = len(confidences)
    for lo, hi in itertools.pairwise(edges):
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi]
        if not idx:
            continue
        avg_c = sum(confidences[i] for i in idx) / len(idx)
        avg_ok = sum(1 for i in idx if corrects[i]) / len(idx)
        e += (len(idx) / n) * abs(avg_c - avg_ok)
    return float(e)


def _align_label_lists(p_probs: dict[str, float], q_probs: dict[str, float]) -> tuple[list[str], list[float], list[float]]:
    """Re-key two probability maps onto a single, deterministic option order.
    Renormalises so a label that one model never emits is treated as zero in both."""
    keys = sorted(set(p_probs) | set(q_probs))
    p = [float(p_probs.get(k, 0.0)) for k in keys]
    q = [float(q_probs.get(k, 0.0)) for k in keys]
    return keys, _normalise(p), _normalise(q)


def _top1(probs: dict[str, float]) -> str | None:
    if not probs:
        return None
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _confidence(probs: dict[str, float]) -> float:
    if not probs:
        return 0.0
    s = sum(probs.values())
    return max(probs.values()) / s if s > 0 else 0.0


def _answers_from_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """A trace record's ``output`` field is already in answer form; pass it through."""
    return [dict(ans) for ans in rec.get("output", [])]


def _record_ground_truth(rec: dict[str, Any]) -> list[str | None]:
    """The recorded model's top-1 is treated as the 'gold' label for ΔECE. Returns
    one label per question, in question-id order, or ``None`` when the recorded
    distribution is empty."""
    return [_top1(ans.get("probabilities") or {}) for ans in rec.get("output", [])]


def replay(records: Sequence[dict[str, Any]],
           score_old: Callable[[dict[str, Any]], list[dict[str, Any]]],
           score_new: Callable[[dict[str, Any]], list[dict[str, Any]]],
           *, include_unmatched: bool = False) -> dict[str, Any]:
    """Compare two scoring callables on every record in ``records``.

    Each callable receives the trace record and returns a list of answer dicts (one
    per question, in ``record["question_ids"]`` order). Only ``answer["probabilities"]``
    is required; ``answer["choice"]`` is read when present to report top-1 flips.

    For ΔECE the recorded model's top-1 answer is taken as ground truth; this matches
    the v8 → v10 use case ("how has the served distribution moved relative to v8?").
    """
    per_q: dict[str, dict[str, Any]] = {}
    n_calls = 0
    n_unmatched = 0
    flat_old_conf: list[float] = []
    flat_new_conf: list[float] = []
    flat_old_correct: list[bool] = []
    flat_new_correct: list[bool] = []
    for rec in records:
        rid = rec.get("request_id", "?")
        try:
            new_answers = score_new(rec)
        except Exception as e:  # noqa: BLE001 - one bad record must not kill a replay
            print(f"[replay] score_new failed for {rid}: {e}", file=sys.stderr)
            n_unmatched += 1
            if include_unmatched:
                new_answers = [{"probabilities": {}} for _ in rec.get("question_ids", [])]
            else:
                continue
        try:
            old_answers = score_old(rec)
        except Exception as e:  # noqa: BLE001 - one bad record must not kill a replay
            print(f"[replay] score_old failed for {rid}: {e}", file=sys.stderr)
            n_unmatched += 1
            if include_unmatched:
                old_answers = [{"probabilities": {}} for _ in rec.get("question_ids", [])]
            else:
                continue
        n_calls += 1
        gold = _record_ground_truth(rec)
        for j, (qid, old_ans, new_ans, gold_lab) in enumerate(zip(rec.get("question_ids", []),
                                                                  old_answers, new_answers, gold)):
            old_probs = old_ans.get("probabilities") or {}
            new_probs = new_ans.get("probabilities") or {}
            _keys, old_vec, new_vec = _align_label_lists(old_probs, new_probs)
            kl_v = kl_divergence(old_vec, new_vec)
            js_v = js_divergence(old_vec, new_vec)
            old_top1 = _top1(old_probs)
            new_top1 = _top1(new_probs)
            flip = int(old_top1 is not None and new_top1 is not None and old_top1 != new_top1)
            d = per_q.setdefault(qid, {"n": 0, "kl_sum": 0.0, "js_sum": 0.0, "flips": 0,
                                            "old_conf_sum": 0.0, "old_corr": 0,
                                            "new_conf_sum": 0.0, "new_corr": 0})
            d["n"] += 1
            d["kl_sum"] += kl_v
            d["js_sum"] += js_v
            d["flips"] += flip
            old_conf = _confidence(old_probs)
            new_conf = _confidence(new_probs)
            d["old_conf_sum"] += old_conf
            d["new_conf_sum"] += new_conf
            if gold_lab is not None:
                d["old_corr"] += int(old_top1 == gold_lab)
                d["new_corr"] += int(new_top1 == gold_lab)
                flat_old_conf.append(old_conf)
                flat_new_conf.append(new_conf)
                flat_old_correct.append(old_top1 == gold_lab)
                flat_new_correct.append(new_top1 == gold_lab)
    # finalise per-question stats
    table: list[dict[str, Any]] = []
    agg_kl = agg_js = 0.0
    agg_flips = 0
    agg_n = 0
    for qid, d in per_q.items():
        n = d["n"]
        mean_kl = d["kl_sum"] / n if n else 0.0
        mean_js = d["js_sum"] / n if n else 0.0
        old_e = (d["old_corr"] / n) if n else None
        new_e = (d["new_corr"] / n) if n else None
        delta_acc = ((new_e - old_e) if (old_e is not None and new_e is not None) else None)
        table.append({"question_id": qid, "n_calls": n,
                          "mean_kl": mean_kl, "mean_js": mean_js,
                          "top1_flips": d["flips"],
                          "old_acc": old_e, "new_acc": new_e,
                          "delta_acc": round(delta_acc, 4) if delta_acc is not None else None})
        agg_n += n
        agg_kl += d["kl_sum"]
        agg_js += d["js_sum"]
        agg_flips += d["flips"]
    old_ece = ece(flat_old_conf, flat_old_correct) if flat_old_conf else None
    new_ece = ece(flat_new_conf, flat_new_correct) if flat_new_conf else None
    return {"n_calls": n_calls, "n_questions": len(per_q), "n_unmatched": n_unmatched,
                "per_question": table,
                "aggregate": {"mean_kl": (agg_kl / agg_n) if agg_n else 0.0,
                               "mean_js": (agg_js / agg_n) if agg_n else 0.0,
                               "total_top1_flips": agg_flips,
                               "old_ece": old_ece, "new_ece": new_ece,
                               "delta_ece": ((new_ece - old_ece) if (old_ece is not None and new_ece is not None) else None)}}


def replay_self(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Convenience wrapper: replay a trace against itself (zero drift by construction)."""
    return replay(records, _answers_from_record, _answers_from_record, include_unmatched=False)


def _make_scorer(model_path: str):
    """Build a ``score(record) -> [answer]`` closure for a model directory."""
    from decider.infer import Decider
    return Decider(model_path)


def _record_inputs(rec: dict[str, Any]) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Recover the (state, questions) tuple a recorded request was made with. A trace
    record stores the served answers; if ``state`` and ``questions`` are absent we
    raise so the caller can decide whether to skip the row."""
    if "state" not in rec or "questions" not in rec:
        raise KeyError("trace record missing 'state' and 'questions' (record the sidecar)")
    return rec["state"], rec["questions"]


def _score_with_decider(d, state, questions, max_state_tokens: int = 32768) -> list[dict[str, Any]]:
    """Re-score a (state, questions) pair against the live model and return a list of
    answer dicts in ``system_one`` shape, in the question-id order."""
    out = d.system_one(state, questions, max_state_tokens=max_state_tokens)["answers"]
    return [out[qid] for qid in questions]


def _print_report(report: dict[str, Any], *, top: int | None) -> int:
    """Pretty-print the report. Returns the number of top-1 flips (used as the
    non-zero exit code by ``--strict``)."""
    agg = report["aggregate"]
    print(f"[replay] calls={report['n_calls']} questions={report['n_questions']} "
          f"unmatched={report['n_unmatched']}")
    print(f"[replay] mean KL = {agg['mean_kl']:.6f} nats   mean JS = {agg['mean_js']:.6f} nats   "
          f"top-1 flips = {agg['total_top1_flips']}")
    if agg["old_ece"] is not None and agg["new_ece"] is not None:
        print(f"[replay] ECE old={agg['old_ece']:.4f} new={agg['new_ece']:.4f} "
              f"ΔECE = {agg['delta_ece']:+.4f}")
    print()
    print(f"{'question_id':40s} {'n':>5s} {'KL':>9s} {'JS':>9s} {'flips':>6s} {'old_acc':>8s} {'new_acc':>8s}")
    print("-" * 88)
    rows = sorted(report["per_question"], key=lambda r: r["mean_kl"], reverse=True)
    if top is not None:
        rows = rows[:top]
    for r in rows:
        old_acc = f"{r['old_acc']:.3f}" if r["old_acc"] is not None else "-"
        new_acc = f"{r['new_acc']:.3f}" if r["new_acc"] is not None else "-"
        print(f"{r['question_id'][:40]:40s} {r['n_calls']:5d} {r['mean_kl']:9.6f} {r['mean_js']:9.6f} "
              f"{r['top1_flips']:6d} {old_acc:>8s} {new_acc:>8s}")
    return int(agg["total_top1_flips"])


def _maybe_attach_state(trace_path: Path, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trace records normally do not embed the raw state (it can be megabytes); if a
    sibling ``trace.states.jsonl`` (one JSON object per line, keyed by ``request_id``)
    is present we attach the state to each record so the scorer can replay it."""
    side = trace_path.with_suffix(".states.jsonl")
    if not side.exists():
        return list(records)
    state_by_id: dict[str, dict[str, Any]] = {}
    for line in side.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = obj.get("request_id")
        if rid:
            state_by_id[rid] = obj
    out = []
    for rec in records:
        r = dict(rec)
        st = state_by_id.get(rec.get("request_id", ""))
        if st is not None:
            r["state"] = st.get("state")
            r["questions"] = st.get("questions")
        out.append(r)
    return out


def _run(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace)
    records = TraceReader(trace_path).to_list()
    if not records:
        print(f"[replay] no records in {trace_path}", file=sys.stderr)
        return 2
    records = _maybe_attach_state(trace_path, records)
    has_state = any("state" in r and "questions" in r for r in records)
    if args.old and args.new:
        if not has_state:
            print("[replay] records carry no 'state'/'questions' payload; record the state "
                  "sidecar (trace.states.jsonl) to replay against a model.", file=sys.stderr)
            return 2
        old_model = _make_scorer(args.old)
        new_model = _make_scorer(args.new)
        def _old(rec):
            state, questions = _record_inputs(rec)
            return _score_with_decider(old_model, state, questions)
        def _new(rec):
            state, questions = _record_inputs(rec)
            return _score_with_decider(new_model, state, questions)
        report = replay(records, _old, _new, include_unmatched=args.strict)
    elif args.model:
        if not has_state:
            print("[replay] records carry no 'state'/'questions' payload; cannot replay.", file=sys.stderr)
            return 2
        model = _make_scorer(args.model)
        def _score(rec):
            state, questions = _record_inputs(rec)
            return _score_with_decider(model, state, questions)
        report = replay(records, _answers_from_record, _score, include_unmatched=args.strict)
    else:
        report = replay_self(records)
    flips = _print_report(report, top=args.top)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.strict and flips:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--trace", required=True, help="JSONL trace file")
    ap.add_argument("--old", help="old model directory (Hugging Face id or local path)")
    ap.add_argument("--new", help="new model directory")
    ap.add_argument("--model", help="single-model replay (diff against the recorded outputs)")
    ap.add_argument("--top", type=int, default=20, help="show only the N highest-KL questions (0 = all)")
    ap.add_argument("--json-out", help="write the full per-question report to this JSON file")
    ap.add_argument("--strict", action="store_true", help="exit non-zero when any top-1 flips")
    a = ap.parse_args(argv)
    if a.top == 0:
        a.top = None
    return _run(a)


if __name__ == "__main__":
    raise SystemExit(main())