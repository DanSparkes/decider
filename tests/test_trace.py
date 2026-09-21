"""Tests for ``decider.trace`` (writer, reader, hash helpers) and ``decider.replay``
(metrics, replay-on-itself, round trip via the JSONL file on disk).

The replay-on-itself test enforces the contract that replaying a trace with itself
as both the old and the new model produces zero KL, zero JS, zero top-1 flips, and
zero ΔECE. A live model replay is exercised with a tiny stand-in scorer that
returns the recorded answers; the live ``Decider`` path is exercised in
``examples/trace_replay.py`` where the user's environment has the weights.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from decider.replay import (
    _align_label_lists,
    _normalise,
    _top1,
    ece,
    js_divergence,
    kl_divergence,
    replay,
    replay_self,
)
from decider.trace import (
    REQUIRED_TOP_LEVEL,
    SCHEMA_VERSION,
    TraceReader,
    TraceWriter,
    group_by_state,
    hash_state,
    make_output_records,
    records_match,
)

FIXTURE = Path(__file__).parent / "fixtures" / "example_trace.jsonl"


def _sample_records():
    return [
        {"ts": 1.0, "request_id": "r1", "model": "decider-2b", "layout": "state_first",
             "state_hash": "sha1:abc", "question_ids": ["team", "urgent"], "n_opts": [3, 2],
             "prompt_token_count": 180,
             "output": [
                 {"type": "choice", "choice": "billing", "confidence": 0.77,
                      "probabilities": {"billing": 0.77, "technical": 0.19, "sales": 0.04}},
                 {"type": "noul", "noul": 0.92, "confidence": 0.92,
                      "probabilities": {"no": 0.08, "yes": 0.92}}]},
        {"ts": 2.0, "request_id": "r2", "model": "decider-2b", "layout": "state_first",
             "state_hash": "sha1:abc", "question_ids": ["team", "urgent"], "n_opts": [3, 2],
             "prompt_token_count": 180,
             "output": [
                 {"type": "choice", "choice": "technical", "confidence": 0.61,
                      "probabilities": {"billing": 0.21, "technical": 0.61, "sales": 0.18}},
                 {"type": "noul", "noul": 0.74, "confidence": 0.74,
                      "probabilities": {"no": 0.26, "yes": 0.74}}]},
    ]


def test_required_top_level_is_complete():
    for key in ("ts", "request_id", "model", "layout", "state_hash",
                "question_ids", "n_opts", "prompt_token_count", "output"):
        assert key in REQUIRED_TOP_LEVEL


def test_trace_writer_appends_one_json_object_per_line(tmp_path):
    p = tmp_path / "trace.jsonl"
    w = TraceWriter(p)
    rid = w.log(model="decider-2b", layout="state_first", state_hash="sha1:abc",
                question_ids=["team"], n_opts=[3], prompt_token_count=80,
                output=[{"type": "choice", "choice": "billing", "confidence": 0.77,
                             "probabilities": {"billing": 0.77, "technical": 0.19, "sales": 0.04}}])
    rid2 = w.log(model="decider-2b", layout="state_first", state_hash="sha1:def",
                 question_ids=["urgent"], n_opts=[2], prompt_token_count=70,
                 output=[{"type": "noul", "noul": 0.9, "confidence": 0.9,
                              "probabilities": {"no": 0.1, "yes": 0.9}}])
    w.close()
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["request_id"] == rid and recs[1]["request_id"] == rid2
    assert recs[0]["schema_version"] == SCHEMA_VERSION
    assert all(k in recs[0] for k in REQUIRED_TOP_LEVEL)


def test_trace_writer_survives_partial_crash_via_truncated_last_line(tmp_path):
    """A crash mid-write leaves the trailing line unterminated; the reader must skip
    the partial line rather than raise."""
    p = tmp_path / "trace.jsonl"
    w = TraceWriter(p)
    w.log(model="m", layout="state_first", state_hash="sha1:abc",
          question_ids=["q"], n_opts=[2], prompt_token_count=10,
          output=[{"type": "choice", "choice": "a", "confidence": 0.9,
                       "probabilities": {"a": 0.9, "b": 0.1}}])
    w.flush()
    # simulate a crash that closes mid-line
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": 3.0, "request_id": "r-crash", "mo')     # truncated
    recs = list(TraceReader(p).records())
    assert len(recs) == 1
    assert recs[0]["request_id"] == recs[0]["request_id"]        # any rid value


def test_trace_reader_validates_required_fields(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(json.dumps({"ts": 1.0, "request_id": "r1", "model": "m", "layout": "state_first",
                                 "state_hash": "sha1:abc", "question_ids": ["q"], "n_opts": [2],
                                 "prompt_token_count": 10,
                                 "output": [{"type": "choice", "choice": "a", "confidence": 0.9,
                                              "probabilities": {"a": 0.9, "b": 0.1}}]}) + "\n" +
                json.dumps({"request_id": "r2"}) + "\n",
                encoding="utf-8")
    problems = TraceReader(p).validate()
    assert any("missing 'ts'" in s for s in problems)


def test_hash_state_is_stable_for_equivalent_dicts_and_strings():
    a = hash_state({"a": 1, "b": 2})
    b = hash_state({"b": 2, "a": 1})           # key order must not matter
    c = hash_state("[1, 2]")
    assert a == b and c.startswith("sha1:") and len(a) == 5 + 16


def test_make_output_records_from_system_one_shape():
    answers = {"team": {"type": "choice", "choice": "billing", "confidence": 0.77,
                            "probabilities": {"billing": 0.77, "technical": 0.19, "sales": 0.04}},
               "refund": {"type": "noul", "noul": 0.95, "probabilities": {"no": 0.05, "yes": 0.95}}}
    out = make_output_records(answers)
    assert [r["type"] for r in out] == ["choice", "noul"]
    assert out[0]["choice"] == "billing"
    assert out[1]["noul"] == 0.95
    assert all("probabilities" in r for r in out)


def test_records_match_and_group_by_state():
    recs = _sample_records()
    assert records_match(recs[0], recs[1])
    recs[1]["layout"] = "schema_first"
    assert not records_match(recs[0], recs[1])
    g = group_by_state(recs)
    assert len(g) == 2                                    # layouts differ -> two groups
    recs[1]["layout"] = "state_first"
    recs[1]["state_hash"] = "sha1:xyz"
    assert not records_match(recs[0], recs[1])
    g = group_by_state(recs)
    assert len(g) == 2                                    # state hash differs -> two groups


def test_fixture_loads_and_passes_validation():
    recs = TraceReader(FIXTURE).to_list()
    assert recs and TraceReader(FIXTURE).validate() == []
    fields = {k for k in recs[0]}
    for k in REQUIRED_TOP_LEVEL:
        assert k in fields


# --- replay.py pure-Python metrics ---------------------------------------------

def test_normalise_is_safe_under_zero_sum():
    assert sum(_normalise([0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert _normalise([0.0, 0.0]) == [0.5, 0.5]                # uniform fallback


def test_kl_zero_for_identical_distributions():
    p = [0.1, 0.4, 0.5]
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_is_asymmetric_and_nonzero_for_known_pair():
    p = [0.5, 0.5]
    q = [0.9, 0.1]
    kpq = kl_divergence(p, q)
    kqp = kl_divergence(q, p)
    assert kpq > 0 and kqp > 0
    assert abs(kpq - kqp) > 1e-3


def test_kl_is_infinite_when_p_has_mass_outside_q():
    p = [0.7, 0.3]
    q = [1.0, 0.0]
    assert kl_divergence(p, q) == float("inf")


def test_js_is_symmetric_and_zero_for_identical():
    p = [0.2, 0.3, 0.5]
    q = [0.5, 0.3, 0.2]
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12
    assert js_divergence(p, q) <= math.log(2) + 1e-9


def test_ece_is_zero_for_perfectly_calibrated_series():
    # conf=0.5 with half-correct -> bin accuracy = bin conf -> zero contribution.
    conf = [0.5] * 20
    correct = [True] * 10 + [False] * 10
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)
    # conf=1.0 with all-correct -> bin accuracy 1.0 = conf -> zero contribution.
    conf = [1.0] * 5
    correct = [True] * 5
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)
    # conf=0.0 with all-wrong -> bin accuracy 0.0 = conf -> zero contribution.
    conf = [0.0] * 5
    correct = [False] * 5
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_align_label_lists_renormalises_after_merge():
    keys, p, q = _align_label_lists({"a": 0.6, "b": 0.4}, {"b": 0.5, "c": 0.5})
    assert keys == ["a", "b", "c"]
    assert sum(p) == pytest.approx(1.0)
    assert sum(q) == pytest.approx(1.0)


def test_top1_picks_the_argmax_label():
    assert _top1({"a": 0.1, "b": 0.7, "c": 0.2}) == "b"
    assert _top1({}) is None


# --- replay contract: replay-on-itself is zero drift ---------------------------

def test_replay_self_against_fixture_is_zero_drift():
    recs = TraceReader(FIXTURE).to_list()
    report = replay_self(recs)
    agg = report["aggregate"]
    assert agg["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert agg["mean_js"] == pytest.approx(0.0, abs=1e-9)
    assert agg["total_top1_flips"] == 0
    for row in report["per_question"]:
        assert row["mean_kl"] == pytest.approx(0.0, abs=1e-9)
        assert row["mean_js"] == pytest.approx(0.0, abs=1e-9)
        assert row["top1_flips"] == 0
    # ΔECE is None when no recorded ground truth is reachable; with the fixture
    # the recorded top-1 is reachable so ΔECE must be exactly 0.
    assert agg["old_ece"] is not None and agg["new_ece"] is not None
    assert agg["delta_ece"] == pytest.approx(0.0, abs=1e-9)


def test_replay_detects_a_flipped_top1():
    recs = TraceReader(FIXTURE).to_list()
    def _new(rec):
        ans = []
        for o in rec["output"]:
            a = dict(o)
            probs = dict(o["probabilities"])
            if probs:
                # flip the argmax to the second-largest label
                items = sorted(probs.items(), key=lambda kv: -kv[1])
                _top, second = items[0][0], items[1][0]
                a["probabilities"] = {k: (1.0 if k == second else 0.0) for k in probs}
            ans.append(a)
        return ans
    report = replay(recs, lambda r: list(r["output"]), _new, include_unmatched=False)
    assert report["aggregate"]["total_top1_flips"] >= 1
    assert report["aggregate"]["mean_kl"] > 0


def test_replay_records_kl_is_zero_under_zero_divergence():
    recs = TraceReader(FIXTURE).to_list()
    def _same(rec): return list(rec["output"])
    report = replay(recs, _same, _same)
    assert report["aggregate"]["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert report["aggregate"]["mean_js"] == pytest.approx(0.0, abs=1e-9)
    assert report["aggregate"]["total_top1_flips"] == 0


# --- round-trip through a TraceWriter / TraceReader ----------------------------

def test_writer_reader_round_trip_preserves_schema(tmp_path):
    p = tmp_path / "trace.jsonl"
    w = TraceWriter(p)
    w.log(model="decider-2b", layout="state_first", state_hash="sha1:abc",
          question_ids=["team", "urgent"], n_opts=[3, 2], prompt_token_count=180,
          output=[{"type": "choice", "choice": "billing", "confidence": 0.77,
                       "probabilities": {"billing": 0.77, "technical": 0.19, "sales": 0.04}},
                  {"type": "noul", "noul": 0.92, "confidence": 0.92,
                       "probabilities": {"no": 0.08, "yes": 0.92}}])
    w.close()
    recs = TraceReader(p).to_list()
    assert len(recs) == 1
    r = recs[0]
    assert r["model"] == "decider-2b"
    assert r["state_hash"] == "sha1:abc"
    assert r["question_ids"] == ["team", "urgent"]
    assert r["n_opts"] == [3, 2]
    assert r["output"][0]["choice"] == "billing"
    assert sum(r["output"][0]["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
    assert TraceReader(p).validate() == []