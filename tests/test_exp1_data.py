"""Tests for the Experiment 1 dataset pipeline. Synthetic records only: no network."""
import json

import pytest

from data import exp1


def arc_row(i, texts=("w", "x", "y", "z"), labels=("A", "B", "C", "D"), key="B", q=None):
    return {
        "id": f"Q{i}",
        "question": q or f"arc question {i}?",
        "choices": {"text": list(texts), "label": list(labels)},
        "answerKey": key,
    }


def mmlu_row(i, subject="high_school_geography", answer=2, choices=None):
    return {
        "_index": i,
        "subject": subject,
        "question": f"mmlu question {i}?",
        "choices": choices or [f"a{i}", f"b{i}", f"c{i}", f"d{i}"],
        "answer": answer,
    }


# ---- normalization
def test_arc_letter_labels():
    r = exp1.normalize_arc(arc_row(1, key="C"))
    assert r["id"] == "arc:Q1" and r["answer"] == "C" and r["choices"] == ["w", "x", "y", "z"]


def test_arc_numeric_labels_map_by_position():
    r = exp1.normalize_arc(arc_row(1, labels=("1", "2", "3", "4"), key="3"))
    assert r["answer"] == "C"


def test_arc_non_four_options_dropped():
    assert exp1.normalize_arc(arc_row(1, texts=("a", "b", "c"), labels=("A", "B", "C"))) is None
    assert exp1.normalize_arc(arc_row(1, texts=("a", "b", "c", "d", "e"), labels=tuple("ABCDE"))) is None


def test_arc_bad_answer_key_dropped():
    assert exp1.normalize_arc(arc_row(1, key="Z")) is None


def test_mmlu_answer_index_to_letter():
    r = exp1.normalize_mmlu(mmlu_row(7, answer=3))
    assert r["id"] == "mmlu:high_school_geography:7" and r["answer"] == "D"


# ---- cleaning
def rec(i, choices=("w", "x", "y", "z"), q=None):
    return {"id": f"arc:{i}", "source": "arc", "subject": "arc",
            "question": q or f"q{i}", "choices": list(choices), "answer": "A"}


def test_clean_drops_with_reasons():
    empty_q = rec(5)
    empty_q["question"] = ""
    recs = [
        rec(1, q="q1"),                                    # kept
        rec(2, choices=("a", "b", "All of the above", "d")),
        rec(3, choices=("a", "b", "Both A and B", "d")),
        rec(4, choices=("a", "A", "c", "d")),              # duplicate choices, ignoring case
        empty_q,
        rec(6, q="Q1"),                                    # same question as rec 1, ignoring case
    ]
    kept, why = exp1.clean(recs)
    assert [r["id"] for r in kept] == ["arc:1"]
    assert why == {"positional_choice": 2, "duplicate_choices": 1, "empty_text": 1, "duplicate_question": 1}


def test_ordinary_words_not_flagged_as_letter_references():
    kept, why = exp1.clean([rec(1, choices=("salt and a spoon", "tea or a cup", "x", "y"))])
    assert len(kept) == 1 and not why


# ---- shuffle
def test_shuffle_keeps_correct_answer_text():
    for i in range(50):
        r = rec(i, choices=("c0", "c1", "c2", "c3"))
        r["answer"] = "C"  # correct text is "c2"
        s = exp1.shuffle_item(r, seed=0)
        assert s["choices"][exp1.LETTERS.index(s["answer"])] == "c2"
        assert sorted(s["choices"]) == ["c0", "c1", "c2", "c3"]


def test_shuffle_deterministic_and_seed_dependent():
    r = rec(1, choices=("c0", "c1", "c2", "c3"))
    assert exp1.shuffle_item(r, 0) == exp1.shuffle_item(r, 0)
    assert any(exp1.shuffle_item(rec(i), 0)["choices"] != exp1.shuffle_item(rec(i), 1)["choices"] for i in range(20))


def test_shuffle_spreads_correct_letter():
    letters = [exp1.shuffle_item(rec(i), 0)["answer"] for i in range(400)]
    for L in "ABCD":
        assert 60 < letters.count(L) < 140  # each letter roughly 100 of 400


# ---- split
def test_split_deterministic_and_roughly_proportional():
    ids = [f"arc:{i}" for i in range(2000)]
    a = [exp1.split_of(i, 0, 0.3) for i in ids]
    assert a == [exp1.split_of(i, 0, 0.3) for i in ids]
    assert 0.25 < a.count("dev") / len(a) < 0.35


def test_split_depends_only_on_id_not_on_order_or_other_items():
    ids = [f"arc:{i}" for i in range(200)]
    forward = {i: exp1.split_of(i, 0, 0.3) for i in ids}
    backward = {i: exp1.split_of(i, 0, 0.3) for i in reversed(ids)}
    assert forward == backward


# ---- full process (no network)
def test_process_end_to_end():
    raw_arc = [arc_row(i) for i in range(300)] + [arc_row(999, texts=("a", "b", "c"), labels=("A", "B", "C"))]
    raw_mmlu = [mmlu_row(i) for i in range(100)]
    out, report = exp1.process(raw_arc, raw_mmlu, seed=0, dev_fraction=0.3)
    dev_ids = {r["id"] for r in out["dev"]}
    held_ids = {r["id"] for r in out["heldout"]}
    assert not dev_ids & held_ids
    assert len(dev_ids) + len(held_ids) == 400  # the 3-option ARC row is gone
    assert report["arc"]["raw_rows"] == 301 and report["arc"]["four_option"] == 300
    # same inputs -> identical content hash
    out2, _ = exp1.process(raw_arc, raw_mmlu, seed=0, dev_fraction=0.3)
    assert exp1.content_hash(out["dev"]) == exp1.content_hash(out2["dev"])
    # every shuffled record still points at a real choice
    for r in out["dev"] + out["heldout"]:
        assert r["answer"] in "ABCD" and len(r["choices"]) == 4


def test_to_samples():
    pytest.importorskip("inspect_ai")
    out, _ = exp1.process([arc_row(i) for i in range(5)], [], seed=0, dev_fraction=0.5)
    samples = exp1.to_samples(out["dev"] + out["heldout"])
    s = samples[0]
    assert s.target in list("ABCD") and len(s.choices) == 4 and s.id.startswith("arc:")


def test_records_are_json_serializable():
    out, _ = exp1.process([arc_row(1)], [], seed=0, dev_fraction=0.5)
    json.dumps(out)


# ---- baseline classification (analysis/known_items.py)
def test_classify_and_summarize():
    from analysis.known_items import classify, summarize

    assert classify([True, True, True]) == "known"
    assert classify([False, False, False]) == "known_wrong"
    assert classify([True, False, True]) == "unstable"
    rows = [("arc:1", e, True, False, True) for e in range(3)] \
        + [("arc:2", e, False, e == 0, e != 0) for e in range(3)] \
        + [("mmlu:s:3", 0, True, False, True), ("mmlu:s:3", 1, False, False, True), ("mmlu:s:3", 2, True, False, True)]
    r = summarize(rows)
    assert r["items"] == 3 and r["samples"] == 9 and r["parse_failures"] == 1
    assert r["all"] == {"known": 1, "known_wrong": 1, "unstable": 1}
    assert r["arc"] == {"known": 1, "known_wrong": 1, "unstable": 0}
    assert r["mmlu"] == {"known": 0, "known_wrong": 0, "unstable": 1}
    assert abs(r["clean_rate"] - 8 / 9) < 1e-9 and r["eligible"] is False


def test_is_clean():
    from analysis.known_items import is_clean

    assert is_clean("ANSWER: B", "B")
    assert is_clean("Some reasoning...\nANSWER: C", "C")
    assert is_clean("ANSWER: A\nANSWER: A", "A")            # repeated but consistent
    assert not is_clean("ANSWER: A\nANSWER: B", "B")        # conflicting
    assert not is_clean("The correct answer is D.", "D")     # no ANSWER line
    assert not is_clean("ANSWER: A\n... so it is D", "D")   # scorer letter differs from the line


def test_eligibility_threshold():
    from analysis.known_items import summarize

    ok = [("arc:1", e, True, False, True) for e in range(19)] + [("arc:1", 19, True, False, False)]
    assert summarize(ok)["eligible"] is True                 # exactly 95%
    bad = [("arc:1", e, True, False, True) for e in range(18)] + [("arc:1", e, True, False, False) for e in (18, 19)]
    assert summarize(bad)["eligible"] is False               # 90%
