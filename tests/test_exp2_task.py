from tasks.feedback import make_samples, parse_reply, phrasing_for
from tasks.prompts_exp2 import FRAMES, PHRASINGS, prompt_hash, render

REC = {"id": "gsm8k-test-1", "question": "Q?", "solution_correct": "a\nFinal answer: 4", "solution_gross": "a\nFinal answer: 43",
       "solution_corrupted": "a\nFinal answer: 5"}


def test_parse_ok_and_strict():
    assert parse_reply("RATING: 8\nVERDICT: correct") == (8, "correct")
    assert parse_reply("RATING: 3/10\nVERDICT: Incorrect.") == (3, "incorrect")
    assert parse_reply("RATING: 10\nVERDICT: incorrect") == (10, "incorrect")


def test_parse_failures():
    assert parse_reply("RATING: 11\nVERDICT: correct") == (None, "correct")
    assert parse_reply("RATING: 0\nVERDICT: correct")[0] is None
    assert parse_reply("RATING: 7\nRATING: 6\nVERDICT: correct")[0] is None   # conflicting
    assert parse_reply("The rating is 7")[0] is None
    assert parse_reply("RATING: 7\nVERDICT: correct\nVERDICT: incorrect")[1] is None
    assert parse_reply("") == (None, None)


def test_samples_cover_design_and_are_balanced():
    ss = make_samples([dict(REC, id=f"i{k}") for k in range(400)], seed=0)
    assert len(ss) == 400 * 3 * 3
    assert {s.metadata["frame"] for s in ss} == set(FRAMES)
    for f in FRAMES:
        share = sum(s.metadata["phrasing"] for s in ss if s.metadata["frame"] == f) / (400 * 3)
        assert 0.4 < share < 0.6
    assert phrasing_for("x", "proud", 0) == phrasing_for("x", "proud", 0)


def test_render_only_frame_differs_and_no_correctness_claim():
    a, b = render(REC, "correct", "proud", 0), render(REC, "correct", "unhappy", 0)
    assert a.replace(PHRASINGS["proud"][0], "") == b.replace(PHRASINGS["unhappy"][0], "")
    assert "Final answer: 5" in render(REC, "corrupted", "neutral", 1)
    for f in FRAMES:
        for p in PHRASINGS[f]:
            assert not any(w in p.lower() for w in ("correct", "wrong", "mistake", "right"))
    assert len(prompt_hash()) == 12
