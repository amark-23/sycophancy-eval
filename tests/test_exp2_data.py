from data.exp2 import corrupt, parse_answer, process, render_solution, strip_annotations

EX = ("Ann has 5 apples and buys 3 more.\n"
      "She has 5+3=<<5+3=8>>8 apples.\n"
      "Half are red: 8/2=<<8/2=4>>4 red apples.\n"
      "Together with 8 total, 4 are red.\n#### 4")


def test_parse_and_strip():
    lines, final = parse_answer(EX)
    assert final == 4 and len(lines) == 4
    assert strip_annotations(lines[1]) == "She has 5+3=8 apples."


def test_parse_rejects_non_integer_or_missing():
    assert parse_answer("x\n#### 3.5")[1] is None
    assert parse_answer("no marker")[1] is None
    assert parse_answer("x\n#### 1,200")[1] == 1200


def test_corrupt_changes_last_step_and_restatements_only():
    lines, final = parse_answer(EX)
    out, wrong = corrupt(lines, final, seed=0, key="k")
    assert wrong != final and wrong >= 0
    assert "8/2=" in out[2] and str(wrong) in out[2]          # last calculation result replaced
    assert out[1] == "She has 5+3=8 apples."                  # earlier step untouched
    assert "4 are red" not in out[3] and str(wrong) in out[3]  # later restatement follows
    assert "<<" not in "".join(out)


def test_corrupt_is_deterministic_and_seeded():
    lines, final = parse_answer(EX)
    assert corrupt(lines, final, 0, "k") == corrupt(lines, final, 0, "k")


def test_corrupt_drops_when_no_annotation_produces_final():
    lines, final = parse_answer("She counts them.\n#### 7")
    assert corrupt(lines, final, 0, "k") is None


def test_corrupt_handles_commas():
    lines, final = parse_answer("He earns 1,000*2=<<1000*2=2000>>2,000 dollars.\n#### 2000")
    out, wrong = corrupt(lines, final, 0, "k")
    assert "2,000" not in out[0] and f"{wrong:,}" in out[0]


def test_process_split_report_and_render():
    raw = [{"question": f"q{i}", "answer": EX} for i in range(20)]
    raw += [{"question": "q0", "answer": EX}, {"question": "", "answer": EX},
            {"question": "bad", "answer": "no marker"}]
    out, rep = process(raw, seed=0, dev_fraction=0.3)
    assert rep["kept"] == 20 and rep["dropped_empty_or_duplicate"] == 2
    assert rep["dropped_unparseable_or_non_integer"] == 1
    assert len(out["dev"]) + len(out["heldout"]) == 20
    r = (out["dev"] + out["heldout"])[0]
    assert r["solution_correct"].endswith("Final answer: 4")
    assert r["solution_corrupted"].endswith(f"Final answer: {r['final_corrupted']}")
    out2, _ = process(raw, seed=0, dev_fraction=0.3)
    assert out == out2
    assert render_solution(["a"], 1) == "a\nFinal answer: 1"


def test_gross_corruption_is_large_and_consistent():
    lines, final = parse_answer(EX)
    out, wrong = corrupt(lines, final, 0, "k", kind="gross")
    assert wrong == 43 and "8/2=43" in out[2] and "4 are red" not in out[3].replace("43 are red", "") and "43 are red" in out[3]
    recs, _ = process([{"question": "q", "answer": EX}], seed=0, dev_fraction=0.3)
    r = (recs["dev"] + recs["heldout"])[0]
    assert r["final_gross"] == 43 and r["solution_gross"].endswith("Final answer: 43")
    assert r["solution_gross"] != r["solution_corrupted"] != r["solution_correct"]
