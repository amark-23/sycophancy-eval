from analysis.feedback_metrics import compute_metrics


def _rows(n=40, a=2.0, q=1.0):
    rows = []
    for i in range(n):
        base = 5 + (i % 3) * 0.5  # item-level offset, shared across cells (paired design)
        for v, vq in (("correct", q), ("corrupted", 0.0)):
            for f, fa in (("neutral", 0.0), ("proud", a), ("unhappy", -a)):
                rows.append({"item_id": f"i{i}", "version": v, "frame": f, "phrasing": i % 2, "epoch": 0,
                             "rating": base + vq + fa / 2, "verdict": "correct" if vq + fa > 0 else "incorrect"})
    return rows


def test_known_effects_recovered():
    m = compute_metrics(_rows(a=2.0, q=1.0))
    assert abs(m["A"]["est"] - 2.0) < 1e-9 and abs(m["Q"]["est"] - 1.0) < 1e-9
    assert abs(m["proud - neutral"]["est"] - 1.0) < 1e-9 and abs(m["unhappy - neutral"]["est"] + 1.0) < 1e-9
    assert m["A/Q"] is not None and abs(m["A/Q"]["est"] - 2.0) < 1e-9
    assert m["A"]["n_items"] == 40


def test_ratio_withheld_when_q_is_zero():
    m = compute_metrics(_rows(a=2.0, q=0.0))
    assert m["Q"]["est"] == 0 and m["A/Q"] is None


def test_parse_failures_excluded_not_imputed():
    rows = _rows(n=10)
    for r in rows[:6]:
        r["rating"] = None
    m = compute_metrics(rows)
    assert m["counts"][("neutral", "rating_fail")] + m["counts"][("proud", "rating_fail")] >= 1
    assert m["A"]["n_items"] == 9  # item i0 lost cells, so it drops out of the paired metric


def test_gross_version_reported_separately_and_not_in_primary_A():
    rows = _rows(a=2.0, q=1.0)
    gross = []
    for r in rows:
        if r["version"] == "corrupted":
            g = dict(r)
            g["version"] = "gross"
            g["rating"] = r["rating"] - 3.0
            gross.append(g)
    m = compute_metrics(rows + gross)
    assert abs(m["A"]["est"] - 2.0) < 1e-9 and m["A"]["n_items"] == 40
    assert abs(m["Q_gross"]["est"] - 4.0) < 1e-9 and abs(m["Q"]["est"] - 1.0) < 1e-9
    assert abs(m["A_gross"]["est"] - 2.0) < 1e-9
    assert m["A/Q_gross"] is not None and abs(m["A/Q_gross"]["est"] - 0.5) < 1e-9
