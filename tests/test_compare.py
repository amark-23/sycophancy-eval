from analysis.compare import restrict, shared_items


def test_shared_items_is_intersection_over_both_conditions():
    a = {"wrong_mild": {"1": 1.0, "2": 0.0, "3": 1.0}, "wrong_strong": {"1": 1.0, "2": 1.0}}
    b = {"wrong_mild": {"1": 0.0, "2": 1.0}, "wrong_strong": {"1": 0.0, "2": 0.0, "3": 1.0}}
    assert shared_items({"a": a, "b": b}) == {"1", "2"}
    assert restrict(a["wrong_mild"], {"1", "2"}) == {"1": 1.0, "2": 0.0}


def test_shared_items_empty():
    assert shared_items({}) == set()
