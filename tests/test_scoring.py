import math

import pandas as pd

from phi_deid.scoring import is_match, match_record, outcome_rows, prf, privacy_rates

GOLD = {"label": "NAME", "start": 4, "end": 13, "text": "Dr. Lopez", "asq_type": "NAME"}


def pred(start, end, label="NAME"):
    return {"type": label, "start": start, "end": end, "text": "x", "grounded": True}


def test_strict_needs_exact_boundaries():
    assert is_match(GOLD, pred(4, 13), "strict")
    assert not is_match(GOLD, pred(8, 13), "strict")


def test_overlap_tolerates_boundaries_but_not_label():
    assert is_match(GOLD, pred(8, 13), "overlap")
    assert not is_match(GOLD, pred(8, 13, "DATE"), "overlap")


def test_any_ignores_label():
    assert is_match(GOLD, pred(8, 13, "DATE"), "any")
    assert not is_match(GOLD, pred(20, 25), "any")


def test_unaligned_spans_never_match():
    assert not is_match(GOLD, pred(None, None), "any")


def test_match_is_one_to_one_and_prefers_largest_overlap():
    preds = [pred(4, 7), pred(4, 13)]
    assert match_record([GOLD], preds, "overlap") == {0: 1}


def test_prf_and_privacy_rates():
    rows = (outcome_rows("c", "q1", [GOLD], [pred(4, 13), pred(20, 25)])
            + outcome_rows("c", "q2", [], [pred(0, 3)])
            + outcome_rows("c", "q3", [], []))
    frame = pd.DataFrame(rows)
    scores = prf(frame, "strict")
    assert (scores["TP"], scores["FP"], scores["FN"]) == (1, 2, 0)
    assert scores["recall"] == 1.0 and math.isclose(scores["precision"], 0.3333)
    assert privacy_rates(frame, {"q1"}, {"q2", "q3"}) == (1.0, 0.5)
