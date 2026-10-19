import math

import pytest

from selfrag.models import Beam, Candidate
from selfrag.scoring.scores import END, s_rel, s_sup, s_use, segment_score, select_beams


def test_s_rel():
    assert s_rel({"Relevant": 0.8, "Irrelevant": 0.2}) == pytest.approx(0.8)


def test_s_sup_plan_example():
    # Fully 0.6 / Partially 0.3 / No 0.1 -> 0.6 + 0.5 * 0.3 = 0.75
    assert s_sup({"Fully": 0.6, "Partially": 0.3, "No": 0.1}) == pytest.approx(0.75)


def test_s_use_weights_from_minus_one_to_one():
    assert s_use({"1": 1, "2": 0, "3": 0, "4": 0, "5": 0}) == pytest.approx(-1.0)
    assert s_use({"1": 0, "2": 0, "3": 0, "4": 0, "5": 1}) == pytest.approx(1.0)
    assert s_use({"1": 0, "2": 0, "3": 0.5, "4": 0.5, "5": 0}) == pytest.approx(0.25)


REL = {"Relevant": 0.8, "Irrelevant": 0.2}
SUP = {"Fully": 0.6, "Partially": 0.3, "No": 0.1}
USE = {"1": 0, "2": 0, "3": 0, "4": 1, "5": 0}


def test_segment_score_weighted_sum():
    c = Candidate("x", None, rel=REL, sup=SUP, use=USE)
    assert segment_score(c) == pytest.approx(1.0 * 0.8 + 1.0 * 0.75 + 0.5 * 0.5)
    assert segment_score(c, w_sup=2.0) == pytest.approx(0.8 + 1.5 + 0.25)


def test_segment_score_generation_probability():
    assert segment_score(Candidate("x", None, gen_logprob=math.log(0.5))) == pytest.approx(0.5)
    # a log-prob of 0.0 is probability 1, not "missing"
    assert segment_score(Candidate("x", None, gen_logprob=0.0)) == pytest.approx(1.0)
    assert segment_score(Candidate("x", None)) == 0.0


def test_hard_constraint_rejects_unsupported():
    c = Candidate("x", None, sup={"Fully": 0.1, "Partially": 0.2, "No": 0.7})
    assert segment_score(c, hard_sup=True) == float("-inf")
    assert segment_score(c, hard_sup=False) > float("-inf")


def test_select_beams_keeps_best_b_with_citations():
    good = Candidate("good", "p1", rel=REL, sup=SUP)
    bad = Candidate("bad", "p2", rel={"Relevant": 0.3, "Irrelevant": 0.7},
                    sup={"Fully": 0, "Partially": 0, "No": 1})
    mid = Candidate("mid", "p3", rel=REL, sup={"Fully": 0, "Partially": 1, "No": 0})
    kept = select_beams([Beam()], [[bad, good, mid]], B=2)
    assert [b.segments for b in kept] == [["good"], ["mid"]]
    assert kept[0].citations == ["p1"] and kept[0].last_passage == "p1"


def test_select_beams_end_finishes_beam_unchanged():
    beam = Beam(["first"], ["p1"], [1.0], ["fully"], last_passage="p1")
    kept = select_beams([beam], [[Candidate(END, "p2")]], B=2)
    assert len(kept) == 1
    assert (kept[0].segments, kept[0].scores, kept[0].supports, kept[0].done) == \
        (["first"], [1.0], ["fully"], True)


def test_select_beams_records_support_label():
    kept = select_beams([Beam()], [[Candidate("x", "p1", sup=SUP), Candidate("y", None)]], B=2)
    assert {b.segments[0]: b.supports for b in kept} == {"x": ["fully"], "y": [None]}


def test_beam_score_is_mean_of_segments():
    assert Beam(["a", "b"], [None, None], [1.0, 2.0]).score == pytest.approx(1.5)
    assert Beam().score == 0.0
