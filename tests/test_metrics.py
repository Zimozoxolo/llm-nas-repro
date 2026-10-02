import numpy as np

from llmnas.metrics import FrontEvaluator, hypervolume_2d, igd, objectives, pareto_mask


def test_pareto_mask():
    pts = np.array([[1, 3], [2, 2], [3, 1], [2, 3], [3, 3], [1, 3]])
    assert pareto_mask(pts).tolist() == [True, True, True, False, False, False]


def test_hypervolume_known_value():
    pts = np.array([[0.0, 0.5], [0.5, 0.0]])
    assert np.isclose(hypervolume_2d(pts, np.array([1.0, 1.0])), 0.5 * 1 + 0.5 * 0.5)


def test_igd_zero_on_true_front():
    f = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert igd(f, f) == 0.0


def test_evaluator_true_front_scores_one():
    rng = np.random.default_rng(0)
    lat, acc = rng.random(500) * 10, rng.random(500) * 100
    ev = FrontEvaluator(objectives(lat, acc))
    s = ev.score(ev.true_front)
    assert np.isclose(s["hv"], 1.0) and np.isclose(s["igd"], 0.0)
    worse = ev.score(ev.true_front[::3])
    assert worse["hv"] < 1.0 and worse["igd"] > 0.0


def test_unmeasured_points_ignored():
    pts = np.array([[np.nan, -90.0], [1.0, -50.0], [2.0, -60.0]])
    assert pareto_mask(pts).tolist() == [False, True, True]
    ev = FrontEvaluator(pts)
    assert len(ev.true_front) == 2 and ev.score(pts)["hv"] == 1.0
