import pytest

from moopf import scenario
from moopf import cost


@pytest.fixture(scope='session')
def inputs():
    # Network, active profiles, reactive profiles, active net loads, battery specs
    net, dfp, dfq, df, batteries = scenario.get_network_and_timeseries(path="../")
    return net, dfp, dfq, df, batteries


def test_scenario_exists(inputs):
    net, dfp, dfq, df, batteries = inputs
    assert len(df) >= 0


def test_winter(inputs):
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)

    model = cost.OptimizeCost()
    model.solve(p_kw, {i: dict(batteries) for i in df.columns})
    assert round(model.actual_cost, 3) == 565.563
    assert model.is_complement
    assert round(model.complement_product, 3) == 0.0

    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()
    score = cost.Score(net)
    score.score(p_kw, q_kvar, model.schedules["schedules"])
    assert score.actual_cost == model.actual_cost
    assert round(score.total_losses, 3) == 55.109  # 51.971
    assert round(score.vm_square_error, 3) == 0.609  # 0.628
    assert round(score.max_upper_demand, 1) == 118.7  # 82.4
    assert round(score.max_back_feed, 1) == 0.0  # 0.0


def test_winter_col(inputs):  # Minimize a community cost including grid fees
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)

    model = cost.OptimizeCost()
    model.solve_with_community_cost = True
    model.solve(p_kw, {i: dict(batteries) for i in df.columns})
    assert round(model.actual_cost, 3) == 564.197
    assert model.is_complement
    assert round(model.complement_product, 3) == 0.0

    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()
    score = cost.Score(net)
    score.solve_with_community_cost = True
    score.score(p_kw, q_kvar, model.schedules["schedules"])
    assert pytest.approx(score.actual_cost, abs=1e-6) == model.actual_cost
    assert round(score.total_losses, 3) == 55.169  # 51.971
    assert round(score.vm_square_error, 3) == 0.61  # 0.628
    assert round(score.max_upper_demand, 1) == 118.7  # 82.4
    assert round(score.max_back_feed, 1) == 0.0  # 0.0


def test_summer(inputs):
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=False)

    model = cost.OptimizeCost()
    model.solve(p_kw, {i: dict(batteries) for i in df.columns})
    assert round(model.actual_cost, 3) == 118.763
    assert model.is_complement
    assert round(model.complement_product, 3) == 0.0

    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()
    score = cost.Score(net)
    score.score(p_kw, q_kvar, model.schedules["schedules"])
    assert pytest.approx(score.actual_cost, abs=1) == model.actual_cost
    assert round(score.total_losses, 3) == 12.862  # 9.957
    assert round(score.vm_square_error, 3) == 0.102  # 0.079
    assert round(score.max_upper_demand, 1) == 46.9  # 25.0
    assert round(score.max_back_feed, 1) == -36.2  # -16.1


def test_summer_col(inputs):  # Minimize a community cost including grid fees
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=False)

    model = cost.OptimizeCost()
    model.solve_with_community_cost = True
    model.solve(p_kw, {i: dict(batteries) for i in df.columns})
    assert round(model.actual_cost, 3) == 95.873
    assert model.is_complement
    assert round(model.complement_product, 3) == 0.0