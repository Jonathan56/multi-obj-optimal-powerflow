import pytest

from moopf import scenario
from moopf import losses
from moopf import cost


@pytest.fixture(scope='session')
def inputs():
    # Network, active profiles, reactive profiles, active net loads, battery specs
    net, dfp, dfq, df, batteries = scenario.get_network_and_timeseries(path="../")
    return net, dfp, dfq, df, batteries


def test_scenario_exists(inputs):
    net, dfp, dfq, df, batteries = inputs
    assert len(df) >= 0


def test_winter_with_binaries(inputs):  # Binary model
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.include_binary_constraint = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns})

    assert round(model.actual_cost, 3) == 602.638  # quite bigger (592.756)
    assert round(model.total_losses, 3) == 51.926
    assert round(model.vm_square_error, 3) == 0.628
    assert round(model.max_upper_demand, 1) == 82.8
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert model.is_complement_battery
    assert round(model.complement_battery_product, 6) == 0.0


def test_winter(inputs):  # (normal case)
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns})

    assert round(model.actual_cost, 3) == 592.757
    assert round(model.total_losses, 3) == 51.971
    assert round(model.vm_square_error, 3) == 0.628
    assert round(model.max_upper_demand, 1) == 82.4
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert model.is_complement_battery
    assert round(model.complement_battery_product, 6) == 0.002682

    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()
    score = cost.Score(net)
    score.score(p_kw, q_kvar, model.schedules["schedules"])
    assert pytest.approx(score.actual_cost, abs=1e-6) == model.actual_cost
    assert round(score.total_losses, 3) == round(model.total_losses, 3)
    assert round(score.vm_below_percent, 3) == round(model.vm_below_percent, 3)
    assert round(score.vm_square_error, 3) == 0.552  # 0.628
    assert round(score.max_upper_demand, 1) == round(model.max_upper_demand, 1)
    assert round(score.max_back_feed, 1) == round(model.max_back_feed, 1)


def test_winter_col(inputs):  # (normal case but collective cost)
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.solve_with_community_cost = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns})

    assert round(model.actual_cost, 3) == 576.307
    assert round(model.total_losses, 3) == 51.971
    assert round(model.vm_square_error, 3) == 0.628
    assert round(model.max_upper_demand, 1) == 82.4
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert model.is_complement_battery
    assert round(model.complement_battery_product, 6) == 0.002682


def test_winter_no_efficiency(inputs):  # No efficiency in winter (to check impact on result)
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.remove_battery_efficiency = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns})

    assert round(model.actual_cost, 3) == 590.807  # 592.756 (similar)
    assert round(model.total_losses, 3) == 50.839  # 51.971 (larger batteries => lower losses)
    assert round(model.vm_square_error, 3) == 0.612  # 0.628
    assert round(model.max_upper_demand, 1) == 73.1  # 82.4 (!!)
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert not model.is_complement_battery
    assert round(model.complement_battery_product, 3) == 807.866


def test_summer_no_efficiency(inputs):  # No efficiency in the summer necessary to avoid binary constraints
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=False)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.remove_battery_efficiency = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns})

    assert round(model.actual_cost, 3) == 133.99
    assert round(model.total_losses, 3) == 10.704
    assert round(model.vm_square_error, 3) == 0.081
    assert round(model.max_upper_demand, 1) == 25.0
    assert round(model.max_back_feed, 1) == -18.8
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert not model.is_complement_battery  # It's ok since battery_in _out are always together with coef=1
    assert round(model.complement_battery_product, 3) == 927.722

    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()
    score = cost.Score(net)
    score.score(p_kw, q_kvar, model.schedules["schedules"])
    assert score.actual_cost == model.actual_cost
    assert pytest.approx(score.total_losses, abs=1e-3) == model.total_losses
    assert round(score.vm_below_percent, 3) == round(model.vm_below_percent, 3)
    assert round(score.vm_square_error, 3) == 0.072  # 0.079
    assert round(score.max_upper_demand, 1) == round(model.max_upper_demand, 1)
    assert round(score.max_back_feed, 1) == round(model.max_back_feed, 1)
