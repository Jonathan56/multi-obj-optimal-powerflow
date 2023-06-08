import pytest

from simu import scenario
from simu import losses


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
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.MIPGap = 1e-2
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns}, 580, 565.563)

    assert round(model.actual_cost, 3) == 580.058
    assert round(model.total_losses, 3) == 51.98
    assert round(model.vm_square_error, 3) == 0.629
    assert round(model.max_upper_demand, 1) == 82.8
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert round(model.complement_battery_product, 3) == 0.0


def test_winter_col(inputs):  # Impose a constraint on collective cost in the losses minimization
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=True)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.solve_with_community_cost = True
    model.MIPGap = 1e-2
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns}, 567, 564.197)

    assert round(model.actual_cost, 3) == 567.057
    assert round(model.total_losses, 3) == 52.382
    assert round(model.vm_square_error, 3) == 0.637
    assert round(model.max_upper_demand, 1) == 97.0
    assert round(model.max_back_feed, 1) == 0.0
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert round(model.complement_battery_product, 3) == 0.0
    assert round(model.complement_netload_product, 3) == 0.0
    assert round(model.distance_to_minimum_cost, 3) == 2.803


def test_summer(inputs):
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=False)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.remove_battery_efficiency = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns}, 120, 118.763)

    assert round(model.actual_cost, 3) == 120.012
    assert round(model.total_losses, 3) == 10.715
    assert round(model.vm_square_error, 3) == 0.082
    assert round(model.max_upper_demand, 1) == 25.0
    assert round(model.max_back_feed, 1) == -18.9
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert round(model.complement_battery_product, 3) == 904.938


def test_summer_col(inputs):  # Impose a constraint on collective cost in the losses minimization
    net, dfp, dfq, df, batteries = inputs
    p_kw = scenario.get_day_from_consumption_level(df, highest_consumption=False)
    q_kvar = dfq.loc[p_kw.index[0]:p_kw.index[-1], :].copy()

    model = losses.OptimizeLosses()
    model.solve_with_community_cost = True
    model.remove_battery_efficiency = True
    model.MIPGap = 1e-4
    model.TimeLimit = 300
    model.solve(net, p_kw, q_kvar, {i: dict(batteries) for i in df.columns}, 100, 95.873)

    assert round(model.actual_cost, 3) == 100.01
    assert round(model.total_losses, 3) == 10.704
    assert round(model.vm_square_error, 3) == 0.081
    assert round(model.max_upper_demand, 1) == 25.0
    assert round(model.max_back_feed, 1) == -18.8
    assert model.is_l_binding_constraint
    assert model.is_voltage_in_range_power_flow
    assert round(model.complement_battery_product, 3) == 946.088
    assert round(model.complement_netload_product, 3) == 0.123
    assert round(model.distance_to_minimum_cost, 3) == 4.004
