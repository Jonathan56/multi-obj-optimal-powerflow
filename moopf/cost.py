import copy

import pytest
import pandas as pd
import pandapower as pp
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


class OptimizeCost(object):
    """
    Linear program to minimize costs of energy when centrally controlling a fleet of batteries.

    Parameters:
    -----------
    * solve_with_community_cost: instead of minimizing the sum of individual costs, it minimizes a community cost.
        This community cost include grid fees to exchange power between community members.

    * grid_buy, grid_sell, & grid_fee: to modify prices (scalar).

    * max_participant_netload: to ensure that fix cost from subscribed power is respected.

    * freq: granularity of the timeseries (default 15 minutes).

    Results:
    --------
    * schedules: it includes all decision variables from the optimization problem (battery_energy)

    * actual_cost: resulting energy cost in Euros from the optimization
    """

    def __init__(self, solve_with_community_cost=False,
                 grid_buy=0.2276, grid_sell=0.13, grid_fee=0.0315, max_participant_netload=12,
                 remove_battery_efficiency=False, lower_battery_capacity_by=1.0,
                 lower_battery_power_by=1.0, **_):

        # Optimization model
        self.grid_buy = grid_buy
        self.grid_sell = grid_sell
        self.grid_fee = grid_fee
        self.freq: int = 15
        self.max_participant_netload = max_participant_netload
        self.solve_with_community_cost = solve_with_community_cost
        self.remove_battery_efficiency = remove_battery_efficiency
        self.lower_battery_capacity_by = lower_battery_capacity_by
        self.lower_battery_power_by = lower_battery_power_by
        self.m = None
        self.participants_ids = None
        self.batteries = None

        # Optimization validity
        self.complement_tolerance: float = 1e-6
        self.complement_product = None
        self.is_complement: bool = False

        # Results
        self.input_p_kw = None
        self.schedules = None

        # Objective results
        self.actual_cost = None

    def solve(self, df_unctrl_p, _batteries: dict):
        self.participants_ids = df_unctrl_p.columns  # bus of participant
        self.batteries = copy.deepcopy(_batteries)

        self.input_p_kw = df_unctrl_p.copy()
        unctrl_p = df_unctrl_p.copy()
        unctrl_p.index = range(0, len(unctrl_p))  # Time is range(len(unctrl))
        unctrl_p = unctrl_p.T.to_dict()

        assert list(self.batteries.keys()) == list(self.participants_ids)
        assert self.grid_sell < self.grid_buy

        # Remove efficiency
        if self.remove_battery_efficiency:
            for p in self.batteries.keys():
                self.batteries[p]["eta"] = 1.0
                self.batteries[p]["max_kwh"] *= self.lower_battery_capacity_by
                self.batteries[p]["min_kw"] *= self.lower_battery_power_by

        if not self.solve_with_community_cost:
            self.m = self._solve_ind(unctrl_p, self.batteries)
        else:
            assert self.grid_buy - self.grid_sell > self.grid_fee
            self.m = self._solve_col(unctrl_p, self.batteries)
        self.post_process(df_unctrl_p)
        return None

    def _solve_ind(self, unctrl_p: dict, _batteries: dict, ):
        m = pyo.ConcreteModel()

        m.horizon = pyo.Set(initialize=list(unctrl_p.keys()), ordered=True)
        m.participants = pyo.Set(initialize=self.participants_ids, ordered=True)

        m.cost = pyo.Var(m.horizon, m.participants, domain=pyo.Reals)
        m.net_load_pos = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.net_load_neg = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_in = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_out = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_energy = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)

        m.grid_buy = pyo.Param(initialize=self.grid_buy)
        m.grid_sell = pyo.Param(initialize=self.grid_sell)
        m.max_participant_netload = pyo.Param(initialize=self.max_participant_netload)
        m.deltat = pyo.Param(initialize=self.freq / 60)
        m.last = pyo.Param(initialize=m.horizon.last())

        # Battery constraints
        def r_battery_max_power_in(m, t, p):
            return m.battery_in[t, p] <= _batteries[p]["max_kw"]
        m.r_battery_max_power_in = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_power_in)

        def r_battery_max_power_out(m, t, p):
            return m.battery_out[t, p] <= _batteries[p]["min_kw"]
        m.r_battery_max_power_out = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_power_out)

        def r_battery_energy(m, t, p):
            if t == 0:
                return m.battery_energy[t, p] == _batteries[p]["init_kwh"]
            else:
                return (
                        m.battery_energy[t, p]
                        == m.battery_energy[t - 1, p]
                        + m.battery_in[t - 1, p] * m.deltat * _batteries[p]["eta"]
                        - m.battery_out[t - 1, p] * m.deltat / _batteries[p]["eta"]
                )
        m.r_battery_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_energy)

        def r_battery_min_energy(m, t, p):
            return (
                    m.battery_energy[t, p]
                    >= _batteries[p]["max_kwh"] * _batteries[p]["offset"]
            )
        m.r_battery_min_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_min_energy)

        def r_battery_max_energy(m, t, p):
            return m.battery_energy[t, p] <= _batteries[p]["max_kwh"] * (
                    1 - _batteries[p]["offset"]
            )
        m.r_battery_max_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_energy)

        # To avoid issue with the last power set point not constrained by energy bounds
        def r_battery_end_power_out(m, p):
            return m.battery_out[m.last, p] == 0.0
        m.r_battery_end_power_out = pyo.Constraint(m.participants, rule=r_battery_end_power_out)

        def r_battery_end_power_in(m, p):
            return m.battery_in[m.last, p] == 0.0
        m.r_battery_end_power_in = pyo.Constraint(m.participants, rule=r_battery_end_power_in)

        # End energy fixed to simplify trade-off comparison
        def r_battery_end_energy(m, p):
            return m.battery_energy[m.last, p] == _batteries[p]["init_kwh"]
        m.r_battery_end_energy = pyo.Constraint(m.participants, rule=r_battery_end_energy)

        # Energy balance
        def r_participant_net_load(m, t, p):
            return (
                    m.net_load_pos[t, p] - m.net_load_neg[t, p] ==
                    m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p]
            )
        m.r_participant_net_load = pyo.Constraint(m.horizon, m.participants, rule=r_participant_net_load)

        def r_participant_max_net_load_pos(m, t, p):
            return m.net_load_pos[t, p] <= m.max_participant_netload
        m.r_participant_max_net_load_pos = pyo.Constraint(m.horizon, m.participants, rule=r_participant_max_net_load_pos)

        def r_participant_max_net_load_neg(m, t, p):
            return m.net_load_neg[t, p] <= m.max_participant_netload
        m.r_participant_max_net_load_neg = pyo.Constraint(m.horizon, m.participants, rule=r_participant_max_net_load_neg)

        # Cost
        def r_participant_cost(m, t, p):
            return (
                    m.cost[t, p] ==
                    m.grid_buy * m.net_load_pos[t, p] - m.grid_sell * m.net_load_neg[t, p]
            )
        m.r_participant_cost = pyo.Constraint(m.horizon, m.participants, rule=r_participant_cost)

        # Objective functions
        def costs(m):
            return sum(sum(m.cost[t, p] for p in m.participants) for t in m.horizon)
        m.objective = pyo.Objective(rule=costs, sense=pyo.minimize)

        with SolverFactory("gurobi") as opt:
            _ = opt.solve(m, tee=False)
        return m

    def _solve_col(self, unctrl_p: dict, _batteries: dict, ):
        m = pyo.ConcreteModel()

        m.horizon = pyo.Set(initialize=list(unctrl_p.keys()), ordered=True)
        m.participants = pyo.Set(initialize=self.participants_ids, ordered=True)

        m.cost = pyo.Var(m.horizon, domain=pyo.Reals)
        m.community_import = pyo.Var(m.horizon, domain=pyo.NonNegativeReals)
        m.community_export = pyo.Var(m.horizon, domain=pyo.NonNegativeReals)
        m.participant_import = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_in = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_out = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_energy = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)

        m.grid_buy = pyo.Param(initialize=self.grid_buy)
        m.grid_sell = pyo.Param(initialize=self.grid_sell)
        m.grid_fee = pyo.Param(initialize=self.grid_fee)
        m.max_participant_netload = pyo.Param(initialize=self.max_participant_netload)
        m.deltat = pyo.Param(initialize=self.freq / 60)
        m.last = pyo.Param(initialize=m.horizon.last())

        # Battery constraints
        def r_battery_max_power_in(m, t, p):
            return m.battery_in[t, p] <= _batteries[p]["max_kw"]
        m.r_battery_max_power_in = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_power_in)

        def r_battery_max_power_out(m, t, p):
            return m.battery_out[t, p] <= _batteries[p]["min_kw"]
        m.r_battery_max_power_out = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_power_out)

        def r_battery_energy(m, t, p):
            if t == 0:
                return m.battery_energy[t, p] == _batteries[p]["init_kwh"]
            else:
                return (
                        m.battery_energy[t, p]
                        == m.battery_energy[t - 1, p]
                        + m.battery_in[t - 1, p] * m.deltat * _batteries[p]["eta"]
                        - m.battery_out[t - 1, p] * m.deltat / _batteries[p]["eta"]
                )
        m.r_battery_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_energy)

        def r_battery_min_energy(m, t, p):
            return (
                    m.battery_energy[t, p]
                    >= _batteries[p]["max_kwh"] * _batteries[p]["offset"]
            )
        m.r_battery_min_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_min_energy)

        def r_battery_max_energy(m, t, p):
            return m.battery_energy[t, p] <= _batteries[p]["max_kwh"] * (
                    1 - _batteries[p]["offset"]
            )
        m.r_battery_max_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_energy)

        # To avoid issue with the last power set point not constrained by energy bounds
        def r_battery_end_power_out(m, p):
            return m.battery_out[m.last, p] == 0.0
        m.r_battery_end_power_out = pyo.Constraint(m.participants, rule=r_battery_end_power_out)

        def r_battery_end_power_in(m, p):
            return m.battery_in[m.last, p] == 0.0
        m.r_battery_end_power_in = pyo.Constraint(m.participants, rule=r_battery_end_power_in)

        # End energy fixed to simplify trade-off comparison
        def r_battery_end_energy(m, p):
            return m.battery_energy[m.last, p] == _batteries[p]["init_kwh"]
        m.r_battery_end_energy = pyo.Constraint(m.participants, rule=r_battery_end_energy)

        # Energy balance
        def r_participant_import(m, t, p):
            return (
                    m.participant_import[t, p] >=
                    m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p]
            )
        m.r_participant_import = pyo.Constraint(m.horizon, m.participants, rule=r_participant_import)

        def r_participant_max_import(m, t, p):
            return m.participant_import[t, p] <= m.max_participant_netload
        m.r_participant_max_import = pyo.Constraint(m.horizon, m.participants, rule=r_participant_max_import)

        def r_community_import_export(m, t):
            return m.community_import[t] - m.community_export[t] == sum(
                m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p] for p in m.participants)
        m.r_community_import_export = pyo.Constraint(m.horizon, rule=r_community_import_export)

        # Cost
        def r_community_cost(m, t):
            return (
                    m.cost[t] ==
                    m.grid_buy * m.community_import[t]
                    - m.grid_sell * m.community_export[t]
                    + m.grid_fee * (
                            sum(m.participant_import[t, p] for p in m.participants)
                            - m.community_import[t]
                    )
            )
        m.r_community_cost = pyo.Constraint(m.horizon, rule=r_community_cost)

        # Objective functions
        def costs(m):
            return sum(m.cost[t] for t in m.horizon)
        m.objective = pyo.Objective(rule=costs, sense=pyo.minimize)

        with SolverFactory("gurobi") as opt:
            _ = opt.solve(m, tee=False)
        return m

    def post_process(self, df_unctrl_p):
        self.schedules = self.get_timeseries_from_pyomo(self.input_p_kw.index)
        self.schedules["schedules"] = (self.schedules["battery_in"] - self.schedules["battery_out"]).copy()
        self.schedules["cost"] = self.schedules["cost"] / (60 / self.freq)

        # Check constraints
        self.is_complement = self.check_complement()

        # Get objective values
        self.actual_cost = self.schedules["cost"].sum().sum()
        return None

    def get_timeseries_from_pyomo(self, time_index):
        timeseries = dict()
        for var in self.m.component_objects(pyo.Var):
            _data = getattr(self.m, var.name).get_values()
            if var.name in ["0 dimension variable"]:  # 0-dimension
                timeseries[var.name] = _data[None]
                continue

            if self.solve_with_community_cost:
                if var.name in ["cost", "community_import", "community_export"]:  # 1-dimension
                    timeseries[var.name] = pd.DataFrame(index=["none"], data=_data).transpose()
                    timeseries[var.name].columns = [var.name]
                    timeseries[var.name].index = time_index
                    continue

            # Else data is 2-dimension
            timeseries[var.name] = pd.DataFrame(index=["none"], data=_data).transpose().unstack(level=1)
            timeseries[var.name].columns = timeseries[var.name].columns.levels[1]
            timeseries[var.name].index = time_index
        return timeseries

    def check_complement(self):
        self.complement_product = 0

        for b in self.schedules["battery_in"].columns:
            self.complement_product += (self.schedules["battery_in"][b] * self.schedules["battery_out"][b]).sum()

        if not self.solve_with_community_cost:
            for p in self.schedules["net_load_pos"].columns:
                self.complement_product += (self.schedules["net_load_pos"][p] * self.schedules["net_load_neg"][p]).sum()
        return True if self.complement_product <= self.complement_tolerance else False

    def print_summary(self):
        print("# Objectives")
        print(f"Cost = {round(self.actual_cost, 3)} euros")
        print("")

        # Passes test
        print("# Validity")
        print(f"Complement = {'Ok' if self.is_complement else 'ABOVE TOLERANCE'} ({round(self.complement_product, 6)})")
        print("")

    def set_score(self, score):
        # Getting network data related to the cost optimization
        self.line_losses = score.line_losses
        self.upstream_losses = score.upstream_losses
        self.total_losses = score.total_losses
        self.vm_above_percent = score.vm_above_percent
        self.vm_below_percent = score.vm_below_percent
        self.vm_square_error = score.vm_square_error
        self.max_upper_demand = score.max_upper_demand
        self.max_back_feed = score.max_back_feed
        self.complement_netload_product = 0


class Score(object):
    """Provide a score in the domain (cost versus line losses) for a given battery schedule"""

    def __init__(self, _net, solve_with_community_cost=False, upstream_losses_percent=0.02,
                 grid_buy=0.2276, grid_sell=0.13, grid_fee=0.0315, slack_bus_vm_pu=1.0, **_):
        # Parameters
        self.grid_buy = grid_buy
        self.grid_sell = grid_sell
        self.grid_fee = grid_fee
        self.net = _net.deepcopy()
        self.slack_bus_vm_pu = slack_bus_vm_pu
        self.upstream_losses_percent = upstream_losses_percent
        self.solve_with_community_cost = solve_with_community_cost
        self.freq = 15

        self.net["ext_grid"].loc[0, "vm_pu"] = self.slack_bus_vm_pu
        self.net["sgen"].loc[:, "p_mw"] = [0.0] * len(self.net["sgen"])  # Disable static generators (PV)
        self.net["sgen"].loc[:, "q_mvar"] = [0.0] * len(self.net["sgen"])
        self.net["sgen"].loc[:, "in_service"] = [False] * len(self.net["sgen"])

        # Results
        self.actual_cost = None

        self.upstream_power_flow = pd.Series()
        self.line_losses = None
        self.upstream_losses = None
        self.total_losses = None

        self.voltages = pd.DataFrame()
        self.vm_above_percent = None
        self.vm_below_percent = None
        self.vm_square_error = None

        self.max_upper_demand = None
        self.max_back_feed = None

    def reset_initial_values(self, bus_p_kw):
        # Results
        self.actual_cost = 0

        self.upstream_power_flow = pd.Series()
        self.line_losses = 0
        self.upstream_losses = 0
        self.total_losses = 0

        self.voltages = pd.DataFrame(columns=range(len(bus_p_kw.columns)))
        self.vm_above_percent = 0
        self.vm_below_percent = 0
        self.vm_square_error = 0

        self.max_upper_demand = 0
        self.max_back_feed = 0
        return None

    def score(self, df_unctrl_p, df_unctrl_q, df_schedules):
        # Get actual demand at each bus with added batteries power
        df_p_kw = (df_unctrl_p + df_schedules).copy().loc[:, df_unctrl_p.columns]  # Control time-series
        df_q_kvar = df_unctrl_q.copy()

        # ----------------- Specific to our scenario ---------------  # Shrink at bus time-series
        temp = df_p_kw.copy()  # Shrink columns back to number of load in net (assume first columns are integers)
        number_participant_at_buses = int(len(df_p_kw.columns) / len(self.net.load))
        for col in temp.columns[:len(self.net.load)]:
            extra_columns = [col + nb * 0.1 for nb in range(1, number_participant_at_buses)]
            temp.loc[:, col] += temp.loc[:, extra_columns].sum(axis=1)
            temp.drop(columns=extra_columns, inplace=True)
        assert temp.sum().sum() == pytest.approx(df_p_kw.sum().sum(), abs=1e-6)
        bus_p_kw = temp.copy()

        bus_q_kvar = df_q_kvar.copy()
        for col in bus_q_kvar.columns[:len(self.net.load)]:
            extra_columns = [col + nb * 0.1 for nb in range(1, number_participant_at_buses)]
            bus_q_kvar.loc[:, col] += bus_q_kvar.loc[:, extra_columns].sum(axis=1)
            bus_q_kvar.drop(columns=extra_columns, inplace=True)

        # Switch to load index and not bus name
        bus_p_kw.columns = list(range(0, len(bus_p_kw.columns)))
        bus_q_kvar.columns = list(range(0, len(bus_q_kvar.columns)))
        # ----------------- Specific to our scenario ---------------

        self.reset_initial_values(bus_p_kw)
        time_indexes = bus_p_kw.index.tolist()  # Run power flow for each step
        for index in time_indexes:
            self.net["load"].loc[:, "p_mw"] = bus_p_kw.loc[index, :] / 1000
            self.net["load"].loc[:, "q_mvar"] = bus_q_kvar.loc[index, :] / 1000
            pp.runpp(self.net)

            # Get results
            mask = self.net["res_bus"].index.isin(bus_p_kw.columns)
            self.voltages.loc[index, :] = self.net["res_bus"][mask].loc[:, "vm_pu"].values
            self.upstream_power_flow.loc[index] = self.net["res_ext_grid"]["p_mw"].iloc[0] * 1000
            self.line_losses += self.net["res_line"]["pl_mw"].sum() * 1000 / (60 / self.freq)
        self.score_grid_constraints()

        if self.solve_with_community_cost:
            self.calculate_community_cost(df_p_kw)
        else:
            self.calculate_sum_of_ind_cost(df_p_kw)
        return None

    def score_grid_constraints(self):
        self.upstream_losses = self.upstream_power_flow.clip(lower=0).sum() * self.upstream_losses_percent / (60 / self.freq)
        self.total_losses = self.line_losses + self.upstream_losses

        self.vm_above_percent = (self.voltages.applymap(lambda x: 1 if x > 1.05 else 0).sum() * 100 / len(self.voltages)).max()
        self.vm_below_percent = (self.voltages.applymap(lambda x: 1 if x < 0.95 else 0).sum() * 100 / len(self.voltages)).max()
        self.vm_square_error = (self.slack_bus_vm_pu - self.voltages).pow(2).sum().sum()

        self.max_upper_demand = self.upstream_power_flow.clip(lower=0).max()
        self.max_back_feed = self.upstream_power_flow.clip(upper=0).min()

    def calculate_sum_of_ind_cost(self, df_p_kw):
        cost = 0
        for col in df_p_kw.columns:
            cost += (df_p_kw.loc[:, col].clip(lower=0).sum() * self.grid_buy
                     - df_p_kw.loc[:, col].clip(upper=0).abs().sum() * self.grid_sell) / (60 / self.freq)
        self.actual_cost = cost
        return None

    def calculate_community_cost(self, df_p_kw):
        community_import = df_p_kw.sum(axis=1).clip(lower=0).sum(axis=0) / (60 / self.freq)
        community_export = abs(df_p_kw.sum(axis=1).clip(upper=0).sum(axis=0) / (60 / self.freq))
        sum_participant_import = df_p_kw.clip(lower=0).sum(axis=1).sum(axis=0) / (60 / self.freq)
        self.actual_cost = (self.grid_buy * community_import
                            - self.grid_sell * community_export
                            + self.grid_fee * (sum_participant_import - community_import)
                            )
        return None
