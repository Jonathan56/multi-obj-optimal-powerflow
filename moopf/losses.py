import copy

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import plotly.express as px
import pandapower.topology as top
import pandapower as pp
import networkx as nx
import pytest


class OptimizeLosses(object):
    """
    Quadratic program to minimize losses on a radial network when centrally controlling a fleet of batteries.

    Default: solve version with battery efficiency, no binaries, constrain sum of individual costs if any target.

    Parameters:
    -----------
    * solve_with_community_cost : if target_cost is False return the cost of energy for the community - else solves the
        minimization with a constraint on community cost.

    * target_cost : constraint a cost for the minimization losses problem.

    * upstream_losses_percent : percent of losses from the upstream network consumption (default 2%).

    * grid_buy, grid_sell, & grid_fee : to modify prices (scalar).

    * include_binary_constraint : include a binary constraint to ensure that battery cannot charge and discharge
        at the same time to limit line losses from surplus power. Turns the problem into MIQP which might not converge.

    * remove_battery_efficiency : simplification of the battery model which avoids including binary constraint.

    * max_participant_netload : to ensure that fix cost from subscribed power is respected.

    * slack_bus_vm_pu: voltage in per unit at the slack bus.

    Results:
    --------
    * schedules : it includes all decision variables from the optimization problem (battery_energy)

    * actual_cost : resulting energy cost in Euros from the optimization

    * total_losses : resulting losses from the lines (line_losses + upstream_losses)

    * max_upper_demand, max_back_feed, vm_square_error, vm_below_percent, vm_above_percent: secondary results on other
        aspects of grid constraints.

    """
    def __init__(self, solve_with_community_cost=False, upstream_losses_percent=0.02,
                 grid_buy=0.4, grid_sell=0.068, grid_fee=0.1, include_binary_constraint=False,
                 remove_battery_efficiency=False, max_participant_netload=12, slack_bus_vm_pu=1.0,
                 complement_tolerance=1e-2, lower_battery_capacity_by=1.0, lower_battery_power_by=1.0, **_):

        # Topology
        self.branch_impedance = None
        self.i_base = None
        self.z_base = None
        self.v_base = None
        self.s_base_kva = None
        self.s_base = None
        self.downstream_bus = None
        self.directed_branches = None
        self.participant_at_bus = None
        self.all_buses = None

        # Optimization model
        self.grid_buy = grid_buy
        self.grid_sell = grid_sell
        self.grid_fee = grid_fee
        self.upstream_losses_percent = upstream_losses_percent
        self.max_participant_netload = max_participant_netload
        self.target_cost = None
        self.minimum_cost = None
        self.cost_constraint_tolerance = 1e-4
        self.solve_with_community_cost = solve_with_community_cost
        self.include_binary_constraint = include_binary_constraint
        self.remove_battery_efficiency = remove_battery_efficiency
        self.lower_battery_capacity_by = lower_battery_capacity_by
        self.lower_battery_power_by = lower_battery_power_by
        self.freq: int = 15
        self.slack_bus_vm_pu: float = slack_bus_vm_pu
        self.slack_bus = 42
        self.m = None
        self.participants_ids = None
        self.MIPGap = 1e-4
        self.TimeLimit = 300
        self.batteries = None

        # Optimization validity
        self.voltage_tolerance: float = 1e-3
        self.complement_tolerance: float = complement_tolerance
        self.l_binding_tolerance: float = 1e-5
        self.vm_error = None
        self.complement_battery_product = None
        self.complement_netload_product = None
        self.distance_to_minimum_cost = None
        self.l_binding_percent = None
        self.is_voltage_in_range_power_flow: bool = False
        self.is_l_binding_constraint: bool = False
        self.is_complement_battery: bool = False
        self.is_complement_netload = False

        self._net_check = None  # Intermediary results
        self.power_flow_vm = None
        self.power_flow_output = None
        self.vm_idx = None
        self.obj_losses = None
        self.obj_upstream_losses = None

        # Results
        self.schedules = None
        self.input_p_kw = None

        # Objective results
        self.line_losses = None
        self.upstream_losses = None
        self.total_losses = None

        self.max_upper_demand = None
        self.max_back_feed = None
        self.vm_square_error = None
        self.vm_below_percent = None
        self.vm_above_percent = None

        self.actual_cost = None
        self.cost_losses = None

    def get_network_topology_no_transfo(self, _net):
        # All buses
        self.all_buses = _net["bus"].index.to_list()

        self.participant_at_bus = {bus: [] for bus in self.all_buses}
        for bus in self.all_buses:
            for participant in self.participants_ids:
                if int(bus) == int(participant):
                    self.participant_at_bus[bus].append(participant)

        # Directed branch
        mg = top.create_nxgraph(_net)
        depth_from_root = [n for n in nx.traversal.bfs_tree(mg, self.slack_bus, depth_limit=None)]
        all_branches = [(i, j) for i, j in zip(_net["line"]["from_bus"], _net["line"]["to_bus"])]
        self.directed_branches = []
        for (i, j) in all_branches:
            if depth_from_root.index(i) > depth_from_root.index(j):
                self.directed_branches.append((j, i))
            else:
                self.directed_branches.append((i, j))

        # Downstream bus with depth=1
        self.downstream_bus = {bus: [j for (i, j) in self.directed_branches if i == bus] for bus in self.all_buses}

        # Per-unit base
        self.s_base = _net.sn_mva  # MVA --> 1
        self.s_base_kva = self.s_base * 1000
        self.v_base = _net["bus"].loc[0, "vn_kv"]  # kV --> 0.4
        self.z_base = self.v_base ** 2 / self.s_base  # Ohm --> 0.160

        # Branch impedance
        self.branch_impedance = {}
        for branch in self.directed_branches:
            i, j = branch
            mask = (_net["line"]["from_bus"] == i) & (_net["line"]["to_bus"] == j)
            if _net["line"][mask].empty:
                mask = (_net["line"]["from_bus"] == j) & (_net["line"]["to_bus"] == i)
            self.branch_impedance[(i, j)] = {"r": (_net["line"][mask]["r_ohm_per_km"].values[0] *
                                                   _net["line"][mask]["length_km"].values[0] / self.z_base),
                                             "x": (_net["line"][mask]["x_ohm_per_km"].values[0] *
                                                   _net["line"][mask]["length_km"].values[0] / self.z_base)}
        return True

    def solve(self, _net, df_unctrl_p, df_unctrl_q, _batteries: dict, target_cost=False, minimum_cost=0.0):
        self.batteries = copy.deepcopy(_batteries)
        self.target_cost = target_cost
        self.minimum_cost = minimum_cost
        self.participants_ids = df_unctrl_p.columns  # bus of participant
        self.get_network_topology_no_transfo(_net)

        self.input_p_kw = df_unctrl_p.copy()
        unctrl_p = df_unctrl_p.copy()
        unctrl_p.index = range(0, len(unctrl_p))   # Time is range(len(unctrl))
        unctrl_p = unctrl_p.T.to_dict()

        unctrl_q = df_unctrl_q.copy()
        unctrl_q.index = range(0, len(unctrl_q))   # Time is range(len(unctrl))
        unctrl_q = unctrl_q.T.to_dict()

        assert list(self.batteries.keys()) == list(self.participants_ids)
        assert self.grid_sell < self.grid_buy

        # Remove efficiency
        if self.remove_battery_efficiency:
            for p in self.batteries.keys():
                self.batteries[p]["eta"] = 1.0
                self.batteries[p]["max_kwh"] *= self.lower_battery_capacity_by
                self.batteries[p]["min_kw"] *= self.lower_battery_power_by

        self.m = self._solve(unctrl_p, unctrl_q, self.batteries)
        self.post_process(_net, df_unctrl_p, df_unctrl_q)
        return None

    def _solve(self, unctrl_p: dict, unctrl_q: dict, _batteries: dict):
        m = pyo.ConcreteModel()

        m.horizon = pyo.Set(initialize=list(unctrl_p.keys()), ordered=True)
        m.participants = pyo.Set(initialize=self.participants_ids, ordered=True)
        m.lines = pyo.Set(initialize=self.directed_branches, ordered=True)
        m.buses = pyo.Set(initialize=self.all_buses, ordered=True)

        m.battery_in = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_out = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.battery_energy = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)

        m.p = pyo.Var(m.horizon, m.lines, within=pyo.Reals)
        m.q = pyo.Var(m.horizon, m.lines, within=pyo.Reals)
        m.v = pyo.Var(m.horizon, m.buses, within=pyo.NonNegativeReals)
        m.l = pyo.Var(m.horizon, m.lines, within=pyo.NonNegativeReals)
        for t in m.horizon:
            m.v[t, self.slack_bus].fix(self.slack_bus_vm_pu)
        m.upstream_losses = pyo.Var(m.horizon, domain=pyo.NonNegativeReals)

        # Additional parameters include : participant_at_bus[j], branch_impedance[(i, j)]["r"] downstream_bus[j]
        m.deltat = pyo.Param(initialize=self.freq / 60)
        m.last = pyo.Param(initialize=m.horizon.last())
        m.s_base_kva = pyo.Param(initialize=self.s_base_kva)
        m.max_participant_netload = pyo.Param(initialize=self.max_participant_netload)
        m.upstream_losses_percent = pyo.Param(initialize=self.upstream_losses_percent)

        # Battery constraints
        def r_battery_min_energy(m, t, p):
            return m.battery_energy[t, p] >= _batteries[p]["max_kwh"] * _batteries[p]["offset"]
        m.r_battery_min_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_min_energy)

        def r_battery_max_energy(m, t, p):
            return m.battery_energy[t, p] <= _batteries[p]["max_kwh"] * (1 - _batteries[p]["offset"])
        m.r_battery_max_energy = pyo.Constraint(m.horizon, m.participants, rule=r_battery_max_energy)

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

        if self.include_binary_constraint:
            m.binary_battery = pyo.Var(m.horizon, m.participants, domain=pyo.Boolean)

            def r_enforce_complementary(m, t, p):
                return (m.battery_in[t, p] + m.battery_out[t, p] ==
                        m.binary_battery[t, p] * m.battery_in[t, p] + (1 - m.binary_battery[t, p]) * m.battery_out[t, p])
            m.r_enforce_complementary = pyo.Constraint(m.horizon, m.participants, rule=r_enforce_complementary)

        # Ensure that participants remain within their power contract
        def r_participant_max_net_load(m, t, p):
            return m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p] <= m.max_participant_netload
        m.r_participant_max_net_load = pyo.Constraint(m.horizon, m.participants, rule=r_participant_max_net_load)

        # Power flow constraints (SOCP)
        def r_dist_flow_p(m, t, i, j):
            pj = sum(m.battery_in[t, k] - m.battery_out[t, k] + unctrl_p[t][k] for k in self.participant_at_bus[j]) / m.s_base_kva
            return (m.p[t, (i, j)] == self.branch_impedance[(i, j)]["r"] * m.l[t, (i, j)]
                    + pj
                    + sum(m.p[t, (j, k)] for k in self.downstream_bus[j]))
        m.r_dist_flow_p = pyo.Constraint(m.horizon, m.lines, rule=r_dist_flow_p)

        # Reactive power flow
        def r_dist_flow_q(m, t, i, j):
            qj = sum(unctrl_q[t][k] for k in self.participant_at_bus[j]) / m.s_base_kva
            return (m.q[t, (i, j)] == self.branch_impedance[(i, j)]["x"] * m.l[t, (i, j)]
                    + qj
                    + sum(m.q[t, (j, k)] for k in self.downstream_bus[j]))
        m.r_dist_flow_q = pyo.Constraint(m.horizon, m.lines, rule=r_dist_flow_q)

        # Voltage at node j (follow line since slack bus is fixed)
        def r_dist_flow_v(m, t, i, j):
            return (m.v[t, j] == m.v[t, i]
                    - 2 * (self.branch_impedance[(i, j)]["r"] * m.p[t, (i, j)] + self.branch_impedance[(i, j)]["x"] * m.q[t, (i, j)])
                    + (self.branch_impedance[(i, j)]["r"]**2 + self.branch_impedance[(i, j)]["x"]**2) * m.l[t, (i, j)])
        m.r_dist_flow_v = pyo.Constraint(m.horizon, m.lines, rule=r_dist_flow_v)

        # L(i, j) representing the current magnitude on each line
        def r_dist_flow_l(m, t, i, j):
            return m.p[t, (i, j)]**2 + m.q[t, (i, j)]**2 <= m.l[t, (i, j)] * m.v[t, i]
        m.r_dist_flow_l = pyo.Constraint(m.horizon, m.lines, rule=r_dist_flow_l)

        # Voltage bounds at each node
        def r_dist_flow_v_bounds(m, t, bus):
            return pyo.inequality(0.80**2, m.v[t, bus], 1.2**2)
        m.r_dist_flow_v_bounds = pyo.Constraint(m.horizon, m.buses, rule=r_dist_flow_v_bounds)

        # Upstream losses positive number
        def r_upstream_losses(m, t):
            return m.upstream_losses[t] >= m.p[t, (42, 3)] * m.upstream_losses_percent
        m.r_upstream_losses = pyo.Constraint(m.horizon, rule=r_upstream_losses)

        # Constraint the cost of the solution
        if self.target_cost and not self.solve_with_community_cost:
            self.add_individual_costs(m, unctrl_p)

        if self.target_cost and self.solve_with_community_cost:
            self.add_community_cost(m, unctrl_p)

        # Objective functions
        def losses(m):
            return sum(
                m.upstream_losses[t]
                + sum(self.branch_impedance[(i, j)]["r"] * m.l[t, (i, j)] for (i, j) in m.lines)
                for t in m.horizon
            )
        m.objective = pyo.Objective(rule=losses, sense=pyo.minimize)

        with SolverFactory("gurobi") as opt:
            opt.options["mipgap"] = self.MIPGap
            opt.options["timelimit"] = self.TimeLimit
            _ = opt.solve(m, tee=False)
        return m

    def add_individual_costs(self, m, unctrl_p):
        m.cost = pyo.Var(m.horizon, m.participants, domain=pyo.Reals)
        m.net_load_pos = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.net_load_neg = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.target_cost = pyo.Param(initialize=self.target_cost)
        m.grid_buy = pyo.Param(initialize=self.grid_buy)
        m.grid_sell = pyo.Param(initialize=self.grid_sell)

        # Energy balance
        def r_participant_net_load(m, t, p):
            return (m.net_load_pos[t, p] - m.net_load_neg[t, p] ==
                    m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p])
        m.r_participant_net_load = pyo.Constraint(m.horizon, m.participants, rule=r_participant_net_load)

        # Sum of individual costs
        def r_participant_cost(m, t, p):
            return m.cost[t, p] == m.grid_buy * m.net_load_pos[t, p] - m.grid_sell * m.net_load_neg[t, p]
        m.r_participant_cost = pyo.Constraint(m.horizon, m.participants, rule=r_participant_cost)

        def r_enforce_total_cost_upper(m):
            return (sum(sum(m.cost[t, p] for p in m.participants) for t in m.horizon) * m.deltat
                    <= (1 + self.cost_constraint_tolerance) * m.target_cost)
        m.r_enforce_total_cost_upper = pyo.Constraint(rule=r_enforce_total_cost_upper)

        def r_enforce_total_cost_lower(m):
            return (sum(sum(m.cost[t, p] for p in m.participants) for t in m.horizon) * m.deltat
                    >= (1 - self.cost_constraint_tolerance) * m.target_cost)
        m.r_enforce_total_cost_lower = pyo.Constraint(rule=r_enforce_total_cost_lower)
        return None

    def add_community_cost(self, m, unctrl_p):
        m.cost = pyo.Var(m.horizon, domain=pyo.Reals)
        m.community_import = pyo.Var(m.horizon, domain=pyo.NonNegativeReals)
        m.community_export = pyo.Var(m.horizon, domain=pyo.NonNegativeReals)
        m.participant_import = pyo.Var(m.horizon, m.participants, domain=pyo.NonNegativeReals)
        m.grid_buy = pyo.Param(initialize=self.grid_buy)
        m.grid_sell = pyo.Param(initialize=self.grid_sell)
        m.grid_fee = pyo.Param(initialize=self.grid_fee)
        m.target_cost = pyo.Param(initialize=self.target_cost)

        # Energy balance
        def r_participant_import(m, t, p):
            return m.participant_import[t, p] >= m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p]
        m.r_participant_import = pyo.Constraint(m.horizon, m.participants, rule=r_participant_import)

        def r_community_import_export(m, t):
            return m.community_import[t] - m.community_export[t] == sum(
                m.battery_in[t, p] - m.battery_out[t, p] + unctrl_p[t][p] for p in m.participants)
        m.r_community_import_export = pyo.Constraint(m.horizon, rule=r_community_import_export)

        # Community cost with network fees
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

        # Constrained cost value
        def r_enforce_total_cost_upper(m):
            return sum(m.cost[t] for t in m.horizon) * m.deltat <= (1 + self.cost_constraint_tolerance) * m.target_cost
        m.r_enforce_total_cost_upper = pyo.Constraint(rule=r_enforce_total_cost_upper)

        def r_enforce_total_cost_lower(m):
            return sum(m.cost[t] for t in m.horizon) * m.deltat >= (1 - self.cost_constraint_tolerance) * m.target_cost
        m.r_enforce_total_cost_lower = pyo.Constraint(rule=r_enforce_total_cost_lower)

    def post_process(self, _net, df_unctrl_p, df_unctrl_q):
        self.schedules = self.get_timeseries_from_pyomo_opf(df_unctrl_p.index)
        self.schedules["schedules"] = (self.schedules["battery_in"] - self.schedules["battery_out"]).copy()
        self.schedules["v"] = self.schedules["v"].applymap(lambda x: np.sqrt(x))
        if "cost" in self.schedules:
            self.schedules["cost"] = self.schedules["cost"] / (60 / self.freq)

        # Check battery complement
        self.complement_battery_product = 0
        self.is_complement_battery = self.check_complement_battery()

        # Check netload complement
        self.complement_netload_product = 0
        if self.target_cost and self.solve_with_community_cost:
            self.is_complement_netload = self.check_complement_community_netload()
        if self.target_cost and not self.solve_with_community_cost:
            self.is_complement_netload = self.check_complement_netload()

        # Check that SOCP is exact (normally yes)
        self.is_l_binding_constraint = self.check_l_binding_constraint()
        self.is_voltage_in_range_power_flow = self.check_voltage_in_range_power_flow(_net, df_unctrl_p, df_unctrl_q)

        # Get objective values
        _losses = self.schedules["l"].copy()
        _losses.columns = [str(c) for c in _losses.columns]
        for key in self.schedules["l"].columns:
            _losses.loc[:, str(key)] *= self.branch_impedance[key]["r"]
        self.obj_losses = _losses.sum().sum()
        self.line_losses = (self.obj_losses / (60 / self.freq)) * self.s_base_kva

        self.obj_upstream_losses = (self.schedules["upstream_losses"]).sum().values[0]
        self.upstream_losses = (self.obj_upstream_losses / (60 / self.freq)) * self.s_base_kva
        self.total_losses = self.line_losses + self.upstream_losses
        self.cost_losses = self.total_losses * self.grid_buy

        # Get secondary results from power flow
        self.vm_above_percent = (self.schedules["v"].applymap(lambda x: 1 if x > 1.05 else 0).sum() * 100 / len(self.schedules["v"])).max()
        self.vm_below_percent = (self.schedules["v"].applymap(lambda x: 1 if x < 0.95 else 0).sum() * 100 / len(self.schedules["v"])).max()
        self.vm_square_error = (self.slack_bus_vm_pu - self.schedules["v"]).pow(2).sum().sum()
        self.max_upper_demand = (self.schedules["p"].loc[:, [(42, 3)]] * self.s_base_kva).clip(lower=0).max().values[0]
        self.max_back_feed = (self.schedules["p"].loc[:, [(42, 3)]] * self.s_base_kva).clip(upper=0).min().values[0]

        # Sum of individual costs
        if self.target_cost:
            self.actual_cost = self.schedules["cost"].sum().sum()

        elif not self.solve_with_community_cost:
            self.schedules["cost"] = pd.DataFrame()
            subset_p_kw = (df_unctrl_p + self.schedules["schedules"].loc[:, df_unctrl_p.columns]).copy()
            for col in subset_p_kw.columns:
                self.schedules["cost"].loc[df_unctrl_p.index[-1], col] = (
                         subset_p_kw.loc[:, col].clip(lower=0).sum() * self.grid_buy
                         - subset_p_kw.loc[:, col].clip(upper=0).abs().sum() * self.grid_sell) / (60 / self.freq)
            self.actual_cost = self.schedules["cost"].sum().sum()

        # Community cost
        else:
            subset_p_kw = (df_unctrl_p + self.schedules["schedules"].loc[:, df_unctrl_p.columns]).copy()
            community_import = subset_p_kw.sum(axis=1).clip(lower=0).sum(axis=0) / (60 / self.freq)
            community_export = abs(subset_p_kw.sum(axis=1).clip(upper=0).sum(axis=0) / (60 / self.freq))
            sum_participant_import = subset_p_kw.clip(lower=0).sum(axis=1).sum(axis=0) / (60 / self.freq)
            self.actual_cost = (self.grid_buy * community_import
                                - self.grid_sell * community_export
                                + self.grid_fee * (sum_participant_import - community_import)
                                )
        return None

    def get_timeseries_from_pyomo_opf(self, time_index):
        timeseries = dict()
        for var in self.m.component_objects(pyo.Var):
            # Regroup line indexes into a tuple
            _data = getattr(self.m, var.name).get_values()
            if var.name in ["0 dimension variable"]:
                timeseries[var.name] = _data[None]
                continue

            if self.solve_with_community_cost and var.name in ["cost"]:  # Cost can be 1 or 2-dimension
                timeseries[var.name] = pd.DataFrame(index=["none"], data=_data).transpose()
                timeseries[var.name].columns = [var.name]
                timeseries[var.name].index = time_index
                continue

            if var.name in ["upstream_losses", "community_import", "community_export"]:  # 1-dimension
                timeseries[var.name] = pd.DataFrame(index=["none"], data=_data).transpose()
                timeseries[var.name].columns = [var.name]
                timeseries[var.name].index = time_index
                continue

            if var.name in ["p", "q", "l"]:
                _data = {(k[0], (k[1], k[2])): v for k, v in _data.items()}
            timeseries[var.name] = pd.DataFrame(index=["none"], data=_data).transpose().unstack(level=1)
            timeseries[var.name].columns = timeseries[var.name].columns.levels[1]
            timeseries[var.name].index = time_index
        return timeseries

    def check_complement_battery(self):
        for b in self.schedules["battery_in"].columns:
            _compl_df = pd.concat([self.schedules["battery_in"][b], self.schedules["battery_out"][b]], keys=["in", "out"], axis=1)

            _scaling = _compl_df.max(axis=1)   # complement = (2 * 0.5) / 2 = 0.5 kWh
            self.complement_battery_product += (_compl_df["in"] * _compl_df["out"] / _scaling).sum() / (60 / self.freq)
        return True if self.complement_battery_product <= self.complement_tolerance else False

    def check_complement_netload(self):
        for p in self.schedules["net_load_pos"].columns:
            _compl_df = pd.concat([self.schedules["net_load_pos"][p], self.schedules["net_load_neg"][p]], keys=["in", "out"], axis=1)

            _scaling = _compl_df.max(axis=1)   # complement = (2 * 0.5) / 2 = 0.5 kWh
            self.complement_netload_product += (_compl_df["in"] * _compl_df["out"] / _scaling).sum() / (60 / self.freq)
        self.complement_netload_product *= (self.grid_buy - self.grid_sell)
        self.distance_to_minimum_cost = self.target_cost - self.complement_netload_product - self.minimum_cost  # Should be actual cost!
        return True if self.complement_netload_product <= self.complement_tolerance else False

    def check_complement_community_netload(self):
        _compl_df = pd.concat([self.schedules["community_import"]["community_import"],
                               self.schedules["community_export"]["community_export"]], keys=["in", "out"], axis=1)

        _scaling = _compl_df.max(axis=1)   # complement = (2 * 0.5) / 2 = 0.5 kWh
        self.complement_netload_product += (_compl_df["in"] * _compl_df["out"] / _scaling).sum() / (60 / self.freq)
        self.complement_netload_product *= (self.grid_buy - self.grid_sell)
        self.distance_to_minimum_cost = self.target_cost - self.complement_netload_product - self.minimum_cost
        return True if self.complement_netload_product <= self.complement_tolerance else False

    def check_l_binding_constraint(self):
        constraint = None
        for constraint in list(self.m.component_objects(pyo.Constraint)):
            if constraint.name == "r_dist_flow_l":
                break  # Select corresponding constraint constraint = "r_dist_flow_l"

        self.l_binding_percent = 0
        _count, _total = 0, len(list(constraint.keys()))
        for key in list(constraint.keys()):
            _count += 1 if pytest.approx(
                constraint[key].body(), abs=self.l_binding_tolerance ) == float(constraint[key].upper) else 0

        self.l_binding_percent = _count * 100 / _total
        return True if self.l_binding_percent >= 100 else False

    def check_voltage_in_range_power_flow(self, _net, df_unctrl_p, df_unctrl_q):
        # Remove transformer
        _net_check = _net.deepcopy()
        _net_check["ext_grid"].loc[0, "vm_pu"] = self.slack_bus_vm_pu

        # Pick a time (is there any production ?)
        if df_unctrl_p.clip(upper=0).sum().sum() >= -0.001:  # yes
            self.vm_idx = self.schedules["v"][self.schedules["v"].min().idxmin()].idxmin()
        else:
            self.vm_idx = self.schedules["v"][self.schedules["v"].max().idxmax()].idxmax()

        # Get actual demand at each bus with added batteries power (index = net.load.index)
        subset_p_kw = (df_unctrl_p
                       + self.schedules["schedules"]).loc[[self.vm_idx], :].copy().loc[:, df_unctrl_p.columns]
        subset_q_kvar = df_unctrl_q.loc[[self.vm_idx], :].copy()

        # Shrink columns back to number of load in net
        temp = subset_p_kw.copy()
        number_participant_at_buses = int(len(subset_p_kw.columns) / len(_net_check.load))
        for col in temp.columns[:len(_net_check.load)]:
            extra_columns = [col + nb * 0.1 for nb in range(1, number_participant_at_buses)]
            temp.loc[:, col] += temp.loc[:, extra_columns].sum(axis=1)
            temp.drop(columns=extra_columns, inplace=True)
        assert temp.sum().sum() == pytest.approx(subset_p_kw.sum().sum(), abs=1e-6)
        subset_p_kw = temp.copy()

        for col in subset_q_kvar.columns[:len(_net_check.load)]:
            extra_columns = [col + nb * 0.1 for nb in range(1, number_participant_at_buses)]
            subset_q_kvar.loc[:, col] += subset_q_kvar.loc[:, extra_columns].sum(axis=1)
            subset_q_kvar.drop(columns=extra_columns, inplace=True)

        # Switch to load index and not bus name
        subset_p_kw.columns = list(range(0, len(subset_p_kw.columns)))
        subset_q_kvar.columns = list(range(0, len(subset_p_kw.columns)))

        # Disable static generators (PV)
        _net_check["sgen"].loc[:, "p_mw"] = [0.0] * len(_net_check["sgen"])
        _net_check["sgen"].loc[:, "q_mvar"] = [0.0] * len(_net_check["sgen"])
        _net_check["sgen"].loc[:, "in_service"] = [False] * len(_net_check["sgen"])

        _net_check["load"].loc[:, "p_mw"] = subset_p_kw.loc[self.vm_idx, :] / 1000
        _net_check["load"].loc[:, "q_mvar"] = subset_q_kvar.loc[self.vm_idx, :] / 1000
        pp.runpp(_net_check)
        self._net_check = _net_check.deepcopy()

        self.power_flow_vm = _net_check["res_bus"][["vm_pu"]].T.copy()
        self.power_flow_vm.index = [self.vm_idx]

        # Get grid outcomes
        res_bus = _net_check["res_bus"][_net_check["res_bus"].index.isin(self.participants_ids)]
        self.power_flow_output = dict()
        self.power_flow_output["vm_min"] = round(res_bus["vm_pu"].min(), 3)
        self.power_flow_output["vm_max"] = round(res_bus["vm_pu"].max(), 3)
        self.power_flow_output["line_loading_max"] = round(_net_check["res_line"]["loading_percent"].max(), 3)
        self.power_flow_output["line_losses_kwh"] = round(_net_check["res_line"]["pl_mw"].sum() * 1000 / (60 / self.freq), 3)
        self.power_flow_output["ext_grid_kw"] = round(_net_check["res_ext_grid"]["p_mw"].iloc[0] * 1000, 3)

        self.vm_error = (self.schedules["v"].loc[self.vm_idx, :].min() - self.power_flow_vm.min().min())
        return True if np.abs(self.vm_error) <= self.voltage_tolerance else False

    def print_summary(self, verbose: bool = False):
        print("# Objectives")
        print(f"Network losses = {round(self.obj_losses, 5)}  | value = {round(self.line_losses, 3)} kWh")
        print(f"Upstream losses = {round(self.obj_upstream_losses, 5)}  | value = {round(self.upstream_losses, 3)} kWh")
        print(f"Cost = {round(self.actual_cost, 3)} euros      |  Losses cost = {round(self.cost_losses, 3)} euros")
        print("")

        # Passes test
        print("# Validity")
        print(f"Complement = {'Ok' if self.is_complement_battery else 'ABOVE TOLERANCE'} ({round(self.complement_battery_product, 6)})")
        print(f"L binding  = {'Ok' if self.is_l_binding_constraint else 'ABOVE TOLERANCE'} ({round(self.l_binding_percent, 3)})")
        print(f"Pandapower = {'Ok' if self.is_voltage_in_range_power_flow else 'ABOVE TOLERANCE'} ({round(self.vm_error, 6)})")
        print("")

        # Secondary objectives
        print("# Secondary objectives")
        print(f"Vm squared error   = {round(self.vm_square_error, 3)}")
        print(f"Ext. grid max      = {round(self.max_upper_demand, 1)} kW")
        print(f"Ext. grid backfeed = {round(self.max_back_feed, 1)} kW")

        if verbose:
            # Power flow validation
            print("# Pandapower")
            print(f"Vm min       = {self.power_flow_output['vm_min']} pu")
            print(f"Vm max       = {self.power_flow_output['vm_max']} pu")
            print(f"Line loading = {self.power_flow_output['line_loading_max']} %")
            print(f"Line losses  = {self.power_flow_output['line_losses_kwh']} kWh")
            print(f"Ext. grid    = {self.power_flow_output['ext_grid_kw']} kW")
            print("")

    def plot_voltage_profiles(self, _net, top_bus: int = 42, add_pandapower: bool = True):
        buses_distance_m = {}
        for bus in _net["bus"].index.to_list():
            buses_distance_m[bus] = int(round(top.calc_distance_to_bus(_net, bus).loc[top_bus] * 1000, 0))
        buses_distance_m = pd.Series(buses_distance_m)

        graph = buses_distance_m.to_frame("distance_m")
        graph["bus"] = graph.index

        graph["optimization"] = self.schedules["v"].loc[self.vm_idx, :]
        graph["pandapower"] = self.power_flow_vm.loc[self.vm_idx, :]

        # Compare line current
        _i_line = self._net_check["res_line"].loc[:, ["i_ka"]]
        _i_line["index"] = [
            (9, 2), (11, 13), (3, 6), (1, 8), (7, 10), (10, 9), (3, 7), (6, 11),
                            (13, 5), (3, 0), (3, 1), (8, 12), (5, 4), (42, 3)]
        _i_line.set_index("index", drop=True, inplace=True)
        _i_line = _i_line.reindex(self.schedules["l"].loc[self.vm_idx, :].index)

        _i_line["opti_i_ka"] = (self.schedules["l"].loc[self.vm_idx, :]).pow(1/2) * (1 / (np.sqrt(3) * 0.4))
        _i_line = _i_line * 1000
        _i_line["percent_diff_%"] = round((_i_line["i_ka"] - _i_line["opti_i_ka"]) * 100 / _i_line["i_ka"], 3)
        _i_line["abs_diff_A"] = round(_i_line["i_ka"] - _i_line["opti_i_ka"], 3)

        if add_pandapower:
            _y_vars = ["optimization", "pandapower"]
        else:
            _y_vars = ["optimization"]

        fig = px.scatter(graph, x="distance_m", y=_y_vars, text="bus",
                         color_discrete_sequence=px.colors.qualitative.Set3[3:])
        fig.update_traces(textposition="top right", textfont_size=12)

        for color, _y_var in enumerate(_y_vars):
            for i, j in self.directed_branches:
                fig.add_shape(type='line', x0=graph.loc[i, "distance_m"], y0=graph.loc[i, _y_var],
                              x1=graph.loc[j, "distance_m"], y1=graph.loc[j, _y_var],
                              line_color=px.colors.qualitative.Set3[3 + color], line_width=2)

                if _y_var == "pandapower":
                    _x = graph.loc[i, "distance_m"] + (graph.loc[j, "distance_m"] - graph.loc[i, "distance_m"])/2
                    _y = graph.loc[i, _y_var] + (graph.loc[j, _y_var] - graph.loc[i, _y_var])/2
                    _ax = 7
                    _text = _i_line.loc[[tuple((i, j))], "percent_diff_%"].values[0]
                    if np.abs(_text) > 1e-2:
                        fig.add_annotation(x=_x, y=_y, ax=_ax, text=f"{round(_text, 2)}%")

        fig.update_layout(title=f"Time = {self.vm_idx.strftime('%H:%M')}")
        fig.update_layout(height=700, width=1000, legend_orientation="h")
        return fig
