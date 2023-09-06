import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from moopf.cost import OptimizeCost, Score
from moopf.losses import OptimizeLosses
import logging


def compute_spread(sub_df_unctrl_p, _model):
    # Cost before adding batteries
    _original_cost = pd.Series()
    for _col in sub_df_unctrl_p.columns:
        _cost = (sub_df_unctrl_p.loc[:, _col].clip(lower=0).sum() * _model.grid_buy
                 - sub_df_unctrl_p.loc[:, _col].clip(upper=0).abs().sum() * _model.grid_sell) / (60 / _model.freq)
        _original_cost.loc[_col] = _cost
    _original_cost = _original_cost.to_frame("no battery")
    _original_cost = pd.concat([_original_cost, _model.schedules["cost"].sum().to_frame("battery")], axis=1)

    # Individual impact of adding batteries
    _original_cost["dif"] = _original_cost["battery"] - _original_cost["no battery"]
    return _original_cost["dif"].min(), _original_cost["dif"].median(), _original_cost["dif"].max()


class Tradeoff(object):
    """
    Provide a tradeoff between costs and losses minimization.

    # Parameters:
    -------------
    See cost and losses optimization

    # Results:
    ----------
    results : each trade-off and duration --> ["actual_cost", "total_losses", "vm_above_percent", "vm_below_percent",
                                    "vm_square_error", "max_upper_demand", "max_back_feed", "remove_battery_efficiency"]
    individual_costs : each trade-off, duration, and participant.
    """

    def __init__(self):

        # Tradeoff model
        self.durations = None  # Splitting of time
        self.nb_tradeoffs = 6
        self.batteries = None
        self.net = None
        self.options = None
        self.verbose = False
        self.logger = None

        # Internal
        self._loss_minimization_not_complement = False
        self.remove_efficiency_for = list()
        self.lower_battery_capacity_by = 1.0
        self.lower_battery_power_by = 1.0

        # Results
        self.columns = ["actual_cost", "total_losses", "vm_above_percent", "vm_below_percent",
                        "vm_square_error", "max_upper_demand", "max_back_feed", "remove_battery_efficiency"]
        self.results = None
        self.individual_costs = None
        self.instances = None
        self.summary = None

    def solve(self, _durations, _net, df_unctrl_p, df_unctrl_q, _batteries, _options):
        # Inputs
        self.net = _net.deepcopy()
        self.durations = list(_durations)
        self.batteries = copy.deepcopy(_batteries)  # Sure to reset battery characteristics
        self.options = copy.deepcopy(_options)
        self.options["lower_battery_capacity_by"] = self.lower_battery_capacity_by
        self.options["lower_battery_power_by"] = self.lower_battery_power_by

        # Result structure
        self.logger = logging.getLogger("Tradeoff")
        self.results = [pd.DataFrame(columns=self.columns) for _ in range(self.nb_tradeoffs + 2)]
        self.individual_costs = [pd.DataFrame(columns=df_unctrl_p.columns) for _ in range(self.nb_tradeoffs + 2)]
        self.instances = [list() for _ in range(self.nb_tradeoffs + 2)]

        for duration in tqdm(self.durations, desc="Progress"):
            sub_df_unctrl_p = df_unctrl_p.loc[duration, :].copy()
            sub_df_unctrl_q = df_unctrl_q.loc[duration, :].copy()
            self.logger.debug(f"")
            self.logger.debug(f"----- Info : Running {duration.start.strftime('%d %b %H:%M')} "
                              f"- {duration.stop.strftime('%d %b %H:%M')} | "
                              f"Load = {sub_df_unctrl_p.clip(lower=0).sum().sum() / 4:0.2f} kWh* | "
                              f"Prod = {sub_df_unctrl_p.clip(upper=0).sum().sum() / 4:0.2f} kWh*")
            self.get_problem_bounds(duration, sub_df_unctrl_p, sub_df_unctrl_q)

            # For each point of the trade-off curve
            _tradeoffs = np.linspace(self.results[0].loc[duration.start, "actual_cost"],
                                     self.results[self.nb_tradeoffs + 1].loc[duration.start, "actual_cost"], 20)
            _tradeoffs = [_tradeoffs[0], _tradeoffs[1], _tradeoffs[2],
                          _tradeoffs[4], _tradeoffs[9], _tradeoffs[14]]  # self.nb_tradeoffs = 5
            for tradeoff, target_cost in enumerate(_tradeoffs):
                tradeoff += 1  # Starts at 1 cause the first trade-off (0) is the cost only bound

                # Change target cost if target is too optimistic for approximate model
                target_cost = max(target_cost, self.min_cost_approximate_model)

                if self._loss_minimization_not_complement:
                    self.logger.debug(f"#{tradeoff} - Warning : removing battery efficiency preemptively")
                _instance = OptimizeLosses(**self.options)
                _instance.remove_battery_efficiency = self._loss_minimization_not_complement
                _instance.solve(self.net, sub_df_unctrl_p, sub_df_unctrl_q, self.batteries,
                                target_cost, self.results[0].loc[duration.start, "actual_cost"])

                # Remove efficiency even if loss minimization did not need it
                if not self._loss_minimization_not_complement and not _instance.is_complement_battery:
                    self.logger.debug(f"#{tradeoff} - Warning : removing battery efficiency "
                                      f"as a result of an unsuccessful run")
                    _instance = OptimizeLosses(**self.options)
                    _instance.remove_battery_efficiency = True
                    _instance.solve(self.net, sub_df_unctrl_p, sub_df_unctrl_q, self.batteries,
                                    target_cost, self.results[0].loc[duration.start, "actual_cost"])

                # Validity checks
                None if round(_instance.distance_to_minimum_cost, 3) >= 0 else self.logger.debug(f"Error : {_instance.distance_to_minimum_cost:0.2f}euros")
                None if _instance.is_complement_battery or _instance.remove_battery_efficiency else self.logger.debug(f"Error : battery efficiency issue")
                None if _instance.is_l_binding_constraint else self.logger.debug(f"Error : l binding issue")
                None if _instance.is_voltage_in_range_power_flow else self.logger.debug(f"Error : pandapower divergence")
                self.logging_iteration(tradeoff, sub_df_unctrl_p, _instance)
                self.save_results(_instance, _tradeoff=tradeoff, _duration=duration)

        self.summary = pd.DataFrame(index=range(self.nb_tradeoffs + 2))
        for tradeoff, _result in enumerate(self.results):
            self.summary.loc[tradeoff, "actual_cost"] = _result.loc[:, "actual_cost"].sum()
            self.summary.loc[tradeoff, "total_losses"] = _result.loc[:, "total_losses"].sum()
            self.summary.loc[tradeoff, "vm_above_percent"] = _result.loc[:, "vm_above_percent"].mean()
            self.summary.loc[tradeoff, "vm_below_percent"] = _result.loc[:, "vm_below_percent"].mean()
            self.summary.loc[tradeoff, "vm_square_error"] = _result.loc[:, "vm_square_error"].mean()
            self.summary.loc[tradeoff, "max_upper_demand"] = _result.loc[:, "max_upper_demand"].mean()
            self.summary.loc[tradeoff, "max_back_feed"] = _result.loc[:, "max_back_feed"].mean()

    def get_problem_bounds(self, duration, sub_df_unctrl_p, sub_df_unctrl_q):
        # Cost minimization
        lower_model = OptimizeCost(**self.options)
        lower_model.solve(sub_df_unctrl_p, self.batteries)

        scoring = Score(self.net, **self.options)
        scoring.score(sub_df_unctrl_p, sub_df_unctrl_q, lower_model.schedules["schedules"])
        lower_model.set_score(scoring)  # Run powerflow for each time step

        None if lower_model.is_complement else self.logger.debug(f"Error : cost bound model complement issue")
        self.save_results(lower_model, _tradeoff=0, _duration=duration)
        self.logging_iteration(0, sub_df_unctrl_p, lower_model)

        # Loss minimization
        upper_model = OptimizeLosses(**self.options)
        self._loss_minimization_not_complement = False
        self.min_cost_approximate_model = lower_model.actual_cost

        if duration.start in self.remove_efficiency_for:  # know from previous attempt that efficiency should be removed
            self.logger.debug(f"#{self.nb_tradeoffs + 1} - Warning : removing battery efficiency for this specific day")
            self._loss_minimization_not_complement = True
            upper_model.remove_battery_efficiency = True
            upper_model.solve(self.net, sub_df_unctrl_p, sub_df_unctrl_q, self.batteries)

            # Make sure that min cost is feasible with the approximate model (eta = 1.0, but lower max kWh)
            self.min_cost_approximate_model = self.min_cost_with_approximate_model(sub_df_unctrl_p)

        else:  # Don't remove efficiency a priori unless we detect it is needed
            upper_model.solve(self.net, sub_df_unctrl_p, sub_df_unctrl_q, self.batteries)
            if not upper_model.is_complement_battery:
                self.logger.debug(f"#{self.nb_tradeoffs + 1} - Warning : removing battery efficiency")
                upper_model = OptimizeLosses(**self.options)  # Recreate a model (safety measure)
                self._loss_minimization_not_complement = True
                upper_model.remove_battery_efficiency = True
                upper_model.solve(self.net, sub_df_unctrl_p, sub_df_unctrl_q, self.batteries)

                # Make sure that min cost is feasible with the approximate model (eta = 1.0, but lower max kWh)
                self.min_cost_approximate_model = self.min_cost_with_approximate_model(sub_df_unctrl_p)

        None if upper_model.is_l_binding_constraint else self.logger.debug(
            f"Error : Losses bound model l binding issue = {upper_model.l_binding_percent}%")
        None if upper_model.is_voltage_in_range_power_flow else self.logger.debug(
            f"Error : Losses bound model pandapower divergence = {upper_model.vm_error}p.u")
        self.save_results(upper_model, _tradeoff=self.nb_tradeoffs + 1, _duration=duration)
        self.logging_iteration(self.nb_tradeoffs + 1, sub_df_unctrl_p, upper_model)
        return None

    def min_cost_with_approximate_model(self, sub_df_unctrl_p):
        approximate_model = OptimizeCost(**self.options)
        approximate_model.remove_battery_efficiency = True  # implied that eta = 1.0 but lower_battery_capacity_by < 1.0
        approximate_model.solve(sub_df_unctrl_p, self.batteries)
        return approximate_model.actual_cost

    def save_results(self, _instance, _tradeoff, _duration):
        # Save overall results
        for _col in self.columns:
            self.results[_tradeoff].loc[_duration.start, _col] = float(getattr(_instance, _col))

        if not _instance.solve_with_community_cost:  # If individual results look at sharing disparities
            self.individual_costs[_tradeoff].loc[_duration.start, :] = _instance.schedules["cost"].sum()

        if _tradeoff in [0, 2, self.nb_tradeoffs + 1]:  # Fully save some optimization models
            self.instances[_tradeoff].append(copy.deepcopy(_instance.schedules))

    def logging_iteration(self, _tradeoff, sub_df_unctrl_p, _instance):
        if not _instance.solve_with_community_cost:
            _min, _median, _max = compute_spread(sub_df_unctrl_p, _instance)
        else:
            _min, _median, _max = np.nan, np.nan, np.nan
        _mean_charge = _instance.schedules["schedules"].clip(lower=0).sum(axis=0).mean() / (60 / _instance.freq)
        _max_discharge = _instance.schedules["schedules"].min().min()
        self.logger.debug(
            f"#{_tradeoff} - "
            f"Mean charge = {_mean_charge:0.2f} kWh | Max. discharge {_max_discharge:0.2f} kW | "
            f"Cost = {_instance.actual_cost:0.2f} - {_instance.complement_netload_product:0.2f} euros | "
            f"Spread = {_median:0.2f} ({_min:0.2f} ~ {_max:0.2f}) euros | "
            f"Losses = {_instance.total_losses:0.2f} ({_instance.upstream_losses:0.2f} + {_instance.line_losses:0.2f}) kWh")

