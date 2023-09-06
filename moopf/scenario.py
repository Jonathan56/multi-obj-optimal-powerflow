import pytest
import numpy as np
import pandas as pd
import simbench as sb
from datetime import datetime, timedelta
import pandapower as pp
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Modified --> 26 households 12kW 11000kWh/year, 3kWp PV, 5kWh battery
# Modified --> Cable section 35mm2 (r=0,641 and x=0,101 ohm per km), distance x2
# Modified --> Transformator impedance x1
def get_network_and_timeseries(pv_capacity_kw=3, battery_capacity_kwh=5, path=""):
    # Modifications
    coef_load = 1
    max_p_kw = 12
    coef_pv = 1
    coef_batteries = 1
    coef_cable_length = 2
    r_ohm_per_km = 0.641
    x_ohm_per_km = 0.101
    coef_trafo = 1
    battery_efficiency_one_trip = 0.95

    # Load network data
    sb_scenario = "1-LV-rural1--0-no_sw"
    _net = sb.get_simbench_net(sb_scenario)

    # Modified impedance
    _net["line"].loc[:, "r_ohm_per_km"] = [r_ohm_per_km] * len(_net["line"])
    _net["line"].loc[:, "x_ohm_per_km"] = [x_ohm_per_km] * len(_net["line"])
    _net["line"].loc[:, "c_nf_per_km"] = [0] * len(_net["line"])
    _net["line"].loc[:, "length_km"] *= coef_cable_length

    # Load active power two participant at each node
    # https://selectra.info/energie/guides/conso/consommation-moyenne-electricite
    # Consumption of a 70m2 apartment 11 200 kWh/an
    _dfp = pd.read_pickle(path + "examples/data/fr_quoilin.pickle")

    temp = (_dfp.sum(axis=0) / 4).to_frame("energy")
    temp = temp[temp.energy.between(8500, 14000)].copy()  # 9500 - 12500 (version with 13 households)
    assert len(temp) == len(_net.load.bus.to_list()) * 2  # 1
    assert int(temp.energy.mean()) == 10599  # 10873
    _dfp = _dfp.loc[:, temp.index.to_list()].clip(upper=max_p_kw).copy() * coef_load
    print(f"Max loading = {round(_dfp.sum(axis=1).max(), 3)}")
    del temp

    load_buses = _net.load.bus.to_list()
    load_buses.extend([v + 0.1 for v in load_buses])
    _dfp.columns = load_buses

    # Load reactive with a constant power factor
    _dfq = (_dfp * np.tan(np.arccos(0.9))).copy()
    assert not _dfq.isnull().values.any(), "Oops some values are null"
    assert _dfp[9].iloc[56] / np.sqrt(_dfp[9].iloc[56] ** 2 + _dfq[9].iloc[56] ** 2) == pytest.approx(0.9, abs=1e-6)

    # Production profiles in active power : https://re.jrc.ec.europa.eu/pvg_tools/en/
    pv = pd.read_csv(path + "examples/data/Timeseries_45.752_4.949_SA2_1kWp_crystSi_14_37deg_-2deg_2018_2020.csv",
                     parse_dates=[0], index_col=[0], header=8, skipfooter=13, engine='python',
                     date_parser=lambda x: datetime.strptime(x, '%Y%m%d:%H%M'))

    # P from Watt to KW (data every hour starting at 00:10 am)
    pv = pv[['P']] / 1000
    pv.columns = ["pv_1kw"]
    pv = pv.loc['2018-12-31 23:10:00':'2020-01-01 00:10:00', :]
    pv = pv.resample('1T').interpolate('time')
    pv = pv.resample('15T').last()
    pv = pv.loc["2019-01-01 00:00:00":"2019-12-31 23:45:00", :]

    # Aggregate production and consumption
    pv_capacity = pv_capacity_kw * coef_pv  # kW
    _df = _dfp.copy()
    for col in _df.columns:
        _df[col] -= pv_capacity * pv["pv_1kw"]

    # Batteries
    _size_kwh = battery_capacity_kwh * coef_batteries
    _specs = {
        "min_kw": _size_kwh / 2,
        "max_kw": _size_kwh / 2,
        "max_kwh": _size_kwh,
        "init_kwh": _size_kwh * 0.01,
        "eta": battery_efficiency_one_trip,
        "offset": 0.01}

    # Transformator
    # Understanding the [impedance voltage](https://www.idc-online.com/technical_references/pdfs/
    # electrical_engineering/Transformer_Impedance.pdf) The voltage during the short-circuit test is divided
    # by the reference voltage. This is also equivalent to i_base * z_eq divided by i_base * z_base, thus z_eq,pu
    # zeq = zeq_pu * z_trafo (also zeq_pu = (i_trafo * zeq) / (i_trafo * z_trafo) = vk / v_trafo)
    zeq = ((_net["trafo"].loc[0, "vk_percent"] / 100)
           * (_net["trafo"].loc[0, "vn_lv_kv"] ** 2 / _net["trafo"].loc[0, "sn_mva"]))  # Z_trafo

    req = ((_net["trafo"].loc[0, "vkr_percent"] / 100)
           * (_net["trafo"].loc[0, "vn_lv_kv"] ** 2 / _net["trafo"].loc[0, "sn_mva"]))

    xeq = np.sqrt(zeq ** 2 - req ** 2)
    print(f"Transformator eq. impedance (secondary) = {complex(round(req, 3), round(xeq, 3))} Ohm (x{coef_trafo}).")
    line_length = 0.01
    max_i_ka = _net["trafo"].loc[0, "sn_mva"] / _net["trafo"].loc[0, "vn_lv_kv"]
    pp.drop_trafos(_net, [0])
    _ = pp.create_line_from_parameters(_net, 42, 3, name="transfo replacement",
                                       length_km=line_length * coef_trafo,
                                       r_ohm_per_km=req / line_length,
                                       x_ohm_per_km=xeq / line_length, c_nf_per_km=0,
                                       r0_ohm_per_km=0, x0_ohm_per_km=0, c0_nf_per_km=0,
                                       max_i_ka=max_i_ka, type="cs", max_loading_percent=100, subnet="LV1.101",
                                       voltLvl=7)
    _net["bus"].loc[42, ["name", "vn_kv", "subnet", "min_vm_pu", "voltLvl", "max_vm_pu"]] = ["LV1.101 Bus 42", 0.4,
                                                                                             "LV1.101", 0.9, 7, 1.100]

    # # Change time resolution
    # _dfp = _dfp.resample("30T").mean()
    # _dfq = _dfq.resample("30T").mean()
    # _df = _df.resample("30T").mean()
    return _net, _dfp, _dfq, _df, _specs


def get_day_from_consumption_level(_df, highest_consumption: bool, hours=5, minutes=0, nb_hours=36):
    if highest_consumption:
        idx = _df.sum(axis=1).idxmax()
    else:
        idx = _df.sum(axis=1).idxmin()
    start = datetime(idx.year, idx.month, idx.day, hours, minutes, 0)
    end = start + timedelta(hours=nb_hours)
    return _df.loc[start:end, :].copy()

# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
#
# def convert_load_data_from_h5_to_pickle():
#     filename = ("..." +
#                 ".../Synthetic.Household.Profiles.h5")
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         _df = pd.read_hdf(filename, "TimeSeries")
#
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         houseinfo = pd.read_hdf(filename, "HouseInfo")
#     mask = (houseinfo.Country == "France") & (houseinfo.Nmonths == "12")
#     _df = _df.loc[:, [str(a) for a in houseinfo[mask].pdIndex.tolist()]]
#
#     _df["Index"] = pd.date_range(
#         start="01-01-2019 00:00:00", end="31-12-2019 23:45:00", freq="15T"
#     )
#     _df.set_index("Index", drop=True, inplace=True)
#     _df.to_pickle("inputs/fr_quoilin.pickle")
#     return True
# _ = convert_load_data_from_h5_to_pickle()
