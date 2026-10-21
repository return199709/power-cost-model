"""BTM path: availability, plant, gas. Expected values are worked by hand in comments."""

import numpy as np
import pandas as pd
import pytest

from dcop.btm import availability as av
from dcop.btm.gas import curve_at, gas_ledger_lines, monthly_price_components, online_fraction
from dcop.btm.plant import firm_mw, heat_rate_mmbtu_mwh, installed_kw, plant_ledger_lines, unit_mw_site
from dcop.finance import npv
from dcop.io import load_inputs
from dcop.ledger import LineFactory, capex_annuity_lines


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


@pytest.fixture
def plant(inputs):
    return inputs.plants["RECIP-N2"]


@pytest.fixture
def site(inputs):
    return inputs.sites["SITE-A"]


@pytest.fixture
def gas(inputs):
    return inputs.gas["PLACEHOLDER-HUB"]


def test_availability_table_matches_binomial_closed_form():
    # n = 3 units at 0.9: P(3) = 0.729; P(>=2) = 0.729 + 3 x 0.81 x 0.1 = 0.972; P(>=1) = 1 - 0.001
    t = av.availability_table(3, 0.9)
    assert t.tolist() == pytest.approx([1.0, 0.999, 0.972, 0.729])


def test_eaf_hand_calc(plant):
    # (1 - 14/365) x (1 - 0.03) = 0.9616438 x 0.97
    assert av.eaf(plant) == pytest.approx(0.9616438 * 0.97, abs=1e-6)


def test_required_units_rounds_up():
    out = av.required_units(np.array([0.0, 0.1, 17.0, 17.13, 34.2]), 17.12)
    assert out.tolist() == [0, 1, 1, 2, 2]


def test_site_derates_and_firm_capacity(plant, site):
    # ambient: 0.02 x (35 - 15) / 10 = 4%; elevation: 0.03 x 0.3 = 0.9%
    # unit = 18 x 0.96 x 0.991 = 17.12448 MW; N+2 of 8 units leaves 6 firm units
    assert unit_mw_site(plant, site) == pytest.approx(17.12448)
    assert firm_mw(plant, site) == pytest.approx(6 * 17.12448)
    assert installed_kw(plant) == 144_000.0


def test_delivered_gas_price_hand_calc(tmp_path, gas):
    # (HH 3.50 + basis -0.30) / (1 - 0.02) + 0.25 transport = 3.265306 + 0.25
    for name, v in [("hh.csv", 3.5), ("basis.csv", -0.3)]:
        pd.DataFrame({"month": ["2029-01", "2029-02"], "usd_mmbtu": v}).to_csv(tmp_path / name, index=False)
    g = gas.model_copy(update={"henry_hub_curve_ref": "hh.csv", "basis_curve_ref": "basis.csv"})
    prices = monthly_price_components(g, tmp_path, np.array([2029 * 12, 2029 * 12 + 1]))
    assert prices["henry_hub"].iloc[0] == pytest.approx(3.5 / 0.98)
    assert prices["basis"].iloc[0] == pytest.approx(-0.3 / 0.98)
    assert prices.sum(axis=1).iloc[0] == pytest.approx(3.515306, abs=1e-6)


def test_curve_holds_last_value_after_end_and_rejects_before_start():
    curve = pd.Series([1.0, 2.0], index=[24_000, 24_001])
    assert curve_at(curve, np.array([24_001, 24_005]), "c").tolist() == [2.0, 2.0]
    with pytest.raises(ValueError, match="starts"):
        curve_at(curve, np.array([23_999]), "c")


def test_online_fraction_prorates_first_year():
    fp = pd.Timestamp("2029-07-01")
    assert online_fraction(2028, fp) == 0.0
    assert online_fraction(2029, fp) == pytest.approx(184 / 365)  # Jul 1 to Dec 31
    assert online_fraction(2030, fp) == 1.0


def test_gas_lines_fuel_and_fixed_transport_hand_calc(tmp_path, inputs, gas):
    # 1 MMBtu per hour through 2030 = 8,760 MMBtu. Gas costs 3.515306 delivered.
    for name, v in [("hh.csv", 3.5), ("basis.csv", -0.3)]:
        pd.DataFrame({"month": pd.period_range("2029-01", "2043-12", freq="M").astype(str), "usd_mmbtu": v}).to_csv(
            tmp_path / name, index=False
        )
    g = gas.model_copy(update={"henry_hub_curve_ref": "hh.csv", "basis_curve_ref": "basis.csv", "lateral_capex_usd": 0})
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    factory = LineFactory("S", "btm", "base", {2030: 1000.0})
    lines = gas_ledger_lines(factory, g, inputs.global_cfg, tmp_path, index, np.ones(len(index)), pd.Timestamp("2030-01-01"))
    fuel = sum(ln.usd_real for ln in lines if ln.category == "fuel")
    assert fuel == pytest.approx(8760 * 3.515306, abs=0.01)
    # MDQ = 1 MMBtu/h x 24 = 24 MMBtu/d; FT 0.30 x 24 x 365 = 2,628 a year
    ft = [ln for ln in lines if ln.category == "transport_fixed" and ln.year == 2030]
    assert ft[0].usd_real == pytest.approx(2_628.0)


def test_capex_annuity_pv_equals_principal_discounted_to_base_year(inputs):
    cfg = inputs.global_cfg  # base 2026, 8%, window 2029-2043
    factory = LineFactory("S", "btm", "base", {})
    lines = capex_annuity_lines(factory, "plant_capex", 1_000_000.0, cfg, first_year=2030)
    assert [ln.year for ln in lines] == list(range(2030, 2044))
    assert all(ln.is_capex for ln in lines)
    # Spent at the end of 2029, three years after the base year
    pv = npv({ln.year: ln.usd_real for ln in lines}, cfg.discount_rate, cfg.base_year)
    assert pv == pytest.approx(1_000_000 / 1.08**3)
    # Later in service amortizes over fewer years, so each payment is larger
    later = capex_annuity_lines(factory, "plant_capex", 1_000_000.0, cfg, first_year=2035)
    assert later[0].usd_real > lines[0].usd_real


def test_capex_after_horizon_is_an_error(inputs):
    with pytest.raises(ValueError, match="after the horizon"):
        capex_annuity_lines(LineFactory("S", "btm", "b", {}), "x", 1.0, inputs.global_cfg, 2050)


def test_heat_rate_part_load_and_degradation(tmp_path, plant, site):
    pd.DataFrame({"load_fraction": [0.5, 1.0], "heat_rate_multiplier": [1.15, 1.0]}).to_csv(tmp_path / "pl.csv", index=False)
    p = plant.model_copy(update={"heat_rate_part_load_curve_ref": "pl.csv"})
    unit = unit_mw_site(p, site)
    index = pd.DatetimeIndex(["2031-06-01 12:00"])
    # One unit at 75% load: multiplier 1.075. Two years after the 2029 start: 1 + 0.003 x 2 = 1.006.
    hr = heat_rate_mmbtu_mwh(p, tmp_path, np.array([0.75 * unit]), np.array([1]), unit, index, pd.Timestamp("2029-01-01"))
    assert hr[0] == pytest.approx(8.8 * 1.075 * 1.006)


def test_plant_lines_hand_calc(inputs, plant):
    cfg = inputs.global_cfg
    factory = LineFactory("S", "btm", "base", {2030: 100_000.0})
    p = plant.model_copy(update={"co2_price_usd_t": 50.0})
    lines = plant_ledger_lines(factory, p, cfg, pd.Timestamp("2030-01-01"), {2030: 100_000.0}, {2030: 900_000.0})
    by = lambda cat, sub=None, yr=2030: sum(  # noqa: E731
        ln.usd_real for ln in lines if ln.category == cat and ln.year == yr and (sub is None or ln.subcategory == sub)
    )
    assert by("fixed_om") == pytest.approx(25 * 144_000)  # 3.6M, full year
    assert by("variable_om", "variable") == pytest.approx(6 * 100_000)
    assert by("variable_om", "major_maintenance") == pytest.approx(8 * 100_000)
    assert by("carbon") == pytest.approx(900_000 * 0.0531 * 50)  # 2,389,500
    # Capital: 1,800 $/kW x 144,000 kW = 259.2M principal, recovered over 2030-2043
    pv = npv({ln.year: ln.usd_real for ln in lines if ln.category == "plant_capex"}, 0.08, 2026)
    assert pv == pytest.approx(259_200_000 / 1.08**3)


def test_salvage_is_a_negative_residual_line(inputs, plant):
    cfg = inputs.global_cfg.model_copy(update={"btm_salvage_pct_capex": 0.1})
    lines = plant_ledger_lines(LineFactory("S", "btm", "b", {}), plant, cfg, pd.Timestamp("2030-01-01"), {}, {},
                               lateral_principal_usd=15_000_000)
    residual = [ln for ln in lines if ln.category == "residual"]
    # 10% x (259.2M plant + 21.6M emissions + 15M lateral)
    assert residual[0].usd_real == pytest.approx(-29_580_000)
    assert residual[0].year == 2043
