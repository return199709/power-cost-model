"""Timing engine, unserved energy pricing, and upgrade lines."""

import numpy as np
import pandas as pd
import pytest

from dcop.grid.upgrades import upgrade_ledger_lines
from dcop.io import load_inputs
from dcop.ledger import LineFactory
from dcop.schemas import UpgradeExposure, UpgradeItem
from dcop.timing import btm_lead_months, build_timing, grid_lead_months, unserved_lines


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


def item(**kw) -> UpgradeItem:
    base = dict(name="x", cost_usd_p50=40e6, cost_usd_p90=60e6, probability_triggered=1.0, allocation_pct=1.0,
                ciac=False, refund_schedule=None, lead_time_months=30, on_critical_path=True)
    base.update(kw)
    return UpgradeItem(**base)


def exposure(*items) -> UpgradeExposure:
    return UpgradeExposure(site_id="SITE-A", study_source="test", items=list(items))


def test_grid_lead_is_the_longest_critical_path_item():
    e = exposure(item(lead_time_months=30), item(lead_time_months=48, on_critical_path=False), item(lead_time_months=36))
    assert grid_lead_months(e) == 36
    assert grid_lead_months(None) == 0
    assert grid_lead_months(exposure(item(on_critical_path=False))) == 0


def test_btm_lead_is_permit_then_build_in_parallel_with_lateral(inputs):
    plant, gas = inputs.plants["RECIP-N2"], inputs.gas["PLACEHOLDER-HUB"]
    assert btm_lead_months(plant, gas) == 24  # 9 + 15 beats the 18-month lateral
    assert btm_lead_months(plant.model_copy(update={"permit_lead_months": 18}), gas) == 33
    assert btm_lead_months(plant, gas.model_copy(update={"lateral_lead_months": 40})) == 40


def test_grid_timing_dates_and_mw_online(inputs):
    site, cfg = inputs.sites["SITE-A"], inputs.global_cfg
    t = build_timing("grid", site, cfg.clock_start_date, 30, np.inf, cfg.ops_years)
    assert t.first_power_date == pd.Timestamp("2029-07-01")  # 2027-01-01 + 30 months
    # ramp: 25 MW from 2029-Q1, 50 from 2029-Q3, 100 from 2030-Q1
    assert t.mw_online[pd.Timestamp("2029-01-01")] == 0.0
    assert t.mw_online[pd.Timestamp("2029-04-01")] == 0.0
    assert t.mw_online[pd.Timestamp("2029-07-01")] == 50.0  # power arrives into the 50 MW step
    assert t.full_power_date == pd.Timestamp("2030-01-01")


def test_capacity_limited_path_never_reaches_full_power(inputs):
    site, cfg = inputs.sites["SITE-A"], inputs.global_cfg
    t = build_timing("btm", site, cfg.clock_start_date, 24, 60.0, cfg.ops_years)
    assert t.mw_online.max() == 60.0
    assert pd.isna(t.full_power_date)


def test_delay_and_availability_cost_hand_calc():
    # 10 MW flat through 2029. Power arrives 2029-07-01. Jan-Jun is 181 days x 24 h = 4,344 h.
    index = pd.date_range("2029-01-01", "2030-01-01", freq="h", inclusive="left")
    load = pd.Series(10_000.0, index=index)
    fp = pd.Timestamp("2029-07-01")
    served = np.where(index >= fp, load.to_numpy(), 0.0)
    delivered = served * 0.9
    factory = LineFactory("S", "btm", "base", {2029: 1.0})
    lines = unserved_lines(factory, load, served, delivered, fp, 100.0)
    by = {(ln.category, ln.subcategory): ln.usd_real for ln in lines}
    assert by[("delay", None)] == pytest.approx(10 * 4_344 * 100)  # 4,344,000
    # After power: 184 days x 24 h = 4,416 h x 10 MW x 10% unserved = 4,416 MWh
    assert by[("unserved_energy", "availability")] == pytest.approx(4_416 * 100)
    assert ("unserved_energy", "capacity_shortfall") not in by


def test_capacity_shortfall_is_priced_separately():
    index = pd.date_range("2029-01-01", "2029-02-01", freq="h", inclusive="left")
    load = pd.Series(10_000.0, index=index)
    served = np.full(len(index), 6_000.0)  # capped at 6 MW from the start
    lines = unserved_lines(LineFactory("S", "btm", "b", {}), load, served, served, pd.Timestamp("2029-01-01"), 100.0)
    assert [(ln.category, ln.subcategory) for ln in lines] == [("unserved_energy", "capacity_shortfall")]
    assert lines[0].usd_real == pytest.approx(4 * 744 * 100)  # 4 MW short for 744 h


def test_unserved_energy_needs_a_value():
    index = pd.date_range("2029-01-01", "2029-01-08", freq="h", inclusive="left")
    load = pd.Series(1_000.0, index=index)
    with pytest.raises(ValueError, match="value_of_unserved"):
        unserved_lines(LineFactory("S", "grid", "b", {}), load, np.zeros(len(index)), np.zeros(len(index)),
                       pd.Timestamp("2030-01-01"), None)


def test_no_unserved_energy_needs_no_value():
    index = pd.date_range("2029-01-01", "2029-01-08", freq="h", inclusive="left")
    load = pd.Series(1_000.0, index=index)
    assert unserved_lines(LineFactory("S", "grid", "b", {}), load, load.to_numpy(), load.to_numpy(),
                          pd.Timestamp("2029-01-01"), None) == []


def test_upgrade_expected_value_and_refunds_hand_calc(inputs):
    # 40M x P 0.5 x allocation 0.5 = 10M, paid 2029 (2027-01-01 + 30 months).
    # Refunds 50% after 3 years and 50% after 6: -5M in 2032 and -5M in 2035.
    e = exposure(item(probability_triggered=0.5, allocation_pct=0.5, refund_schedule=[(3, 0.5), (6, 0.5)]))
    lines = upgrade_ledger_lines(LineFactory("S", "grid", "b", {}), e, inputs.global_cfg)
    got = [(ln.category, ln.year, ln.usd_real, ln.is_exposure) for ln in lines]
    assert got == [("upgrade_capex", 2029, 10e6, True), ("upgrade_refund", 2032, -5e6, True), ("upgrade_refund", 2035, -5e6, True)]
    assert lines[0].is_capex and not lines[1].is_capex


def test_ciac_items_are_not_refunded_and_confirmed_items_are_not_exposure(inputs):
    e = exposure(item(ciac=True, refund_schedule=[(3, 1.0)]))
    lines = upgrade_ledger_lines(LineFactory("S", "grid", "b", {}), e, inputs.global_cfg)
    assert [ln.category for ln in lines] == ["upgrade_capex"]
    assert not lines[0].is_exposure


def test_refunds_after_the_horizon_are_dropped(inputs):
    e = exposure(item(refund_schedule=[(20, 1.0)]))
    lines = upgrade_ledger_lines(LineFactory("S", "grid", "b", {}), e, inputs.global_cfg)
    assert [ln.category for ln in lines] == ["upgrade_capex"]


def test_refund_schedule_cannot_exceed_100_percent():
    with pytest.raises(ValueError, match="more than 100%"):
        item(refund_schedule=[(1, 0.7), (2, 0.7)])
