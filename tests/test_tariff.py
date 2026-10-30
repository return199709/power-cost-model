"""Bill replication tests (spec §9, milestone M0).

The tariff below is SYNTHETIC, not a real utility's. The expected values were
worked out by hand (arithmetic shown in comments) and match the figures in
config/tariffs.yaml. Replace with a published tariff and a sample bill from a
real utility to close the M0 acceptance criterion.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dcop.grid.tariff import compute_monthly_bills
from dcop.schemas import DemandCharge, Rider, Tariff, TOUBlock

WINTER = [1, 2, 3, 4, 5, 10, 11, 12]
SUMMER = [6, 7, 8, 9]
W_ON = list(range(7, 21))  # 14 hours
W_OFF = [h for h in range(24) if h not in W_ON]
S_ON = list(range(12, 20))  # 8 hours
S_OFF = [h for h in range(24) if h not in S_ON]


def make_tariff(**overrides) -> Tariff:
    base = dict(
        tariff_id="SYNTH",
        utility="Synthetic",
        name="Synthetic TOU",
        effective_date=date(2027, 1, 1),
        customer_charge_usd_mo=1500.0,
        energy_charges=[
            TOUBlock(season="summer", period="on_peak", months=SUMMER, hours=S_ON, days="weekday", usd_per_mwh=62),
            TOUBlock(season="summer", period="off_peak", months=SUMMER, hours=S_OFF, days="weekday", usd_per_mwh=38),
            TOUBlock(season="summer", period="off_peak", months=SUMMER, days="weekend", usd_per_mwh=38),
            TOUBlock(season="winter", period="on_peak", months=WINTER, hours=W_ON, days="weekday", usd_per_mwh=48),
            TOUBlock(season="winter", period="off_peak", months=WINTER, hours=W_OFF, days="weekday", usd_per_mwh=34),
            TOUBlock(season="winter", period="off_peak", months=WINTER, days="weekend", usd_per_mwh=34),
        ],
        demand_charges=[
            DemandCharge(name="summer_demand", season="summer", months=SUMMER, usd_per_kw_mo=14.50, ratchet_pct=0.8),
            DemandCharge(name="winter_demand", season="winter", months=WINTER, usd_per_kw_mo=9.75, ratchet_pct=0.8),
        ],
        riders=[
            Rider(name="fuel_adjustment", kind="usd_per_mwh", value=2.10),
            Rider(name="transmission", kind="usd_per_kw_mo", value=1.35),
            Rider(name="dsm", kind="pct_of_bill", value=0.015),
        ],
    )
    base.update(overrides)
    return Tariff(**base)


def flat_load(start: str, end: str, kw: float, freq: str = "15min") -> pd.Series:
    idx = pd.date_range(start, end, freq=freq, inclusive="left")
    return pd.Series(kw, index=idx)


def test_winter_month_replicates_hand_bill_to_the_cent():
    # January 2027: 744 h, 21 weekdays x 14 on-peak h = 294 on-peak h, 450 off-peak h.
    load = flat_load("2027-01-01", "2027-02-01", 40_000.0)
    load["2027-01-12 14:00"] = 52_000.0  # one 15-min spike, Tuesday, on-peak
    bill = compute_monthly_bills(make_tariff(), load)
    cats = bill.by_category()

    # MWh: 40 x 744 + 12 MW x 0.25 h = 29,763. On-peak 40 x 294 + 3 = 11,763; off-peak 18,000.
    assert bill.mwh_by_month.iloc[0] == pytest.approx(29_763.0)
    # Energy = 11,763 x 48 + 18,000 x 34 = 564,624 + 612,000
    energy = bill.lines[bill.lines.category == "energy"]["usd"].sum()
    assert energy == pytest.approx(1_176_624.00, abs=0.005)
    # Demand = 52,000 kW x 9.75
    assert cats["demand"] == pytest.approx(507_000.00, abs=0.005)
    # Fees: customer 1,500 + fuel 2.10 x 29,763 (62,502.30) + transmission 1.35 x 52,000 (70,200)
    #       + dsm 1.5% x (1,500 + 1,176,624 + 507,000 + 62,502.30 + 70,200 = 1,817,826.30) = 27,267.3945
    assert cats["riders_fees"] == pytest.approx(1_500 + 62_502.30 + 70_200 + 27_267.3945, abs=0.005)
    # Total 1,845,093.6945
    assert bill.total() == pytest.approx(1_845_093.69, abs=0.005)
    assert "minimum_bill" not in cats.index


def test_summer_month_replicates_hand_bill_to_the_cent():
    # July 2027: 744 h, 22 weekdays x 8 on-peak h = 176 on-peak h, 568 off-peak h. 40 MW flat.
    bill = compute_monthly_bills(make_tariff(), flat_load("2027-07-01", "2027-08-01", 40_000.0))
    cats = bill.by_category()
    # Energy = 40 x 176 x 62 + 40 x 568 x 38 = 436,480 + 863,360
    assert cats["energy"] == pytest.approx(1_299_840.00, abs=0.005)
    assert cats["demand"] == pytest.approx(580_000.00, abs=0.005)  # 40,000 x 14.50
    # Fees: 1,500 + fuel 2.10 x 29,760 (62,496) + transmission 1.35 x 40,000 (54,000)
    #       + dsm 1.5% x 1,997,836 = 29,967.54
    assert cats["riders_fees"] == pytest.approx(1_500 + 62_496 + 54_000 + 29_967.54, abs=0.005)
    assert bill.total() == pytest.approx(2_027_803.54, abs=0.005)


def test_ratchet_sets_next_month_billing_demand():
    # Jan peaks at 52 MW; Feb is flat 40 MW. Feb bills 0.8 x 52,000 = 41,600 kW.
    load = pd.concat(
        [
            flat_load("2027-01-01", "2027-02-01", 40_000.0),
            flat_load("2027-02-01", "2027-03-01", 40_000.0),
        ]
    )
    load["2027-01-12 14:00"] = 52_000.0
    bill = compute_monthly_bills(make_tariff(), load)
    feb_demand = bill.lines.query("category == 'demand' and month == 2")["usd"].sum()
    assert feb_demand == pytest.approx(41_600 * 9.75, abs=0.005)
    assert bill.billing_kw_by_month.iloc[1] == pytest.approx(41_600.0)


def test_prior_demand_history_seeds_the_ratchet():
    load = flat_load("2027-02-01", "2027-03-01", 40_000.0)
    bill = compute_monthly_bills(make_tariff(), load, prior_demand_kw=[60_000.0])
    assert bill.billing_kw_by_month.iloc[0] == pytest.approx(48_000.0)  # 0.8 x 60,000


def test_ratchet_expires_after_lookback_window():
    tariff = make_tariff(
        demand_charges=[
            DemandCharge(name="d", season="all", months=list(range(1, 13)), usd_per_kw_mo=10.0, ratchet_pct=1.0, ratchet_lookback_months=2)
        ]
    )
    load = pd.concat(
        [flat_load(f"2027-0{m}-01", f"2027-0{m + 1}-01", kw) for m, kw in [(1, 50_000.0), (2, 10_000.0), (3, 10_000.0), (4, 10_000.0)]]
    )
    kw = compute_monthly_bills(tariff, load).billing_kw_by_month.to_numpy()
    # Jan 50k; Feb and Mar still see Jan inside the 2-month window; Apr does not.
    assert kw.tolist() == pytest.approx([50_000.0, 50_000.0, 50_000.0, 10_000.0])


def test_demand_basis_15min_vs_ncp_vs_cp():
    load = flat_load("2027-01-01", "2027-02-01", 40_000.0)
    load["2027-01-12 14:00"] = 52_000.0  # one 15-min interval inside hour 14
    tariff = lambda basis: make_tariff(  # noqa: E731
        demand_charges=[DemandCharge(name="d", season="winter", months=WINTER, usd_per_kw_mo=10.0, basis=basis)],
        riders=[],
    )
    kw15 = compute_monthly_bills(tariff("15min"), load).billing_kw_by_month.iloc[0]
    ncp = compute_monthly_bills(tariff("ncp"), load).billing_kw_by_month.iloc[0]
    cp = compute_monthly_bills(
        tariff("cp"), load, cp_hours=pd.DatetimeIndex(["2027-01-20 18:00"])
    ).billing_kw_by_month.iloc[0]
    assert kw15 == pytest.approx(52_000.0)
    assert ncp == pytest.approx(43_000.0)  # (52,000 + 3 x 40,000) / 4
    assert cp == pytest.approx(40_000.0)


def test_cp_basis_without_cp_hours_raises():
    t = make_tariff(demand_charges=[DemandCharge(name="d", season="w", months=WINTER, usd_per_kw_mo=10.0, basis="cp")])
    with pytest.raises(ValueError, match="cp_hours"):
        compute_monthly_bills(t, flat_load("2027-01-01", "2027-02-01", 1000.0))


def test_minimum_bill_trues_up_a_low_usage_month():
    # 1 MW flat in January. Computed bill 44,228.016:
    #   energy 294 x 48 + 450 x 34 = 29,412; demand 1,000 x 9.75 = 9,750; customer 1,500;
    #   fuel 2.10 x 744 = 1,562.40; transmission 1,350; dsm 1.5% x 43,574.40 = 653.616.
    # Floor = 0.6 x (1,500 + 9.75 x 40,000) = 234,900, so the true-up is 190,671.984.
    tariff = make_tariff(min_bill_pct=0.6)
    bill = compute_monthly_bills(tariff, flat_load("2027-01-01", "2027-02-01", 1_000.0), contract_demand_kw=40_000.0)
    assert bill.by_category()["minimum_bill"] == pytest.approx(190_671.984, abs=0.005)
    assert bill.total() == pytest.approx(234_900.00, abs=0.005)


def test_minimum_bill_not_triggered_when_bill_exceeds_floor():
    bill = compute_monthly_bills(
        make_tariff(min_bill_pct=0.6), flat_load("2027-07-01", "2027-08-01", 40_000.0), contract_demand_kw=40_000.0
    )
    assert "minimum_bill" not in bill.by_category().index


def test_unenergized_months_are_not_billed():
    load = pd.concat([flat_load("2027-01-01", "2027-02-01", 0.0), flat_load("2027-02-01", "2027-03-01", 40_000.0)])
    bill = compute_monthly_bills(make_tariff(), load)
    assert set(bill.lines["month"]) == {2}


def test_energy_blocks_must_cover_every_interval_once():
    gap = make_tariff(energy_charges=[TOUBlock(season="s", period="p", months=[1], usd_per_mwh=10)])
    with pytest.raises(ValueError, match="exactly once"):
        compute_monthly_bills(gap, flat_load("2027-02-01", "2027-03-01", 1000.0))
    overlap = make_tariff(
        energy_charges=[
            TOUBlock(season="s", period="a", months=[1], usd_per_mwh=10),
            TOUBlock(season="s", period="b", months=[1], usd_per_mwh=20),
        ]
    )
    with pytest.raises(ValueError, match="exactly once"):
        compute_monthly_bills(overlap, flat_load("2027-01-01", "2027-02-01", 1000.0))


def test_hourly_data_matches_15min_when_load_is_flat():
    a = compute_monthly_bills(make_tariff(), flat_load("2027-07-01", "2027-08-01", 40_000.0, "15min"))
    b = compute_monthly_bills(make_tariff(), flat_load("2027-07-01", "2027-08-01", 40_000.0, "h"))
    assert a.total() == pytest.approx(b.total())


def test_market_indexed_tariff_not_supported_yet():
    with pytest.raises(NotImplementedError):
        compute_monthly_bills(make_tariff(market_indexed=True), flat_load("2027-01-01", "2027-02-01", 1.0))


def test_negative_load_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        compute_monthly_bills(make_tariff(), flat_load("2027-01-01", "2027-02-01", -1.0))
