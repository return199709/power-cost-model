import pandas as pd
import pytest

from dcop.ledger import Ledger, ReconciliationError, reconcile
from dcop.schemas import LedgerLine


def line(year, category, usd, mwh=1000.0, site="A", path="grid", scen="base", **kw):
    return LedgerLine(
        site_id=site, path_id=path, scenario_id=scen, year=year,
        category=category, usd_real=usd, mwh_delivered=mwh, **kw,
    )


def make_ledger():
    return Ledger(
        [
            line(2027, "energy", 40_000), line(2027, "demand", 10_000),
            line(2028, "energy", 60_000, mwh=2000.0), line(2028, "demand", 20_000, mwh=2000.0),
            line(2028, "upgrade_capex", 30_000, mwh=2000.0, is_capex=True),
        ]
    )


def test_aggregate_by_category_uses_all_delivered_mwh_as_denominator():
    agg = make_ledger().aggregate(["category"]).set_index("category")
    # MWh counted once per year: 1000 + 2000 = 3000
    assert agg.loc["energy", "usd"] == 100_000
    assert agg.loc["energy", "usd_per_mwh"] == pytest.approx(100_000 / 3000)
    # capex appears only in 2028 but is still spread over all delivered MWh
    assert agg.loc["upgrade_capex", "usd_per_mwh"] == pytest.approx(30_000 / 3000)
    assert agg["usd_per_mwh"].sum() == pytest.approx(160_000 / 3000)
    assert agg["share"].sum() == pytest.approx(1.0)


def test_aggregate_by_year_and_category_uses_that_years_mwh():
    agg = make_ledger().aggregate(["year", "category"])
    row = agg[(agg.year == 2028) & (agg.category == "energy")].iloc[0]
    assert row.usd_per_mwh == pytest.approx(60_000 / 2000)
    assert row.share == pytest.approx(60_000 / 110_000)


def test_usd_per_kw_yr_counts_kw_years():
    agg = make_ledger().aggregate(["category"], kw_by_site={"A": 500.0}).set_index("category")
    # two site-years x 500 kW = 1000 kW-years
    assert agg.loc["energy", "usd_per_kw_yr"] == pytest.approx(100.0)


def test_inconsistent_mwh_within_a_year_is_rejected():
    led = Ledger([line(2027, "energy", 1.0, mwh=1000.0), line(2027, "demand", 1.0, mwh=999.0)])
    with pytest.raises(ValueError, match="inconsistent"):
        led.validate_mwh()


def test_reconcile_passes_and_detects_mismatch():
    led = make_ledger()
    ok = pd.DataFrame(
        {"site_id": ["A"], "path_id": ["grid"], "scenario_id": ["base"], "total_cost_usd_real": [160_000.0]}
    )
    reconcile(led, ok)
    bad = ok.assign(total_cost_usd_real=159_000.0)
    with pytest.raises(ReconciliationError):
        reconcile(led, bad)


def test_sign_convention_refunds_are_negative():
    led = Ledger([line(2027, "upgrade_capex", 100.0), line(2029, "upgrade_refund", -40.0)])
    assert led.totals()["usd_real"].iloc[0] == pytest.approx(60.0)
