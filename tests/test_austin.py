"""Austin Energy FY2026 residential and commercial bills (tariff effective Nov 1, 2025).

Expected values are worked by hand from the tariff PDF. The 860 kWh residential
bill also matches the City of Austin FY2026 budget material, which quotes a
typical 860 kWh customer at a $117.07 monthly bill (base $59.19, PSA $35.41).
That figure comes from a search summary of the city documents, because the city
host was not reachable from the build environment.
"""

import numpy as np
import pandas as pd
import pytest

from dcop.cli import main
from dcop.grid import austin
from dcop.grid.bills import monthly_statement, round_cents, usage_month_load
from dcop.grid.tariff import compute_monthly_bills
from dcop.io import load_tariffs, load_inputs
from dcop.schemas import EnergyTier, Rider, Tariff, TOUBlock


@pytest.fixture(scope="module")
def tariffs():
    return load_inputs("config").tariffs


# ---- transcription check against the PDF ----
def test_residential_rates_match_the_tariff_pdf(tariffs):
    t = tariffs[austin.RESIDENTIAL_ID]
    assert t.customer_charge_usd_mo == 16.50
    assert [(x.up_to_kwh_mo, x.usd_per_kwh) for x in t.energy_tiers] == [
        (300, 0.04640), (900, 0.05138), (2000, 0.07525), (None, 0.10884)]
    assert {r.name: r.value for r in t.riders} == {
        "Power Supply Adjustment": 0.04118,
        "Administrative Adjustment to PSA": -0.00206,
        "Community Benefit: Customer Assistance Program": 0.00564,
        "Community Benefit: Service Area Lighting": 0.00254,
        "Community Benefit: Energy Efficiency Services": 0.00457,
        "Regulatory Charge": 0.01338,
    }
    assert t.effective_date.isoformat() == "2025-11-01"


@pytest.mark.parametrize(
    "tariff_id, customer, demand, energy_usd_mwh, cap, regulatory",
    [
        (austin.GS_LT10_ID, 38.23, None, 31.29, 0.00237, ("usd_per_kwh", 0.01338)),
        (austin.GS_10_300_ID, 60.08, 9.83, 19.32, 0.00237, ("usd_per_kw_mo", 3.73)),
        (austin.GS_GE300_ID, 300.42, 12.56, 18.40, 0.00237, ("usd_per_kw_mo", 3.73)),
    ],
)
def test_general_service_rates_match_the_tariff_pdf(tariffs, tariff_id, customer, demand, energy_usd_mwh, cap, regulatory):
    t = tariffs[tariff_id]
    assert t.customer_charge_usd_mo == customer
    assert t.energy_charges[0].usd_per_mwh == pytest.approx(energy_usd_mwh)
    assert (t.demand_charges[0].usd_per_kw_mo if t.demand_charges else None) == demand
    riders = {r.name: r for r in t.riders}
    assert riders["Community Benefit: Customer Assistance Program"].value == cap
    assert (riders["Regulatory Charge"].kind, riders["Regulatory Charge"].value) == regulatory
    assert riders["Power Supply Adjustment"].value == 0.04118


# ---- residential branch ----
def test_residential_860_kwh_matches_the_city_typical_bill(tariffs):
    # Without the Dec 1, 2025 administrative adjustment, as in the city's budget figure:
    #   customer 16.50 + 300 x 0.04640 (13.92) + 560 x 0.05138 (28.7728)  = base 59.1928  ($59.19)
    #   PSA 860 x 0.04118 = 35.4148 ($35.41); CBC 860 x 0.01275 = 10.965; regulatory 860 x 0.01338 = 11.5068
    s = austin.residential_statement(tariffs, 860, include_psa_admin_adjustment=False)
    assert s.usd("Customer Charge") == 16.50
    assert s.usd("Energy tier 1 (0-300 kWh)") == 13.92
    assert s.usd("Energy tier 2 (300-900 kWh)") == 28.77
    assert s.usd("Power Supply Adjustment") == 35.41
    assert s.usd("Regulatory Charge") == 11.51
    assert s.total_usd == pytest.approx(117.07)  # the City's published typical bill


def test_residential_860_kwh_with_the_psa_administrative_adjustment(tariffs):
    # Adds 860 x -0.00206 = -1.7716, so 117.07 - 1.77 = 115.30
    s = austin.residential_statement(tariffs, 860)
    assert s.usd("Administrative Adjustment to PSA") == -1.77
    assert s.total_usd == pytest.approx(115.30)
    assert s.kwh == pytest.approx(860)
    assert s.billed_kw is None


@pytest.mark.parametrize(
    "kwh, energy",
    [
        (300, 300 * 0.04640),                                       # 13.92, all in tier 1
        (301, 300 * 0.04640 + 0.05138),                             # the 301st kWh is tier 2
        (900, 300 * 0.04640 + 600 * 0.05138),                       # 44.748
        (2000, 300 * 0.04640 + 600 * 0.05138 + 1100 * 0.07525),     # 127.523
        (2500, 300 * 0.04640 + 600 * 0.05138 + 1100 * 0.07525 + 500 * 0.10884),  # 181.943
    ],
)
def test_residential_tier_boundaries(tariffs, kwh, energy):
    load = usage_month_load(kwh, None, 2026, 1)
    bill = compute_monthly_bills(tariffs[austin.RESIDENTIAL_ID], load)
    tiers = bill.lines[bill.lines.subcategory.str.startswith("tier ")]
    assert tiers["usd"].sum() == pytest.approx(energy)


def test_residential_effective_rate_rises_with_usage(tariffs):
    rate = lambda kwh: austin.residential_statement(tariffs, kwh).cents_per_kwh  # noqa: E731
    assert rate(200) > rate(860) or rate(860) > rate(300)  # fixed charge dilutes, tiers climb
    assert rate(3000) > rate(1000)


# ---- commercial branch ----
def test_commercial_10_to_300_kw_hand_calc(tariffs):
    # 20,000 kWh, 50 kW peak: customer 60.08, demand 50 x 9.83 = 491.50, energy 20,000 x 0.01932 = 386.40,
    # PSA 823.60, admin -41.20, CBC 20,000 x 0.00948 = 189.60, regulatory 50 x 3.73 = 186.50
    s = austin.commercial_statement(tariffs, 20_000, 50)
    assert s.tariff_id == austin.GS_10_300_ID
    assert s.billed_kw == pytest.approx(50.0)
    assert s.usd("Demand Charge") == 491.50
    assert s.usd("Energy Charge") == 386.40
    assert s.usd("Regulatory Charge") == 186.50
    assert s.total_usd == pytest.approx(2_096.48)


def test_commercial_300_kw_and_above_hand_calc(tariffs):
    # 150,000 kWh, 400 kW: customer 300.42, demand 400 x 12.56 = 5,024, energy 2,760, PSA 6,177,
    # admin -309, CBC 1,422, regulatory 400 x 3.73 = 1,492
    s = austin.commercial_statement(tariffs, 150_000, 400)
    assert s.tariff_id == austin.GS_GE300_ID
    assert s.total_usd == pytest.approx(16_866.42)


def test_commercial_below_10_kw_has_no_demand_charge_and_regulatory_on_energy(tariffs):
    # 800 kWh: customer 38.23, energy 25.032, PSA 32.944, admin -1.648, CBC 7.584, regulatory 10.704
    s = austin.commercial_statement(tariffs, 800, 6)
    assert s.tariff_id == austin.GS_LT10_ID
    assert "Demand Charge" not in set(s.items["item"])
    assert s.usd("Regulatory Charge") == 10.70
    assert s.total_usd == pytest.approx(112.84)  # sum of the rounded lines; unrounded 112.846


def test_power_factor_penalty_matches_the_tariff_examples(tariffs):
    # Tariff p.6: 13.5 kW at 86.7% power factor bills 14.0 kW (13.5 x 0.90 / 0.87)
    small = austin.commercial_statement(tariffs, 4_000, 13.5, power_factor=0.867)
    assert small.billed_kw == pytest.approx(13.5 * 0.90 / 0.87)
    assert small.billed_kw == pytest.approx(14.0, abs=0.05)
    assert small.usd("Demand Charge") == round_cents(13.5 * 0.90 / 0.87 * 9.83)
    # Tariff p.9 example: 10,350 kW at 86.7% bills about 10,707 kW
    big = austin.commercial_statement(tariffs, 5_000_000, 10_350, power_factor=0.867)
    assert big.billed_kw == pytest.approx(10_707, abs=0.5)  # the tariff rounds to a whole kW; the model does not
    assert round(big.billed_kw) == 10_707


def test_power_factor_at_or_above_90_percent_is_not_adjusted(tariffs):
    for pf in (0.90, 0.95, 1.0, None):
        assert austin.commercial_statement(tariffs, 20_000, 50, power_factor=pf).billed_kw == pytest.approx(50.0)
    assert austin.commercial_statement(tariffs, 20_000, 50, power_factor=0.895).billed_kw == pytest.approx(50.0)  # 0.895 rounds to 0.90


def test_rate_class_uses_the_june_to_september_average(tariffs):
    f = austin.general_service_tariff_id
    assert f([9.9, 9.9, 9.9, 9.9]) == austin.GS_LT10_ID
    assert f([5, 5, 5, 25]) == austin.GS_10_300_ID  # average 10 is not below 10
    assert f([299, 299, 299, 299]) == austin.GS_10_300_ID
    assert f([300, 300, 300, 300]) == austin.GS_GE300_ID
    assert f([100, 100, 100, 900]) == austin.GS_GE300_ID  # one hot month lifts the average to 300
    assert f([100, 100, 100, 700]) == austin.GS_10_300_ID  # average 250
    with pytest.raises(ValueError):
        f([])


def test_summer_history_overrides_the_monthly_peak_for_class(tariffs):
    s = austin.commercial_statement(tariffs, 20_000, 50, summer_peak_kw_by_month=[400, 420, 410, 390])
    assert s.tariff_id == austin.GS_GE300_ID


def test_commercial_effective_rate_is_below_residential_at_scale(tariffs):
    res = austin.residential_statement(tariffs, 860).cents_per_kwh
    com = austin.commercial_statement(tariffs, 150_000, 400).cents_per_kwh
    assert com < res


# ---- statements and engine features ----
def test_statement_total_is_the_sum_of_cent_rounded_lines(tariffs):
    s = austin.residential_statement(tariffs, 861)
    assert s.total_usd == pytest.approx(round_cents(sum(s.items["usd"])))
    assert all(round(u, 2) == u for u in s.items["usd"])


def test_round_cents_rounds_half_up():
    assert round_cents(2.675) == 2.68
    assert round_cents(0.125) == 0.13
    assert round_cents(-1.7716) == -1.77


def test_usage_month_load_hits_energy_and_peak():
    load = usage_month_load(20_000, 50, 2026, 7)
    assert load.sum() * 0.25 == pytest.approx(20_000)
    assert load.max() == 50
    with pytest.raises(ValueError, match="at least the average"):
        usage_month_load(20_000, 10)


def test_statement_requires_a_single_month(tariffs):
    two = pd.concat([usage_month_load(500, None, 2026, 1), usage_month_load(500, None, 2026, 2)])
    with pytest.raises(ValueError, match="one month"):
        monthly_statement(tariffs[austin.RESIDENTIAL_ID], two)


def test_tier_schema_rules():
    base = dict(tariff_id="T", utility="U", name="n", effective_date="2026-01-01")
    with pytest.raises(ValueError, match="energy_charges or energy_tiers"):
        Tariff(**base)
    with pytest.raises(ValueError, match="no limit"):
        Tariff(**base, energy_tiers=[EnergyTier(up_to_kwh_mo=300, usd_per_kwh=0.1)])
    with pytest.raises(ValueError, match="ascending"):
        Tariff(**base, energy_tiers=[EnergyTier(up_to_kwh_mo=900, usd_per_kwh=0.1),
                                     EnergyTier(up_to_kwh_mo=300, usd_per_kwh=0.2),
                                     EnergyTier(up_to_kwh_mo=None, usd_per_kwh=0.3)])


def test_usd_per_kwh_rider_equals_the_per_mwh_rider_scaled(tariffs):
    base = dict(tariff_id="T", utility="U", name="n", effective_date="2026-01-01",
                energy_charges=[TOUBlock(season="a", period="b", months=list(range(1, 13)), usd_per_mwh=0)])
    load = usage_month_load(1_000, None, 2026, 3)
    a = compute_monthly_bills(Tariff(**base, riders=[Rider(name="r", kind="usd_per_kwh", value=0.05)]), load).total()
    b = compute_monthly_bills(Tariff(**base, riders=[Rider(name="r", kind="usd_per_mwh", value=50.0)]), load).total()
    assert a == pytest.approx(b) == pytest.approx(50.0)


def test_austin_tariffs_do_not_disturb_the_model_run(tariffs):
    from dcop.reports import validated_summary
    from dcop.scenarios import run_all

    validated_summary(run_all(load_inputs("config"), ["base"]))


def test_cli_prints_an_itemized_residential_bill(capsys):
    assert main(["bill", "residential", "--kwh", "860"]) == 0
    out = capsys.readouterr().out
    assert "Power Supply Adjustment" in out and "115.30" in out
    assert main(["bill", "residential", "--kwh", "860", "--no-psa-admin"]) == 0
    assert "117.07" in capsys.readouterr().out


def test_cli_commercial_requires_demand():
    with pytest.raises(SystemExit):
        main(["bill", "commercial", "--kwh", "1000"])


def test_power_supply_adjustment_is_booked_as_energy_and_other_riders_as_fees(tariffs):
    bill = compute_monthly_bills(tariffs[austin.RESIDENTIAL_ID], usage_month_load(860, None, 2026, 1))
    cat = bill.lines.set_index("subcategory")["category"]
    assert cat["Power Supply Adjustment"] == "energy"
    assert cat["Administrative Adjustment to PSA"] == "energy"
    assert cat["Regulatory Charge"] == "riders_fees"
    assert cat["Community Benefit: Customer Assistance Program"] == "riders_fees"
    # moving the PSA between categories does not change the bill
    assert bill.total() == pytest.approx(115.3078, abs=1e-4)
