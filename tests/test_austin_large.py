"""Austin Energy FY2026 Large General and Transmission schedules.

Expected values are worked by hand from the tariff PDF (pp. 10-18, 21, 23-26).
"""

from dataclasses import replace

import pandas as pd
import pytest

from dcop.cli import main
from dcop.grid import austin
from dcop.grid.tariff import annual_average_monthly_load_factor, compute_monthly_bills, monthly_load_factors
from dcop.io import load_inputs
from dcop.reports import summary_table
from dcop.scenarios import run_all


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


@pytest.fixture(scope="module")
def tariffs(inputs):
    return inputs.tariffs


def flat(start, end, kw, freq="h"):
    return pd.Series(float(kw), index=pd.date_range(start, end, freq=freq, inclusive="left"))


# ---- transcription check against the PDF ----
# id: (customer $/mo, demand $/kW, energy $/kWh, PSA $/kWh, admin $/kWh, regulatory $/kW, HLF)
RATES = {
    austin.LGS_PRI_LT3000_ID: (327.73, 12.56, 0.00119, 0.04005, -0.00200, 3.65, False),
    austin.LGS_PRI_3_20_ID: (2731.05, 15.30, 0.00020, 0.04005, -0.00200, 3.65, False),
    austin.LGS_PRI_HLF_3_20_ID: (6881.15, 14.66, 0.0, 0.04005, -0.00200, 3.65, True),
    austin.LGS_PRI_GE20_ID: (3004.16, 16.24, 0.00166, 0.04005, -0.00200, 3.65, False),
    austin.LGS_PRI_HLF_GE20_ID: (21848.40, 15.74, 0.0, 0.04005, -0.00200, 3.65, True),
    austin.TRANS_ID: (3823.47, 14.21, 0.00375, 0.03956, -0.00199, 3.61, False),
    austin.TRANS_HLF_ID: (21848.40, 12.10, 0.0, 0.03956, -0.00199, 3.61, True),
}


@pytest.mark.parametrize("tariff_id", list(RATES))
def test_large_load_rates_match_the_tariff_pdf(tariffs, tariff_id):
    customer, demand, energy, psa, admin, reg, hlf = RATES[tariff_id]
    t = tariffs[tariff_id]
    riders = {r.name: r for r in t.riders}
    assert t.customer_charge_usd_mo == customer
    assert t.demand_charges[0].usd_per_kw_mo == demand
    assert t.energy_charges[0].usd_per_mwh == pytest.approx(energy * 1000)
    assert riders["Power Supply Adjustment"].value == psa
    assert riders["Administrative Adjustment to PSA"].value == admin
    assert riders["Regulatory Charge"].value == reg
    assert riders["Regulatory Charge"].kind == "usd_per_kw_mo"
    assert riders["Community Benefit: Customer Assistance Program"].value == 0.00237
    assert t.power_factor_min == 0.90
    # high-load-factor schedules: contract rules, one Community Benefit line, capped
    assert (t.min_load_factor == 0.85) is hlf
    assert (t.min_bill_trailing_months == 12) is hlf
    assert (riders["Community Benefit: Customer Assistance Program"].annual_cap_usd == 200_000) is hlf
    assert ("Community Benefit: Energy Efficiency Services" in riders) is not hlf
    assert ("Community Benefit: Service Area Lighting" in riders) is not hlf


def test_standard_schedules_carry_the_voltage_specific_community_benefit_rates(tariffs):
    cbc = lambda tid: {r.name.split(": ")[1]: r.value for r in tariffs[tid].riders if r.name.startswith("Community")}  # noqa: E731
    assert cbc(austin.LGS_PRI_GE20_ID) == {"Customer Assistance Program": 0.00237, "Service Area Lighting": 0.00246, "Energy Efficiency Services": 0.00444}
    assert cbc(austin.TRANS_ID) == {"Customer Assistance Program": 0.00237, "Service Area Lighting": 0.00243, "Energy Efficiency Services": 0.00438}


# ---- monthly bills, hand calcs (January, 744 hours) ----
def test_hlf_transmission_bill_hand_calc(tariffs):
    # 65,000,000 kWh at a 100,000 kW peak: customer 21,848.40; demand 100,000 x 12.10 = 1,210,000; energy 0;
    # PSA 65M x 0.03956 = 2,571,400; admin 65M x -0.00199 = -129,350; CAP 65M x 0.00237 = 154,050 (under the cap);
    # regulatory 100,000 x 3.61 = 361,000. Total 4,188,948.40.
    s = austin.large_load_statement(tariffs, 65_000_000, 100_000, "transmission", high_load_factor=True)
    assert s.tariff_id == austin.TRANS_HLF_ID
    assert s.billed_kw == pytest.approx(100_000)
    assert s.usd("Demand Charge") == 1_210_000.00
    assert s.usd("Power Supply Adjustment") == 2_571_400.00
    assert s.usd("Community Benefit: Customer Assistance Program") == 154_050.00
    assert s.total_usd == pytest.approx(4_188_948.40)


def test_standard_transmission_bill_hand_calc(tariffs):
    # customer 3,823.47; demand 1,421,000; energy 65M x 0.00375 = 243,750; PSA 2,571,400; admin -129,350;
    # CBC 65M x 0.00918 = 596,700; regulatory 361,000. Total 5,068,323.47.
    s = austin.large_load_statement(tariffs, 65_000_000, 100_000, "transmission")
    assert s.tariff_id == austin.TRANS_ID
    assert s.total_usd == pytest.approx(5_068_323.47)


def test_hlf_is_cheaper_than_standard_transmission_at_high_load_factor(tariffs):
    hlf = austin.large_load_statement(tariffs, 65_000_000, 100_000, "transmission", high_load_factor=True)
    std = austin.large_load_statement(tariffs, 65_000_000, 100_000, "transmission")
    assert hlf.total_usd < std.total_usd


def test_primary_3_to_20_mw_bill_hand_calc(tariffs):
    # 5,000,000 kWh at 10,000 kW: customer 2,731.05; demand 153,000; energy 1,000; PSA 200,250; admin -10,000;
    # CBC 5M x 0.00927 = 46,350; regulatory 36,500. Total 429,831.05.
    s = austin.large_load_statement(tariffs, 5_000_000, 10_000, "primary")
    assert s.tariff_id == austin.LGS_PRI_3_20_ID
    assert s.total_usd == pytest.approx(429_831.05)


def test_primary_below_3_mw_bill_hand_calc(tariffs):
    # 1,000,000 kWh at 2,000 kW: customer 327.73; demand 25,120; energy 1,190; PSA 40,050; admin -2,000;
    # CBC 9,270; regulatory 7,300. Total 81,257.73.
    s = austin.large_load_statement(tariffs, 1_000_000, 2_000, "primary")
    assert s.tariff_id == austin.LGS_PRI_LT3000_ID
    assert s.total_usd == pytest.approx(81_257.73)


def test_primary_hlf_20_mw_bill_hand_calc(tariffs):
    # 40,000,000 kWh at 60,000 kW: customer 21,848.40; demand 944,400; PSA 1,602,000; admin -80,000;
    # CAP 94,800; regulatory 219,000. Total 2,802,048.40.
    s = austin.large_load_statement(tariffs, 40_000_000, 60_000, "primary", high_load_factor=True)
    assert s.tariff_id == austin.LGS_PRI_HLF_GE20_ID
    assert s.total_usd == pytest.approx(2_802_048.40)


def test_power_factor_penalty_applies_to_large_loads(tariffs):
    # Tariff p.15: 31,000 kW at 86.7% bills 32,068 kW (31,000 x 0.90 / 0.87)
    s = austin.large_load_statement(tariffs, 20_000_000, 31_000, "transmission", power_factor=0.867)
    assert s.billed_kw == pytest.approx(31_000 * 0.90 / 0.87)
    assert round(s.billed_kw) == 32_069  # the tariff prints 32,068: it truncates, the model does not round
    assert s.billed_kw == pytest.approx(32_068, abs=1.0)


# ---- Customer Assistance Program annual cap ----
def cap_by_month(bill):
    cap = bill.lines[bill.lines.subcategory == "Community Benefit: Customer Assistance Program"]
    return cap.set_index("month")["usd"]


def test_cap_stops_charges_once_200k_is_reached_in_the_year(tariffs):
    # 90 MW flat all of 2027. January: 66,960,000 kWh x 0.00237 = 158,695.20. February would add 143,337.60,
    # so only 200,000 - 158,695.20 = 41,304.80 is charged. Nothing after that.
    bill = compute_monthly_bills(tariffs[austin.TRANS_HLF_ID], flat("2027-01-01", "2028-01-01", 90_000))
    cap = cap_by_month(bill)
    assert cap[1] == pytest.approx(158_695.20)
    assert cap[2] == pytest.approx(41_304.80)
    assert cap[3:].sum() == 0
    assert cap.sum() == pytest.approx(200_000.00)


def test_cap_is_prorated_for_a_partial_year_of_service(tariffs):
    # Service from July: six billed months, so the cap is 6/12 x 200,000 = 100,000
    load = pd.concat([flat("2027-01-01", "2027-07-01", 0), flat("2027-07-01", "2028-01-01", 90_000)])
    cap = cap_by_month(compute_monthly_bills(tariffs[austin.TRANS_HLF_ID], load))
    assert cap[7] == pytest.approx(100_000.00)
    assert cap.sum() == pytest.approx(100_000.00)


def test_cap_resets_each_calendar_year(tariffs):
    bill = compute_monthly_bills(tariffs[austin.TRANS_HLF_ID], flat("2027-01-01", "2029-01-01", 90_000))
    by_year = bill.lines[bill.lines.subcategory == "Community Benefit: Customer Assistance Program"].groupby("year")["usd"].sum()
    assert by_year.tolist() == pytest.approx([200_000.0, 200_000.0])


def test_standard_schedules_have_no_cap(tariffs):
    bill = compute_monthly_bills(tariffs[austin.TRANS_ID], flat("2027-01-01", "2028-01-01", 90_000))
    cap = bill.lines[bill.lines.subcategory == "Community Benefit: Customer Assistance Program"]["usd"].sum()
    assert cap > 1_000_000  # about 1.9M uncapped


# ---- trailing-demand minimum bill ----
def uncapped(tariff):
    """Same tariff without the Community Benefit cap, so a short test series is not prorated."""
    return tariff.model_copy(update={"riders": [r.model_copy(update={"annual_cap_usd": None}) for r in tariff.riders]})



def test_minimum_bill_follows_the_highest_kw_of_the_last_12_months(tariffs):
    # January 100 MW, February 40 MW (26,880,000 kWh). February's normal bill is 1,723,835.60:
    #   customer 21,848.40 + demand 484,000 + PSA 1,063,372.80 - admin 53,491.20 + CAP 63,705.60 + regulatory 144,400.
    # Floor = 100,000 x 12.10 + PSA 1,063,372.80 - admin 53,491.20 = 2,219,881.60, so the true-up is 496,046.00.
    load = pd.concat([flat("2027-01-01", "2027-02-01", 100_000), flat("2027-02-01", "2027-03-01", 40_000)])
    bill = compute_monthly_bills(uncapped(tariffs[austin.TRANS_HLF_ID]), load)
    feb = bill.lines[bill.lines.month == 2]
    assert feb[feb.category == "minimum_bill"]["usd"].sum() == pytest.approx(496_046.00)
    assert feb["usd"].sum() == pytest.approx(2_219_881.60)
    assert bill.lines[(bill.lines.month == 1) & (bill.lines.category == "minimum_bill")].empty


def test_minimum_bill_forgets_a_peak_older_than_the_window(tariffs):
    short = uncapped(tariffs[austin.TRANS_HLF_ID]).model_copy(update={"min_bill_trailing_months": 2})
    load = pd.concat([flat("2027-01-01", "2027-02-01", 100_000), flat("2027-02-01", "2027-04-01", 40_000)])
    bill = compute_monthly_bills(short, load)
    assert not bill.lines[(bill.lines.month == 2) & (bill.lines.category == "minimum_bill")].empty
    assert bill.lines[(bill.lines.month == 3) & (bill.lines.category == "minimum_bill")].empty  # Jan fell out


def test_steady_load_never_triggers_the_minimum_bill(tariffs):
    bill = compute_monthly_bills(tariffs[austin.TRANS_HLF_ID], flat("2027-01-01", "2028-01-01", 90_000))
    assert "minimum_bill" not in set(bill.lines.category)


# ---- load factor ----
def test_monthly_load_factor_follows_the_tariff_glossary():
    # 743 h at 80,000 kW and one hour at 100,000 kW: (59,440,000 + 100,000) / (100,000 x 744) = 0.800269
    load = flat("2027-01-01", "2027-02-01", 80_000)
    load.iloc[10] = 100_000
    lf = monthly_load_factors(load)
    assert lf.iloc[0] == pytest.approx(59_540_000 / 74_400_000)
    assert monthly_load_factors(flat("2027-01-01", "2027-02-01", 5_000)).iloc[0] == pytest.approx(1.0)


def test_annual_average_monthly_load_factor_needs_twelve_served_months():
    full = flat("2027-01-01", "2028-01-01", 5_000)
    assert annual_average_monthly_load_factor(full, 2027) == pytest.approx(1.0)
    partial = pd.concat([flat("2027-01-01", "2027-03-01", 0), flat("2027-03-01", "2028-01-01", 5_000)])
    assert annual_average_monthly_load_factor(partial, 2027) is None


def test_base_site_meets_the_hlf_load_factor_requirement(inputs):
    summary = summary_table(run_all(inputs, ["base"]))
    assert summary.loc[summary.path_id == "grid", "notes"].iloc[0] == ""


def test_low_load_factor_site_gets_an_eligibility_note(inputs):
    scen = inputs.scenarios["base"].model_copy(
        update={"scenario_id": "lowlf", "overrides": {"sites.SITE-A.load_factor": 0.6}}
    )
    summary = summary_table(run_all(replace(inputs, scenarios={"lowlf": scen}), ["lowlf"]))
    note = summary.loc[summary.path_id == "grid", "notes"].iloc[0]
    assert "below the 85%" in note and "2030" in note  # the first full year of service


# ---- the site now takes the Austin Energy HLF transmission schedule ----
def test_site_uses_the_austin_hlf_transmission_tariff(inputs):
    site = inputs.sites["SITE-A"]
    assert site.utility == "Austin Energy"
    assert site.grid_paths == [austin.TRANS_HLF_ID]
    df = run_all(inputs, ["base"]).ledger.to_frame().query("path_id == 'grid'")
    assert set(df.note.dropna()) >= {f"tariff={austin.TRANS_HLF_ID}"}


def test_site_ledger_respects_the_annual_cap(inputs):
    df = run_all(inputs, ["base"]).ledger.to_frame()
    cap = df[(df.path_id == "grid") & (df.subcategory == "Community Benefit: Customer Assistance Program")]
    by_year = cap.groupby("year")["usd_real"].sum()
    assert by_year[2029] == pytest.approx(100_000.0)  # half a year of service, prorated cap
    assert by_year[2030:].between(199_999.99, 200_000.01).all()


# ---- classification ----
@pytest.mark.parametrize(
    "peaks, voltage, hlf, expected",
    [
        ([2_999] * 4, "primary", False, austin.LGS_PRI_LT3000_ID),
        ([3_000] * 4, "primary", False, austin.LGS_PRI_3_20_ID),
        ([19_999] * 4, "primary", False, austin.LGS_PRI_3_20_ID),
        ([20_000] * 4, "primary", False, austin.LGS_PRI_GE20_ID),
        ([10_000] * 4, "primary", True, austin.LGS_PRI_HLF_3_20_ID),
        ([50_000] * 4, "primary", True, austin.LGS_PRI_HLF_GE20_ID),
        ([500] * 4, "transmission", False, austin.TRANS_ID),
        ([100_000] * 4, "transmission", False, austin.TRANS_ID),
        ([100_000] * 4, "transmission", True, austin.TRANS_HLF_ID),
    ],
)
def test_large_load_class(peaks, voltage, hlf, expected):
    assert austin.large_load_tariff_id(peaks, voltage, hlf) == expected


def test_hlf_class_needs_the_minimum_size_and_a_known_voltage():
    with pytest.raises(ValueError, match="20,000 kW"):
        austin.large_load_tariff_id([10_000], "transmission", True)
    with pytest.raises(ValueError, match="3,000 kW"):
        austin.large_load_tariff_id([1_000], "primary", True)
    with pytest.raises(ValueError, match="voltage"):
        austin.large_load_tariff_id([10_000], "secondary")


def test_cli_large_bill(capsys):
    assert main(["bill", "large", "--kwh", "65000000", "--kw", "100000", "--voltage", "transmission", "--hlf"]) == 0
    assert "4,188,948.40" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["bill", "large", "--kwh", "1000", "--kw", "100"])


def test_grid_energy_group_is_not_empty_when_the_schedule_has_no_energy_charge(inputs):
    """The HLF transmission schedule has a zero energy rate; the PSA carries the energy cost."""
    from dcop.reports import levelized_components

    comp = levelized_components(run_all(inputs, ["base"]))
    grid = comp[comp.path_id == "grid"].set_index("group")["usd_per_mwh"]
    # PSA 0.03956 less the 0.00199 adjustment is 3.757 cents/kWh, close to $37.6/MWh before escalation and delay
    assert grid["Energy / fuel"] == pytest.approx(37.57, abs=0.1)
    assert grid["Riders & fees"] < grid["Energy / fuel"]
