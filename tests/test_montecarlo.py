"""Monte Carlo, exposure, sensitivities, and the capital recovery timing fix."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dcop.btm.gas import monthly_price_components
from dcop.cli import main
from dcop.finance import npv
from dcop.io import load_distributions, load_inputs
from dcop.ledger import LineFactory, capex_annuity_lines
from dcop.montecarlo import (
    basis_shock,
    path_metrics,
    quantile_sample,
    run_monte_carlo,
    sample_draws,
    sample_efor,
    sample_slip,
    unconfirmed_share,
)
from dcop.reports import exposure_table, tornado_chart
from dcop.scenarios import (
    RunCache,
    _sensitivity_overrides,
    apply_overrides,
    expand_wildcards,
    get_path,
    inputs_tree,
    run_inputs,
    run_scenario,
    run_sensitivities,
)
from dcop.schemas import DiscreteDist, Distributions, SensitivitySpec


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


@pytest.fixture(scope="module")
def dist():
    return load_distributions("config/distributions.yaml")


# ---- samplers ----
def test_quantile_sample_reproduces_the_stated_percentiles():
    u = np.random.default_rng(1).random(400_000)
    x = quantile_sample(u, 20.0, 40.0, 60.0)
    assert np.quantile(x, [0.1, 0.5, 0.9]) == pytest.approx([20.0, 40.0, 60.0], rel=0.01)
    # tails carry the adjacent slope: 20 - 0.25 x 20 = 15 and 60 + 0.25 x 20 = 65
    assert x.min() >= 15.0 - 1e-9 and x.max() <= 65.0 + 1e-9


def test_quantile_sample_is_floored_at_zero_and_handles_a_point_value():
    assert quantile_sample(np.array([0.0]), 1.0, 10.0, 12.0)[0] == 0.0  # 1 - 0.25 x 9 would be negative
    assert quantile_sample(np.linspace(0, 1, 11), 5.0, 5.0, 5.0).tolist() == [5.0] * 11


def test_basis_shock_has_zero_mean_right_skew_and_a_floor():
    z = np.random.default_rng(2).standard_normal(500_000)
    shock = basis_shock(z, 0.5, 0.50)
    assert shock.mean() == pytest.approx(0.0, abs=0.005)
    assert shock.min() > -0.50  # cannot fall below -scale
    assert np.quantile(shock, 0.99) > abs(np.quantile(shock, 0.01))  # right tail longer
    assert not basis_shock(z, 0.0, 0.5).any() and not basis_shock(z, 0.5, 0.0).any()


def test_efor_beta_is_centred_on_the_plant_efor():
    x = sample_efor(np.random.default_rng(3), 0.03, 100.0, 200_000)
    assert x.mean() == pytest.approx(0.03, abs=0.0005)
    assert 0 < x.min() and x.max() < 1
    assert sample_efor(np.random.default_rng(3), 0.0, 100.0, 5).tolist() == [0.0] * 5


def test_slip_frequencies_match_probabilities():
    dist = DiscreteDist(values=[0, 6, 12], probs=[0.5, 0.3, 0.2])
    x = sample_slip(np.random.default_rng(4), dist, 200_000)
    assert [(x == v).mean() for v in (0, 6, 12)] == pytest.approx([0.5, 0.3, 0.2], abs=0.005)


@pytest.mark.parametrize("values, probs", [([0, 6], [0.5, 0.4]), ([0], [0.5, 0.5]), ([-1, 6], [0.5, 0.5]), ([], [])])
def test_discrete_distribution_validation(values, probs):
    with pytest.raises(ValueError):
        DiscreteDist(values=values, probs=probs)


def test_distribution_file_loads_and_rejects_unknown_keys(dist, tmp_path):
    assert dist.n_draws == 2000 and dist.seed == 20261026
    assert len(dist.sensitivities) >= 12
    bad = tmp_path / "d.yaml"
    bad.write_text("n_draws: 10\nbogus: 1\n")
    with pytest.raises(ValueError):
        load_distributions(bad)


# ---- draws ----
def test_draws_are_reproducible_and_seed_dependent(inputs, dist):
    a = sample_draws(inputs, dist, 20, 7).overrides
    assert a == sample_draws(inputs, dist, 20, 7).overrides
    assert a != sample_draws(inputs, dist, 20, 8).overrides


def test_draws_only_touch_the_intended_fields(inputs, dist):
    draws = sample_draws(inputs, dist, 300, 9)
    keys = set().union(*draws.overrides)
    assert keys == {
        "gas.PLACEHOLDER-HUB.basis_adder_usd_mmbtu",
        "plants.RECIP-N2.forced_outage_rate",
        "plants.RECIP-N2.permit_lead_months",
        "upgrades.SITE-A.items.0.cost_usd_sampled", "upgrades.SITE-A.items.0.probability_triggered",
        "upgrades.SITE-A.items.0.lead_time_months",
        "upgrades.SITE-A.items.1.cost_usd_sampled", "upgrades.SITE-A.items.1.probability_triggered",
    }  # item 1 is off the critical path, so its lead time is never slipped
    assert {d["upgrades.SITE-A.items.0.probability_triggered"] for d in draws.overrides} == {1.0}
    assert {d["upgrades.SITE-A.items.1.probability_triggered"] for d in draws.overrides} == {0.0, 1.0}
    assert {d["upgrades.SITE-A.items.0.lead_time_months"] - 30 for d in draws.overrides} <= {0, 6, 12, 18}
    assert {d["plants.RECIP-N2.permit_lead_months"] - 9 for d in draws.overrides} <= {0, 3, 6, 12}
    costs = [d["upgrades.SITE-A.items.0.cost_usd_sampled"] for d in draws.overrides]
    assert 20e6 - 5e6 <= min(costs) and max(costs) <= 60e6 + 5e6  # P10 mirror 20M, P90 60M, tails +/-25%
    for d in draws.overrides[:20]:
        apply_overrides(inputs, d)  # every draw is a valid set of inputs


def test_trigger_frequency_matches_the_item_probability(inputs, dist):
    draws = sample_draws(inputs, dist, 20_000, 10)
    freq = np.mean([d["upgrades.SITE-A.items.1.probability_triggered"] for d in draws.overrides])
    assert freq == pytest.approx(0.6, abs=0.01)


def test_upgrade_exposure_percentiles_match_the_inputs(inputs):
    # one item, certain, 50% allocation, P10/P50/P90 = 20M/40M/60M -> allocated 10M/20M/30M
    item = inputs.upgrades["SITE-A"].items[0].model_copy(
        update={"cost_usd_p10": 20e6, "cost_usd_p50": 40e6, "cost_usd_p90": 60e6, "allocation_pct": 0.5})
    one = inputs.upgrades["SITE-A"].model_copy(update={"items": [item]})
    eff = replace(inputs, upgrades={"SITE-A": one})
    exposure = sample_draws(eff, Distributions(), 60_000, 11).upgrade_exposure_usd["SITE-A"]
    assert np.quantile(exposure, [0.1, 0.5, 0.9]) == pytest.approx([10e6, 20e6, 30e6], rel=0.02)


def test_unconfirmed_share_of_p50_exposure(inputs):
    # item 0: certain, 50% of 40M = 20M. item 1: p = 0.6, 100% of 15M = 15M unconfirmed. 15 / 35.
    assert unconfirmed_share(inputs.upgrades["SITE-A"].items) == pytest.approx(15 / 35)
    assert unconfirmed_share([]) == 0.0


# ---- the engine on draws ----
def degenerate(inputs):
    """Inputs where every sampled quantity is certain, so every draw must equal the base case."""
    items = [i.model_copy(update={"cost_usd_p10": i.cost_usd_p50, "cost_usd_p90": i.cost_usd_p50,
                                  "probability_triggered": 1.0}) for i in inputs.upgrades["SITE-A"].items]
    return replace(inputs, upgrades={"SITE-A": inputs.upgrades["SITE-A"].model_copy(update={"items": items})})


def test_with_no_uncertainty_every_draw_equals_the_deterministic_run(inputs):
    eff = degenerate(inputs)
    result = run_monte_carlo(eff, Distributions(n_draws=6, seed=1), "base")
    base = result.base.set_index(["site_id", "path_id"])["ldcop_usd_mwh"]
    for (site, path), g in result.draws.groupby(["site_id", "path_id"]):
        assert g["ldcop_usd_mwh"].tolist() == pytest.approx([base[(site, path)]] * 6, abs=1e-9)
    # and the deterministic row agrees with the ordinary scenario run
    runs, run_eff = run_scenario(eff, "base")
    for run in runs:
        assert path_metrics(run, run_eff)["ldcop_usd_mwh"] == pytest.approx(base[(run.site_id, run.path_id)])


def test_expected_value_mode_reproduces_the_deterministic_run(inputs):
    result = run_monte_carlo(inputs, Distributions(n_draws=4, seed=1, upgrade_cost=False), "base")
    base = result.base.set_index(["site_id", "path_id"])["ldcop_usd_mwh"]
    for (site, path), g in result.draws.groupby(["site_id", "path_id"]):
        assert g["ldcop_usd_mwh"].tolist() == pytest.approx([base[(site, path)]] * 4, abs=1e-9)


def test_run_cache_does_not_change_results(inputs):
    cache = RunCache()
    for sid in ("base", "delayed_interconnect", "btm_permit_slip"):
        plain, _ = run_scenario(inputs, sid)
        cached, _ = run_scenario(inputs, sid, cache)
        again, _ = run_scenario(inputs, sid, cache)
        for a, b, c in zip(plain, cached, again):
            totals = [sum(ln.usd_real for ln in r.lines) for r in (a, b, c)]
            assert totals == pytest.approx([totals[0]] * 3, abs=1e-6)


@pytest.fixture(scope="module")
def mc(inputs, dist):
    return run_monte_carlo(inputs, dist, "base", n=60, workers=1)


def test_mc_percentiles_are_ordered_and_delays_only_slip_later(mc):
    s = mc.summary.set_index("path_id")
    for path in ("grid", "btm"):
        assert s.loc[path, "ldcop_p10"] <= s.loc[path, "ldcop_p50"] <= s.loc[path, "ldcop_p90"]
        assert s.loc[path, "first_power_p10"] <= s.loc[path, "first_power_p50"] <= s.loc[path, "first_power_p90"]
        assert s.loc[path, "first_power_p10"] >= s.loc[path, "first_power_base"]  # slip is never negative
        assert s.loc[path, "n_failed"] == 0 and s.loc[path, "n_draws"] == 60
    assert s.loc["grid", "ldcop_p90"] > s.loc["grid", "ldcop_base"]  # slips and upgrade costs only add risk upward


def test_p_grid_cheaper_is_the_share_of_paired_draws(mc):
    pivot = mc.draws.pivot(index="draw", columns="path_id", values="ldcop_usd_mwh")
    expected = float((pivot["grid"] < pivot["btm"]).mean())
    assert mc.summary["p_grid_cheaper"].tolist() == pytest.approx([expected, expected])
    assert 0.0 <= expected <= 1.0


def test_mc_is_reproducible_and_independent_of_worker_count(inputs, dist):
    one = run_monte_carlo(inputs, dist, "base", n=24, seed=5, workers=1)
    two = run_monte_carlo(inputs, dist, "base", n=24, seed=5, workers=2)
    pd.testing.assert_frame_equal(one.draws.sort_values(["draw", "path_id"]).reset_index(drop=True),
                                  two.draws.sort_values(["draw", "path_id"]).reset_index(drop=True))
    other = run_monte_carlo(inputs, dist, "base", n=24, seed=6, workers=1)
    assert not one.draws["ldcop_usd_mwh"].equals(other.draws["ldcop_usd_mwh"])


def test_draws_the_engine_rejects_are_counted_not_fatal(inputs):
    dist = Distributions(n_draws=8, seed=1, permit_slip_months=DiscreteDist(values=[240], probs=[1.0]))
    result = run_monte_carlo(inputs, dist, "base")  # 20 years of slip: BTM power lands after the horizon ends
    s = result.summary.set_index("path_id")
    assert s.loc["btm", "n_failed"] == 8
    assert s.loc["grid", "n_failed"] == 0


def test_exposure_summary_reports_percentiles_per_kw_and_flags(mc, inputs):
    e = mc.upgrade_exposure.iloc[0]
    assert e["site_id"] == "SITE-A"
    assert e["upgrade_exposure_p50_usd"] <= e["upgrade_exposure_p90_usd"]
    assert e["upgrade_exposure_p50_usd_per_kw"] == pytest.approx(e["upgrade_exposure_p50_usd"] / 100_000)
    assert e["upgrade_unconfirmed_share"] == pytest.approx(15 / 35)
    assert bool(e["tariff_redesign_risk"]) is False
    flagged = replace(inputs, sites={"SITE-A": inputs.sites["SITE-A"].model_copy(update={"tariff_docket_open": True})})
    res = run_monte_carlo(flagged, Distributions(n_draws=2, seed=1), "base")
    assert bool(res.upgrade_exposure.iloc[0]["tariff_redesign_risk"]) is True


def test_exposure_table_lists_unconfirmed_items_only_by_default(inputs):
    t = exposure_table(inputs)
    assert t["item"].tolist() == ["Line rebuild 12 mi (placeholder)"]
    row = t.iloc[0]
    assert row["probability_triggered"] == 0.6 and row["on_critical_path"] == False  # noqa: E712
    assert row["expected_allocated_usd"] == pytest.approx(0.6 * 1.0 * 15e6)
    assert row["cost_usd_p10"] == 5e6 and row["cost_usd_p90"] == 25e6  # P10 mirrors P90 around P50
    assert len(exposure_table(inputs, include_confirmed=True)) == 2


# ---- gas knobs and the capital timing fix ----
def test_gas_multiplier_and_basis_adder_hand_calc(tmp_path, inputs):
    for name, v in [("hh.csv", 3.5), ("basis.csv", -0.3)]:
        pd.DataFrame({"month": ["2029-01"], "usd_mmbtu": v}).to_csv(tmp_path / name, index=False)
    g = inputs.gas["PLACEHOLDER-HUB"].model_copy(update={
        "henry_hub_curve_ref": "hh.csv", "basis_curve_ref": "basis.csv",
        "henry_hub_multiplier": 1.1, "basis_adder_usd_mmbtu": 0.2})
    p = monthly_price_components(g, tmp_path, np.array([2029 * 12]))
    # (3.5 x 1.1 + (-0.3 + 0.2)) / 0.98 + 0.25 = 3.75 / 0.98 + 0.25
    assert p.sum(axis=1).iloc[0] == pytest.approx(3.75 / 0.98 + 0.25)


def test_capital_recovery_pv_equals_the_outlay_discounted_from_the_exact_date(inputs):
    cfg, r = inputs.global_cfg, inputs.global_cfg.discount_rate
    factory = LineFactory("S", "btm", "b", {})
    lines = capex_annuity_lines(factory, "x", 1_000_000.0, cfg, pd.Timestamp("2029-07-01"))
    by_year = {ln.year: ln.usd_real for ln in lines}
    # 2029-07-01 is 2 + 181/365 years after the end of the base year
    assert npv(by_year, r, cfg.base_year) == pytest.approx(1_000_000 / (1 + r) ** (2 + 181 / 365))
    assert by_year[2029] == pytest.approx(by_year[2030] * 184 / 365)  # first year prorated to Jul 1 - Dec 31
    assert len({round(v, 6) for y, v in by_year.items() if y > 2029}) == 1  # then level


def test_capital_recovery_is_continuous_in_the_in_service_date(inputs):
    cfg = inputs.global_cfg
    factory = LineFactory("S", "btm", "b", {})
    total = lambda d: sum(ln.usd_real for ln in capex_annuity_lines(factory, "x", 1e6, cfg, pd.Timestamp(d)))  # noqa: E731
    assert abs(total("2029-07-02") - total("2029-07-01")) < 1_000  # one day moves the total by under $1k
    assert abs(total("2029-01-01") - total("2028-12-31")) < 5_000  # no jump across the year boundary


def test_earlier_permit_costs_a_little_more_and_later_costs_more(inputs):
    def btm_ldcop(delta):
        eff = apply_overrides(inputs, {"plants.RECIP-N2.permit_lead_months": 9 + delta})
        return next(path_metrics(r, eff)["ldcop_usd_mwh"] for r in run_inputs(eff, "x") if r.path_id == "btm")

    base = btm_ldcop(0)
    assert base < btm_ldcop(-3) < base + 2.0  # plant paid for 3 months before the load arrives
    assert btm_ldcop(1) < btm_ldcop(3) < btm_ldcop(9)  # delay cost builds as more load waits


# ---- override paths and sensitivities ----
def test_selector_and_wildcard_paths(inputs):
    tree = inputs_tree(inputs)
    assert expand_wildcards(tree, "upgrades.SITE-A.items.*.cost_usd_p50") == [
        "upgrades.SITE-A.items.0.cost_usd_p50", "upgrades.SITE-A.items.1.cost_usd_p50"]
    assert get_path(tree, "tariffs.AE-TRANS-HLF-GE20MW.riders.name=Power Supply Adjustment.value") == 0.03956
    eff = apply_overrides(inputs, {"tariffs.AE-TRANS-HLF-GE20MW.riders.name=Power Supply Adjustment.value": 0.05})
    riders = {r.name: r.value for r in eff.tariffs["AE-TRANS-HLF-GE20MW"].riders}
    assert riders["Power Supply Adjustment"] == 0.05 and riders["Regulatory Charge"] == 3.61
    with pytest.raises(KeyError, match="0 list items"):
        apply_overrides(inputs, {"tariffs.AE-TRANS-HLF-GE20MW.riders.name=Nope.value": 1})


def test_sensitivity_modes_and_integer_rounding(inputs):
    tree = inputs_tree(inputs)
    spec = lambda mode, lo, hi, path="global.discount_rate": SensitivitySpec(label="x", path=path, mode=mode, low=lo, high=hi)  # noqa: E731
    low, high = _sensitivity_overrides(inputs, tree, spec("relative", -0.25, 0.5))
    assert (low["global.discount_rate"], high["global.discount_rate"]) == pytest.approx((0.06, 0.12))
    low, high = _sensitivity_overrides(inputs, tree, spec("delta", -0.02, 0.02))
    assert (low["global.discount_rate"], high["global.discount_rate"]) == pytest.approx((0.06, 0.10))
    low, high = _sensitivity_overrides(inputs, tree, spec("absolute", 0.05, 0.11))
    assert (low["global.discount_rate"], high["global.discount_rate"]) == (0.05, 0.11)
    low, high = _sensitivity_overrides(inputs, tree, spec("delta", -3, 6, "plants.{plant}.permit_lead_months"))
    assert (low["plants.RECIP-N2.permit_lead_months"], high["plants.RECIP-N2.permit_lead_months"]) == (6, 15)
    assert all(isinstance(v, int) for v in (*low.values(), *high.values()))


def test_sensitivity_skips_sites_without_the_input(inputs):
    grid_only = inputs.sites["SITE-A"].model_copy(update={"btm_configs": [], "gas_hub": None})
    eff = replace(inputs, sites={"SITE-A": grid_only})
    spec = SensitivitySpec(label="x", path="gas.{hub}.henry_hub_multiplier", mode="absolute", low=0.5, high=2.0)
    assert _sensitivity_overrides(eff, inputs_tree(eff), spec) == ({}, {})


@pytest.fixture(scope="module")
def sens(inputs, dist):
    return run_sensitivities(inputs, "base", dist.sensitivities)


def test_tornado_base_matches_the_deterministic_run_and_swing_is_high_minus_low(sens, mc):
    base = mc.base.set_index("path_id")["ldcop_usd_mwh"]
    for path in ("grid", "btm"):
        assert set(sens.loc[sens.path_id == path, "ldcop_base"].round(9)) == {round(base[path], 9)}
    assert (sens["swing"] - (sens["ldcop_high"] - sens["ldcop_low"]).abs()).abs().max() < 1e-9


def test_tornado_directions_make_economic_sense(sens):
    row = lambda label, path: sens[(sens.label == label) & (sens.path_id == path)].iloc[0]  # noqa: E731
    hh = row("Henry Hub price", "btm")
    assert hh.ldcop_low < hh.ldcop_base < hh.ldcop_high
    assert row("Henry Hub price", "grid").swing < 1e-9  # gas does not touch the grid path
    assert row("Power supply adjustment rate", "btm").swing < 1e-9  # the tariff does not touch BTM
    dr = row("Discount rate", "btm")
    assert dr.ldcop_high > dr.ldcop_low  # capital-heavy path costs more at a higher discount rate
    assert row("Interconnection lead time (months)", "grid").ldcop_high > row("Interconnection lead time (months)", "grid").ldcop_base
    lf = row("Load factor", "btm")
    assert lf.ldcop_high < lf.ldcop_low  # more energy over the same fixed costs
    top_grid = sens[(sens.path_id == "grid")].nlargest(1, "swing").iloc[0]
    assert top_grid.label == "Power supply adjustment rate"


def test_tornado_chart_is_written(sens, tmp_path):
    out = tmp_path / "tornado.png"
    tornado_chart(sens, "SITE-A", out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 10_000
    with pytest.raises(ValueError, match="no sensitivities"):
        tornado_chart(sens, "NOPE", tmp_path / "x.png")


def test_mc_command_prints_and_writes_outputs(tmp_path, capsys):
    assert main(["mc", "--draws", "30", "--workers", "1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Monte Carlo: scenario base, 30 draws" in out and "Unconfirmed upgrade items" in out
    for name in ("mc_draws", "mc_summary", "upgrade_exposure", "exposure_table", "tornado"):
        assert (tmp_path / f"{name}.parquet").exists()
    assert (tmp_path / "tornado_base_SITE-A.png").exists()
