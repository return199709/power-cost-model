"""Hybrid path: gas serves the load until the grid is energized, then the grid takes over."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dcop.btm import availability as avail
from dcop.btm.plant import firm_mw, unit_mw_site
from dcop.finance import npv
from dcop.io import load_inputs
from dcop.ledger import LineFactory, capex_annuity_lines
from dcop.load import site_load_kw
from dcop.montecarlo import path_metrics, run_monte_carlo
from dcop.reports import stack_chart, summary_table, timing_chart, tornado_chart
from dcop.scenarios import apply_overrides, run_all, run_inputs, run_scenario, run_sensitivities
from dcop.schemas import Distributions, HybridConfig
from dcop.timing import PathTiming, build_timing, combine_timings

HYB = "sites.SITE-A.hybrid."


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


def runs_for(inputs, overrides=None):
    eff = apply_overrides(inputs, overrides or {})
    return {r.path_id: r for r in run_inputs(eff, "x")}, eff


def lines_of(run, category, sub=None):
    return [ln for ln in run.lines if ln.category == category and (sub is None or ln.subcategory == sub)]


def ldcop(inputs, overrides=None):
    runs, eff = runs_for(inputs, overrides)
    return path_metrics(runs["hybrid"], eff)["ldcop_usd_mwh"]


# ---- stitching ----
def test_hybrid_serves_gas_before_the_handover_and_the_grid_after(inputs):
    runs, eff = runs_for(inputs)
    h, g = runs["hybrid"], runs["grid"]
    assert h.timing.first_power_date == pd.Timestamp("2029-01-01")  # gas power
    assert g.timing.first_power_date == pd.Timestamp("2029-07-01")  # the handover
    mwh = lambda run: {ln.year: ln.mwh_delivered for ln in run.lines}  # noqa: E731
    # after the handover the grid carries everything, so the years match the grid path exactly
    for year in range(2030, 2044):
        assert mwh(h)[year] == pytest.approx(mwh(g)[year])
    # 2029 is the grid's July-December plus the gas plant's January-June, worked out independently
    site, cfg = eff.sites["SITE-A"], eff.global_cfg
    load = site_load_kw(site, cfg.ops_years)
    plant = eff.plants["RECIP-N2"]
    required = avail.required_units(np.minimum(load.to_numpy(), firm_mw(plant, site) * 1000) / 1000, unit_mw_site(plant, site))
    delivered = np.minimum(load.to_numpy(), firm_mw(plant, site) * 1000) * avail.availability_table(
        plant.n_units, avail.eaf(plant))[required]
    h1 = (load.index >= "2029-01-01") & (load.index < "2029-07-01")
    assert mwh(h)[2029] == pytest.approx(mwh(g)[2029] + delivered[h1].sum() / 1000.0, rel=1e-9)


def test_every_hybrid_line_carries_the_combined_mwh(inputs):
    runs, _ = runs_for(inputs)
    by_year: dict[int, set] = {}
    for ln in runs["hybrid"].lines:
        assert ln.path_id == "hybrid"
        by_year.setdefault(ln.year, set()).add(round(ln.mwh_delivered, 6))
    assert all(len(v) == 1 for v in by_year.values())
    result = run_all(inputs, ["base", "delayed_interconnect"])
    result.ledger.validate_mwh()


def mdqs(line):
    """(bridge MDQ, backup MDQ) in MMBtu/day from a transport line's note."""
    parts = dict(p.split("=") for p in line.note.split("; "))
    return float(parts["mdq_mmbtu_d"]), float(parts["backup_mdq_mmbtu_d"])


def test_fuel_and_firm_transport_stop_at_the_handover(inputs):
    runs, _ = runs_for(inputs)
    h = runs["hybrid"]
    assert {ln.year for ln in lines_of(h, "fuel")} == {2029}
    ft = lines_of(h, "transport_fixed")
    assert [ln.year for ln in ft] == [2029]  # backup_transport_pct is 0, so none after
    bridge_mdq, _ = mdqs(ft[0])
    # gas transport at 0.30 per MMBtu/day for the 181 days from Jan 1 to Jun 30 of 2029
    assert ft[0].usd_real == pytest.approx(0.30 * bridge_mdq * 181, rel=1e-3)


def test_bridge_transport_is_sized_to_the_bridge_and_backup_to_the_full_plant(inputs):
    runs, _ = runs_for(inputs, {HYB + "backup_transport_pct": 0.5})
    ft = {ln.year: ln for ln in lines_of(runs["hybrid"], "transport_fixed")}
    bridge_mdq, backup_mdq = mdqs(ft[2029])
    standalone = lines_of(runs["btm"], "transport_fixed")[0]
    assert backup_mdq == pytest.approx(float(standalone.note.split("=")[1]), rel=1e-4)  # full-load quantity
    assert bridge_mdq < backup_mdq / 3  # the bridge carries 25 of 100 MW
    assert set(ft) == set(range(2029, 2044))
    assert ft[2030].usd_real == pytest.approx(0.30 * 365 * 0.5 * backup_mdq, rel=1e-3)
    # 2029: the bridge quantity for 181 days, then half the backup quantity for 184 days
    assert ft[2029].usd_real == pytest.approx(0.30 * (bridge_mdq * 181 + 0.5 * backup_mdq * 184), rel=1e-3)


def test_fixed_om_continues_as_backup_or_stops(inputs):
    kept, _ = runs_for(inputs)
    assert {ln.year: ln.usd_real for ln in lines_of(kept["hybrid"], "fixed_om")}[2035] == pytest.approx(25 * 144_000)
    off, _ = runs_for(inputs, {HYB + "backup_fixed_om_pct": 0.0})
    om = {ln.year: ln.usd_real for ln in lines_of(off["hybrid"], "fixed_om")}
    assert om == {2029: pytest.approx(25 * 144_000 * 181 / 365)}  # only until the handover


def test_plant_capital_is_still_paid_when_the_plant_is_kept(inputs):
    runs, _ = runs_for(inputs)
    years = {ln.year for ln in lines_of(runs["hybrid"], "plant_capex")}
    assert years == set(range(2029, 2044))
    btm_capex = sum(ln.usd_real for ln in lines_of(runs["btm"], "plant_capex"))
    assert sum(ln.usd_real for ln in lines_of(runs["hybrid"], "plant_capex")) == pytest.approx(btm_capex)


# ---- selling the plant ----
def test_sale_ends_capital_recovery_and_credits_the_proceeds(inputs):
    sold = {HYB + "backup_fixed_om_pct": 0.0, HYB + "sale_at_handover_pct": 0.5}
    runs, _ = runs_for(inputs, sold)
    h = runs["hybrid"]
    assert {ln.year for ln in lines_of(h, "plant_capex")} == {2029}
    proceeds = lines_of(h, "residual", "sale at handover")
    # half of 259.2M plant + 21.6M emissions controls + 15M lateral
    assert [(ln.year, ln.usd_real) for ln in proceeds] == [(2029, pytest.approx(-0.5 * 295_800_000))]


def test_capital_recovery_to_an_end_date_has_the_same_present_value(inputs):
    cfg, r = inputs.global_cfg, inputs.global_cfg.discount_rate
    factory = LineFactory("S", "btm", "b", {})
    start, end = pd.Timestamp("2029-01-01"), pd.Timestamp("2029-07-01")
    horizon = capex_annuity_lines(factory, "x", 1e6, cfg, start)
    short = capex_annuity_lines(factory, "x", 1e6, cfg, start, end=end)
    pv = lambda lines: npv({ln.year: ln.usd_real for ln in lines}, r, cfg.base_year)  # noqa: E731
    assert pv(short) == pytest.approx(pv(horizon)) == pytest.approx(1e6 / (1 + r) ** 2)
    assert [ln.year for ln in short] == [2029]
    with pytest.raises(ValueError, match="not after"):
        capex_annuity_lines(factory, "x", 1e6, cfg, start, end=start)


def test_schedule_alone_does_not_move_the_levelized_cost_but_a_sale_does(inputs):
    kept = ldcop(inputs, {HYB + "backup_fixed_om_pct": 0.0})
    quarter = ldcop(inputs, {HYB + "backup_fixed_om_pct": 0.0, HYB + "sale_at_handover_pct": 0.25})
    half = ldcop(inputs, {HYB + "backup_fixed_om_pct": 0.0, HYB + "sale_at_handover_pct": 0.5})
    assert quarter < kept and half < quarter
    assert (kept - quarter) == pytest.approx(quarter - half, rel=0.01)  # proceeds enter linearly
    assert ldcop(inputs) > kept  # backup O&M adds cost


def test_a_sold_plant_cannot_also_be_backup():
    with pytest.raises(ValueError, match="sold at the handover"):
        HybridConfig(sale_at_handover_pct=0.5, backup_fixed_om_pct=1.0)
    with pytest.raises(ValueError, match="sold at the handover"):
        HybridConfig(sale_at_handover_pct=0.5, backup_fixed_om_pct=0.0, backup_transport_pct=0.1)
    HybridConfig(sale_at_handover_pct=0.5, backup_fixed_om_pct=0.0)


def test_hybrid_needs_both_paths(inputs):
    site = inputs.sites["SITE-A"]
    with pytest.raises(ValueError, match="needs grid_paths, btm_configs and gas_hub"):
        site.model_copy(update={"btm_configs": []}).model_validate(
            {**site.model_dump(), "btm_configs": []})


# ---- unserved energy and timing ----
def test_hybrid_avoids_the_delay_the_grid_suffers(inputs):
    runs, eff = runs_for(inputs)
    assert path_metrics(runs["grid"], eff)["delay_cost_usd"] > 14e6
    assert path_metrics(runs["hybrid"], eff)["delay_cost_usd"] == 0.0  # gas power is on time
    runs, eff = runs_for(inputs, {"upgrades.SITE-A.items.0.lead_time_months": 42})
    assert path_metrics(runs["grid"], eff)["delay_cost_usd"] > 100e6
    assert path_metrics(runs["hybrid"], eff)["delay_cost_usd"] == 0.0


def test_capacity_shortfall_is_priced_only_while_the_plant_serves(inputs):
    over = {"upgrades.SITE-A.items.0.lead_time_months": 42, "plants.RECIP-N2.n_units": 6}  # 68 MW plant, grid in 2030-07
    runs, _ = runs_for(inputs, over)
    short = lines_of(runs["hybrid"], "unserved_energy", "capacity_shortfall")
    assert {ln.year for ln in short} == {2030}  # 100 MW load vs a 68 MW plant, January to June only
    assert not [ln for ln in runs["hybrid"].lines if ln.category == "delay"]


def test_combined_timing_switches_at_the_handover(inputs):
    site, cfg = inputs.sites["SITE-A"], inputs.global_cfg
    gas_t = build_timing("btm", site, cfg.clock_start_date, 24, 60.0, cfg.ops_years)  # 60 MW plant
    grid_t = build_timing("grid", site, cfg.clock_start_date, 30, np.inf, cfg.ops_years)
    both = combine_timings("hybrid", gas_t, grid_t, grid_t.first_power_date, site)
    q = lambda d: pd.Timestamp(d)  # noqa: E731
    assert both.mw_online[q("2029-01-01")] == 25.0 and both.mw_online[q("2029-04-01")] == 25.0  # gas
    assert both.mw_online[q("2029-07-01")] == 50.0 and both.mw_online[q("2030-01-01")] == 100.0  # grid
    assert both.first_power_date == gas_t.first_power_date
    assert both.full_power_date == q("2030-01-01")
    assert gas_t.full_power_date is pd.NaT or pd.isna(gas_t.full_power_date)  # the plant alone never gets there


def test_summary_reports_the_hybrid_dates(inputs):
    s = summary_table(run_all(inputs, ["base"])).set_index("path_id")
    assert str(s.loc["hybrid", "first_power_date"]) == "2029-01-01"
    assert str(s.loc["hybrid", "full_power_date"]) == "2030-01-01"
    assert s.loc["hybrid", "availability"] == pytest.approx(1.0, abs=0.01)


# ---- degenerate case ----
def test_if_the_grid_arrives_first_the_hybrid_is_just_the_grid(inputs):
    runs, eff = runs_for(inputs, {"plants.RECIP-N2.permit_lead_months": 18})  # gas power 2029-10, grid 2029-07
    h, g = runs["hybrid"], runs["grid"]
    assert path_metrics(h, eff)["ldcop_usd_mwh"] == pytest.approx(path_metrics(g, eff)["ldcop_usd_mwh"])
    assert not lines_of(h, "plant_capex") and not lines_of(h, "fuel")
    assert all(ln.path_id == "hybrid" for ln in h.lines) and h.timing.path_id == "hybrid"
    assert any("no bridge plant is built" in n for n in h.notes)


# ---- the rest of the model sees a third path ----
def test_monte_carlo_covers_all_three_paths(inputs):
    result = run_monte_carlo(inputs, Distributions(n_draws=24, seed=3,
                             interconnection_slip_months={"values": [0, 12], "probs": [0.5, 0.5]},
                             permit_slip_months={"values": [0, 9], "probs": [0.5, 0.5]}), "base")
    assert set(result.summary.path_id) == {"grid", "btm", "hybrid"}
    assert len(result.draws) == 24 * 3
    assert result.summary["p_lowest_cost"].sum() == pytest.approx(1.0)  # every draw has one cheapest path
    assert result.summary["n_failed"].sum() == 0


def test_degenerate_monte_carlo_equals_the_base_for_the_hybrid(inputs):
    items = [i.model_copy(update={"cost_usd_p10": i.cost_usd_p50, "cost_usd_p90": i.cost_usd_p50,
                                  "probability_triggered": 1.0}) for i in inputs.upgrades["SITE-A"].items]
    eff = replace(inputs, upgrades={"SITE-A": inputs.upgrades["SITE-A"].model_copy(update={"items": items})})
    result = run_monte_carlo(eff, Distributions(n_draws=3, seed=1), "base")
    base = result.base.set_index("path_id")["ldcop_usd_mwh"]
    h = result.draws[result.draws.path_id == "hybrid"]["ldcop_usd_mwh"]
    assert h.tolist() == pytest.approx([base["hybrid"]] * 3, abs=1e-9)


def test_sensitivity_on_backup_settings_moves_only_the_hybrid(inputs):
    from dcop.io import load_distributions

    sens = run_sensitivities(inputs, "base", load_distributions("config/distributions.yaml").sensitivities)
    row = lambda label, path: sens[(sens.label == label) & (sens.path_id == path)].iloc[0]  # noqa: E731
    label = "Backup gas transport kept after handover"
    assert row(label, "hybrid").ldcop_high > row(label, "hybrid").ldcop_low
    assert row(label, "grid").swing < 1e-9 and row(label, "btm").swing < 1e-9


def test_charts_draw_three_paths(inputs, tmp_path):
    result = run_all(inputs, ["base"])
    stack_chart(result, "base", tmp_path / "stack.png")
    timing_chart(result, "base", tmp_path / "timing.png")
    sens = run_sensitivities(inputs, "base", __import__("dcop.io", fromlist=["x"]).load_distributions(
        "config/distributions.yaml").sensitivities)
    tornado_chart(sens, "SITE-A", tmp_path / "tornado.png")
    for name in ("stack", "timing", "tornado"):
        assert (tmp_path / f"{name}.png").read_bytes()[:4] == b"\x89PNG"


def test_audit_traces_hybrid_lines(inputs):
    from datetime import date

    from dcop.assumptions import COST_LINE_INPUTS, build_audit

    assert "sites.{site}.hybrid" in COST_LINE_INPUTS["hybrid"]["fuel"]
    assert "tariffs.{tariff}.demand_charges" in COST_LINE_INPUTS["hybrid"]["demand"]
    result = run_all(inputs, ["base"])
    audit = build_audit(inputs, ["base"], today=date(2026, 10, 30), result=result)
    rows = [t for t in audit.trace if t.path_id == "hybrid"]
    assert {t.category for t in rows} >= {"fuel", "demand", "plant_capex", "upgrade_capex"}
    a15 = next(r for r in audit.register if r.id == "A15")
    assert "backup_transport_pct" in a15.value and "placeholder" in a15.flags
