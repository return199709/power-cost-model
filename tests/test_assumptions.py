"""Assumptions dump, source flags, staleness, and cost-line tracing (milestone M4)."""

import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

import dcop.assumptions as A
from dcop.assumptions import (
    COST_LINE_INPUTS,
    build_audit,
    expand_for_sites,
    flatten,
    git_commit,
    is_placeholder,
    render_markdown,
    source_flags,
)
from dcop.cli import main
from dcop.io import load_distributions, load_inputs
from dcop.reports import levelized_categories, summary_table
from dcop.scenarios import get_path, inputs_tree, run_all
from dcop.schemas import AssumptionEntry

TODAY = date(2026, 10, 26)


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


@pytest.fixture(scope="module")
def dist():
    return load_distributions("config/distributions.yaml")


@pytest.fixture(scope="module")
def all_scenarios(inputs):
    return list(inputs.scenarios)


@pytest.fixture(scope="module")
def result(inputs):
    return run_all(inputs)


@pytest.fixture(scope="module")
def audit(inputs, all_scenarios, dist, result):
    return build_audit(inputs, all_scenarios, today=TODAY, config_dir=Path("config"), dist=dist, result=result)


# ---- flags ----
@pytest.mark.parametrize("source, expected", [
    ("PLACEHOLDER: made up", True), ("placeholder, not real site data", True), ("synthetic test tariff", True),
    ("Austin Energy FY2026 Electric Tariff", False), ("", False), (None, False)])
def test_placeholder_detection(source, expected):
    assert is_placeholder(source) is expected


def test_staleness_boundary_is_strictly_more_than_stale_days():
    assert source_flags("s", TODAY - timedelta(days=365), TODAY, 365) == ()
    assert source_flags("s", TODAY - timedelta(days=366), TODAY, 365) == ("stale",)
    assert source_flags("s", TODAY - timedelta(days=30), TODAY, 10) == ("stale",)


def test_missing_source_or_date_is_untagged_and_flags_combine():
    assert source_flags(None, TODAY, TODAY, 365) == ("no source",)
    assert source_flags("s", None, TODAY, 365) == ("no as_of",)
    assert source_flags(None, None, TODAY, 365) == ("no source", "no as_of")
    assert source_flags("placeholder", TODAY - timedelta(days=999), TODAY, 365) == ("placeholder", "stale")


# ---- flatten ----
def test_flatten_paths_values_and_selectors(inputs):
    rows = dict(flatten(inputs.tariffs["AE-TRANS-HLF-GE20MW"].model_dump()))
    assert rows["riders.name=Power Supply Adjustment.value"] == "0.03956"
    assert rows["riders.name=Power Supply Adjustment.category"] == "energy"
    assert rows["riders.name=Community Benefit: Customer Assistance Program.annual_cap_usd"] == "200000.0"
    assert rows["demand_charges.name=Demand Charge.usd_per_kw_mo"] == "12.1"
    assert rows["energy_charges.0.months"] == "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
    assert "source" not in rows and "as_of" not in rows  # object-level, shown once in the header
    assert not any(k.endswith(".name") for k in rows if k.startswith("riders."))  # the name is in the path
    tiers = dict(flatten(inputs.tariffs["AE-RES-INSIDE"].model_dump()))
    assert tiers["energy_tiers.1.usd_per_kwh"] == "0.05138" and tiers["energy_tiers.3.up_to_kwh_mo"] == "null"


def test_flattened_paths_work_as_scenario_overrides(inputs):
    from dcop.scenarios import apply_overrides

    rows = dict(flatten(inputs.tariffs["AE-TRANS-HLF-GE20MW"].model_dump()))
    path = "riders.name=Power Supply Adjustment.value"
    eff = apply_overrides(inputs, {f"tariffs.AE-TRANS-HLF-GE20MW.{path}": 0.05})
    assert {r.name: r.value for r in eff.tariffs["AE-TRANS-HLF-GE20MW"].riders}["Power Supply Adjustment"] == 0.05
    assert rows[path] == "0.03956"


# ---- the repo's own config ----
def test_every_input_in_the_repo_config_is_tagged_and_resolves(audit):
    c = audit.counts()
    assert (c["no source"], c["no as_of"], c["unresolved"], c["missing file"]) == (0, 0, 0, 0)
    assert c["items"] > 30


def test_placeholders_are_flagged_and_real_tariffs_are_not(audit):
    flagged = {w: f for w, f in audit.flagged()}
    assert "placeholder" in flagged["site SITE-A"] and "placeholder" in flagged["plant RECIP-N2"]
    assert "tariff AE-TRANS-HLF-GE20MW" not in flagged  # the Austin tariff is sourced, current, and real
    assert "global settings global" not in flagged
    assert any(w.startswith("A2 ") and "placeholder" in f for w, f in flagged.items())


def test_austin_tariff_goes_stale_after_a_year(inputs, all_scenarios):
    fresh = build_audit(inputs, ["base"], today=date(2026, 10, 26))
    assert "tariff AE-TRANS-HLF-GE20MW" not in {w for w, _ in fresh.flagged()}  # 359 days old
    old = build_audit(inputs, ["base"], today=date(2027, 12, 1))
    assert "stale" in dict(old.flagged())["tariff AE-TRANS-HLF-GE20MW"]
    assert any("stale" in f for _, f in old.strict_problems())


def test_strict_problems_ignore_soft_flags(inputs):
    soft = [AssumptionEntry(id="X1", name="n", path="global.discount_rate", source="Finance memo",
                            as_of=TODAY, owner="TBD", confidence="low")]
    from dataclasses import replace

    only_soft = replace(inputs, assumptions=soft, sites={}, tariffs=inputs.tariffs)
    audit = build_audit(only_soft, ["base"], today=TODAY)
    row = next(r for r in audit.register)
    assert set(row.flags) == {"owner TBD", "low confidence"}
    assert not [p for p in audit.strict_problems() if p[0].startswith("X1")]


# ---- register ----
def test_register_expands_per_site_and_resolves_values(audit):
    rows = {(r.id, r.site): r for r in audit.register}
    assert rows[("A3", "SITE-A")].value == "0.9" and rows[("A3", "SITE-A")].path == "sites.SITE-A.load_factor"
    assert rows[("A1", None)].value == "0.08"
    assert rows[("A9", "SITE-A")].value == "['AE-TRANS-HLF-GE20MW']".replace("'", "")
    assert rows[("A6", "SITE-A")].value == "null"  # no carbon price in the base case
    structural = rows[("S1", None)]
    assert structural.path is None and "nothing is sold" in structural.value


def test_register_reports_unresolved_paths_and_missing_owner_flag(inputs):
    from dataclasses import replace

    entries = [AssumptionEntry(id="B1", name="bad", path="global.no_such_field", source="s", as_of=TODAY),
               AssumptionEntry(id="B2", name="owned", path="global.discount_rate", source="s", as_of=TODAY,
                               owner="Jane Doe", confidence="high")]
    audit = build_audit(replace(inputs, assumptions=entries), ["base"], today=TODAY)
    rows = {r.id: r for r in audit.register}
    assert "unresolved" in rows["B1"].flags and rows["B1"].value == "(path not found)"
    assert rows["B2"].flags == ()
    assert any("unresolved" in f for _, f in audit.strict_problems())


def test_expand_for_sites_skips_sites_without_the_input(inputs):
    from dataclasses import replace

    grid_only = inputs.sites["SITE-A"].model_copy(update={"btm_configs": [], "gas_hub": None})
    eff = replace(inputs, sites={"SITE-A": grid_only})
    assert expand_for_sites(eff, "plants.{plant}.capex_usd_kw") == []
    assert expand_for_sites(eff, "tariffs.{tariff}.customer_charge_usd_mo") == [
        ("SITE-A", "tariffs.AE-TRANS-HLF-GE20MW.customer_charge_usd_mo")]
    assert expand_for_sites(eff, "global.discount_rate") == [(None, "global.discount_rate")]


# ---- data files ----
def test_data_files_are_hashed_and_sourced(audit):
    refs = {d.path: d for d in audit.data_files}
    assert {"processed/henry_hub_placeholder.csv", "processed/basis_placeholder.csv",
            "processed/part_load_placeholder.csv", "processed/henry_hub_high_placeholder.csv",
            "processed/henry_hub_low_placeholder.csv"} <= set(refs)  # includes the scenario curves
    d = refs["processed/basis_placeholder.csv"]
    assert d.sha256 == A.sha256_file(Path("data/processed/basis_placeholder.csv"))
    assert "placeholder" in d.flags and d.as_of == date(2026, 10, 21)


def test_data_file_without_manifest_entry_or_on_disk_is_flagged(inputs, tmp_path):
    from dataclasses import replace

    audit = build_audit(replace(inputs, data_sources={}), ["base"], today=TODAY)
    assert all({"no source", "no as_of"} <= set(d.flags) for d in audit.data_files)
    empty = build_audit(replace(inputs, data_dir=tmp_path), ["base"], today=TODAY)
    assert all("missing file" in d.flags and d.sha256 is None for d in empty.data_files)


def test_config_hashes_change_when_a_config_file_changes(tmp_path):
    shutil.copytree("config", tmp_path / "config")
    shutil.copytree("data", tmp_path / "data")
    inputs = load_inputs(tmp_path / "config")
    before = dict(build_audit(inputs, ["base"], today=TODAY, config_dir=tmp_path / "config").config_hashes)
    (tmp_path / "config" / "gas.yaml").write_text((tmp_path / "config" / "gas.yaml").read_text() + "\n# edited\n")
    after = dict(build_audit(inputs, ["base"], today=TODAY, config_dir=tmp_path / "config").config_hashes)
    assert before["gas.yaml"] != after["gas.yaml"]
    assert {k for k in before if before[k] != after[k]} == {"gas.yaml"}


# ---- overrides ----
def test_override_table_shows_base_and_new_values(audit):
    rows = {(o.scenario, o.path): o for o in audit.overrides}
    r = rows[("delayed_interconnect", "upgrades.SITE-A.items.0.lead_time_months")]
    assert (r.base, r.override) == ("30", "42")
    r = rows[("high_gas", "gas.PLACEHOLDER-HUB.henry_hub_curve_ref")]
    assert r.base == "processed/henry_hub_placeholder.csv" and r.override == "processed/henry_hub_high_placeholder.csv"
    assert not [o for o in audit.overrides if o.scenario == "base"]


# ---- the trace: any number back to its inputs ----
def test_every_cost_line_the_model_can_produce_has_traceable_inputs(result, inputs):
    tree = inputs_tree(inputs)
    produced = {(r.path_id, ln.category) for r in result.paths for ln in r.lines}
    assert produced, "expected ledger lines"
    for path_id, category in produced:
        assert category in COST_LINE_INPUTS[path_id], f"no trace for {path_id}/{category}"
    for path_id, categories in COST_LINE_INPUTS.items():
        for category, templates in categories.items():
            for template in templates:
                for _, path in expand_for_sites(inputs, template):
                    get_path(tree, path)  # raises KeyError if the trace points at a field that does not exist


def test_levelized_categories_sum_to_ldcop(result):
    summary = summary_table(result).set_index(["site_id", "path_id", "scenario_id"])["ldcop_usd_mwh"]
    cats = levelized_categories(result).groupby(["site_id", "path_id", "scenario_id"])["usd_per_mwh"].sum()
    assert cats.to_numpy() == pytest.approx(summary.reindex(cats.index).to_numpy())


def test_trace_rows_carry_each_categorys_levelized_cost_and_inputs(audit):
    fuel = next(t for t in audit.trace if t.path_id == "btm" and t.category == "fuel")
    assert fuel.scenario == "base" and fuel.usd_per_mwh == pytest.approx(31.95, abs=0.01)
    assert "gas.PLACEHOLDER-HUB.henry_hub_curve_ref" in fuel.inputs
    assert "plants.RECIP-N2.heat_rate_mmbtu_mwh_hhv_iso" in fuel.inputs
    demand = next(t for t in audit.trace if t.path_id == "grid" and t.category == "demand")
    assert "tariffs.AE-TRANS-HLF-GE20MW.demand_charges" in demand.inputs
    assert not any("gas." in p for p in demand.inputs)  # grid lines do not depend on gas inputs
    total = {}
    for t in audit.trace:
        total[(t.site_id, t.path_id)] = total.get((t.site_id, t.path_id), 0.0) + t.usd_per_mwh
    assert total[("SITE-A", "btm")] == pytest.approx(103.11, abs=0.01)


# ---- markdown ----
def test_markdown_contains_the_audit_trail(audit):
    md = render_markdown(audit)
    for expected in ("# Assumptions and sources", "Run date 2026-10-26", "## Audit summary",
                     "## Assumptions register", "## Inputs used", "### Tariff: AE-TRANS-HLF-GE20MW",
                     "Austin Energy FY2026 Electric Tariff (Nov 1, 2025), pp. 16-18",
                     "`riders.name=Power Supply Adjustment.value` | 0.03956", "## Data files",
                     "## Scenario overrides", "## Where each cost line comes from (scenario base)",
                     "Loaded but not used by this run: ", "tariff EXAMPLE-LGS-TOU", "`gas.yaml`",
                     "monte carlo settings", "A2 Value of unserved energy"):
        assert expected in md, expected
    assert "commit " in md and "SHA-256" in md


def test_markdown_escapes_pipes_in_table_cells(inputs):
    from dataclasses import replace

    entry = AssumptionEntry(id="P1", name="pipe", path="global.discount_rate", source="a | b", as_of=TODAY)
    md = render_markdown(build_audit(replace(inputs, assumptions=[entry]), ["base"], today=TODAY))
    assert "a \\| b" in md


def test_git_commit_falls_back_to_unknown(monkeypatch):
    assert git_commit(Path(".")) != ""
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(A.subprocess, "run", boom)
    assert git_commit() == "unknown"


# ---- CLI ----
def test_audit_command_writes_the_file_and_strict_fails_on_placeholders(tmp_path, capsys):
    assert main(["audit", "--scenario", "base", "--out", str(tmp_path), "--today", "2026-10-26"]) == 0
    text = (tmp_path / "assumptions.md").read_text()
    assert text.startswith("# Assumptions and sources")
    assert "placeholder" in capsys.readouterr().out
    assert main(["audit", "--scenario", "base", "--out", str(tmp_path), "--today", "2026-10-26", "--strict"]) == 2
    assert "strict: site SITE-A: placeholder" in capsys.readouterr().out


def test_run_command_writes_assumptions_with_the_cost_trace(tmp_path):
    assert main(["run", "--scenario", "base", "--out", str(tmp_path)]) == 0
    md = (tmp_path / "assumptions.md").read_text()
    assert "## Where each cost line comes from (scenario base)" in md


def test_audit_command_flags_stale_inputs_in_strict_mode(tmp_path, capsys):
    code = main(["audit", "--scenario", "base", "--out", str(tmp_path), "--today", "2027-12-01", "--strict"])
    assert code == 2
    assert "tariff AE-TRANS-HLF-GE20MW: stale" in capsys.readouterr().out
