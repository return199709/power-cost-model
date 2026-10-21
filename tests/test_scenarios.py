import pytest

from dcop.finance import levelize
from dcop.io import deflate_tariff, load_inputs, resolve_ref
from dcop.ledger import ID_KEYS, reconcile
from dcop.reports import OutOfRangeError, check_bounds, summary_table, validated_summary
from dcop.scenarios import apply_overrides, run_all, run_scenario


@pytest.fixture(scope="module")
def inputs():
    return load_inputs("config")


def test_repo_config_loads_and_cross_references(inputs):
    assert set(inputs.scenarios) >= {"base", "tariff_redesign"}
    assert inputs.sites["SITE-A"].grid_paths[0] in inputs.tariffs


def pick(summary, path, scenario="base"):
    return summary[(summary.path_id == path) & (summary.scenario_id == scenario)].iloc[0]


def test_base_run_reconciles_and_is_within_bounds(inputs):
    result = run_all(inputs, ["base"])
    summary = validated_summary(result)  # reconciles ledger, components, and bounds
    assert set(summary.path_id) == {"grid", "btm"}
    assert summary.ldcop_usd_mwh.between(20, 250).all()
    reconcile(result.ledger, summary)


def test_ldcop_equals_independent_npv_ratio_for_both_paths(inputs):
    result = run_all(inputs, ["base"])
    df = result.ledger.to_frame()
    cfg = inputs.global_cfg
    summary = summary_table(result)
    for path_id, g in df.groupby("path_id"):
        cost = g.groupby("year")["usd_real"].sum().to_dict()
        mwh = g.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()
        d = lambda y: (1 + cfg.discount_rate) ** (y - cfg.base_year)  # noqa: E731
        expected = sum(c / d(y) for y, c in cost.items()) / sum(m / d(y) for y, m in mwh.items())
        assert pick(summary, path_id).ldcop_usd_mwh == pytest.approx(expected)
        assert expected == pytest.approx(levelize(cost, mwh, cfg.discount_rate, cfg.base_year))


def test_every_line_carries_all_three_ids(inputs):
    df = run_all(inputs, ["base"]).ledger.to_frame()
    assert df[ID_KEYS].notna().all().all()


def test_tariff_redesign_affects_grid_only_and_only_after_year_one(inputs):
    summary = summary_table(run_all(inputs, ["base", "tariff_redesign"]))
    base, redesign = pick(summary, "grid"), pick(summary, "grid", "tariff_redesign")
    assert redesign.ldcop_usd_mwh > base.ldcop_usd_mwh
    assert redesign.year1_usd_mwh == pytest.approx(base.year1_usd_mwh)
    assert pick(summary, "btm", "tariff_redesign").ldcop_usd_mwh == pytest.approx(
        pick(summary, "btm").ldcop_usd_mwh
    )


def test_year_one_costs_more_per_mwh_than_year_five(inputs):
    summary = summary_table(run_all(inputs, ["base"]))
    for path in ("grid", "btm"):  # fixed and capital charges over a smaller, ramping load
        row = pick(summary, path)
        assert row.year1_usd_mwh > row.year5_usd_mwh


def test_override_changes_inputs_and_unknown_path_fails(inputs):
    eff = apply_overrides(inputs, {"global.discount_rate": 0.1, "tariffs.EXAMPLE-LGS-TOU.riders.0.value": 5.0})
    assert eff.global_cfg.discount_rate == 0.1
    assert eff.tariffs["EXAMPLE-LGS-TOU"].riders[0].value == 5.0
    assert inputs.global_cfg.discount_rate == 0.08  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(inputs, {"global.no_such_field": 1})
    with pytest.raises(KeyError):
        apply_overrides(inputs, {"sites.NOPE.load_mw_peak": 1})
    with pytest.raises(ValueError):  # re-validated: negative discount rate rejected
        apply_overrides(inputs, {"global.discount_rate": -0.5})


def test_higher_discount_rate_changes_ldcop(inputs):
    from dataclasses import replace

    inputs2 = replace(inputs, scenarios={**inputs.scenarios, "hi": inputs.scenarios["base"].model_copy(
        update={"scenario_id": "hi", "overrides": {"global.discount_rate": 0.12}})})
    s = summary_table(run_all(inputs2, ["base", "hi"]))
    for path in ("grid", "btm"):
        assert pick(s, path, "hi").ldcop_usd_mwh != pytest.approx(pick(s, path).ldcop_usd_mwh)


def test_out_of_range_ldcop_fails_unless_allowed(inputs):
    result = run_all(inputs, ["base"])
    summary = summary_table(result)
    cfg = result.inputs_by_scenario["base"].global_cfg
    tight = cfg.model_copy(update={"ldcop_bounds_usd_mwh": (1.0, 2.0)})
    object.__setattr__(result.inputs_by_scenario["base"], "global_cfg", tight)
    with pytest.raises(OutOfRangeError):
        check_bounds(summary, result)
    object.__setattr__(result.inputs_by_scenario["base"], "global_cfg", tight.model_copy(update={"allow_out_of_range": True}))
    check_bounds(summary, result)


def test_unknown_scenario_is_an_error(inputs):
    with pytest.raises(KeyError):
        run_scenario(inputs, "nope")


def test_deflate_tariff_converts_dollars_but_not_percent_riders():
    raw = {
        "dollar_year": 2028, "customer_charge_usd_mo": 1100.0,
        "energy_charges": [{"usd_per_mwh": 110.0}],
        "demand_charges": [{"usd_per_kw_mo": 11.0}],
        "riders": [{"kind": "usd_per_mwh", "value": 2.2}, {"kind": "pct_of_bill", "value": 0.015}],
    }
    out = deflate_tariff(raw, base_year=2026, inflation=0.10)  # two years at 10% = /1.21
    assert "dollar_year" not in out
    assert out["customer_charge_usd_mo"] == pytest.approx(1100 / 1.21)
    assert out["energy_charges"][0]["usd_per_mwh"] == pytest.approx(110 / 1.21)
    assert out["demand_charges"][0]["usd_per_kw_mo"] == pytest.approx(11 / 1.21)
    assert out["riders"][0]["value"] == pytest.approx(2.2 / 1.21)
    assert out["riders"][1]["value"] == 0.015
    assert raw["dollar_year"] == 2028  # input not mutated


def test_resolve_ref_rejects_absolute_and_traversal(tmp_path):
    (tmp_path / "ok.csv").write_text("x")
    assert resolve_ref(tmp_path, "ok.csv") == (tmp_path / "ok.csv").resolve()
    for bad in ["/etc/passwd", "../x.csv", "a/../../x.csv"]:
        with pytest.raises(ValueError):
            resolve_ref(tmp_path, bad)


def test_yaml_is_loaded_safely(tmp_path):
    from dcop.io import read_yaml

    p = tmp_path / "evil.yaml"
    p.write_text("x: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception):
        read_yaml(p)


# ---- M1: grid vs BTM for one site, components reconcile, timing flips results ----
def test_levelized_components_sum_to_ldcop_for_every_path(inputs):
    from dcop.reports import check_components, levelized_components

    result = run_all(inputs, ["base", "high_gas", "delayed_interconnect"])
    summary = summary_table(result)
    components = levelized_components(result)
    check_components(summary, components)
    assert len(components) == len(summary) * 8  # eight chart groups each


def test_mwh_is_consistent_across_every_ledger_line(inputs):
    run_all(inputs).ledger.validate_mwh()  # all seven scenarios


def test_gas_price_moves_btm_only(inputs):
    s = summary_table(run_all(inputs, ["base", "high_gas", "low_gas"]))
    assert pick(s, "btm", "high_gas").ldcop_usd_mwh > pick(s, "btm").ldcop_usd_mwh > pick(s, "btm", "low_gas").ldcop_usd_mwh
    for scenario in ("high_gas", "low_gas"):
        assert pick(s, "grid", scenario).ldcop_usd_mwh == pytest.approx(pick(s, "grid").ldcop_usd_mwh)
    # +$2/MMBtu x 8.8 MMBtu/MWh / 0.98 retention is about $18/MWh before part-load and degradation
    swing = pick(s, "btm", "high_gas").ldcop_usd_mwh - pick(s, "btm").ldcop_usd_mwh
    assert 17 < swing < 22


def test_delayed_interconnect_slips_grid_first_power_a_year(inputs):
    s = summary_table(run_all(inputs, ["base", "delayed_interconnect"]))
    base, slip = pick(s, "grid"), pick(s, "grid", "delayed_interconnect")
    assert str(base.first_power_date) == "2029-07-01"
    assert str(slip.first_power_date) == "2030-07-01"
    assert slip.delay_cost_usd > base.delay_cost_usd
    assert slip.ldcop_usd_mwh > base.ldcop_usd_mwh
    assert pick(s, "btm", "delayed_interconnect").ldcop_usd_mwh == pytest.approx(pick(s, "btm").ldcop_usd_mwh)


def test_btm_permit_slip_creates_delay_cost_on_btm_only(inputs):
    s = summary_table(run_all(inputs, ["base", "btm_permit_slip"]))
    assert pick(s, "btm").delay_cost_usd == 0.0
    slip = pick(s, "btm", "btm_permit_slip")
    assert str(slip.first_power_date) == "2029-10-01"  # 2027-01-01 + 9 + 18... = 33 months
    assert slip.delay_cost_usd > 0
    assert slip.ldcop_usd_mwh > pick(s, "btm").ldcop_usd_mwh
    assert pick(s, "grid", "btm_permit_slip").ldcop_usd_mwh == pytest.approx(pick(s, "grid").ldcop_usd_mwh)


def test_upgrade_p90_raises_grid_cost_only(inputs):
    s = summary_table(run_all(inputs, ["base", "upgrade_p90"]))
    assert pick(s, "grid", "upgrade_p90").ldcop_usd_mwh > pick(s, "grid").ldcop_usd_mwh
    assert pick(s, "btm", "upgrade_p90").ldcop_usd_mwh == pytest.approx(pick(s, "btm").ldcop_usd_mwh)


def test_undersized_plant_leaves_load_unserved_and_never_hits_full_power(inputs):
    from dataclasses import replace

    scen = inputs.scenarios["base"].model_copy(
        update={"scenario_id": "small", "overrides": {"plants.RECIP-N2.n_units": 6}}
    )
    result = run_all(replace(inputs, scenarios={"small": scen}), ["small"])
    btm = pick(summary_table(result), "btm", "small")
    assert btm.full_power_date is None  # four firm units carry about 68 MW of a 100 MW ramp
    assert btm.unserved_mwh_total > 0
    cats = result.ledger.to_frame().query("path_id == 'btm'")
    assert "capacity_shortfall" in set(cats.subcategory.dropna())


def test_grid_availability_is_one_and_btm_is_below_one(inputs):
    s = summary_table(run_all(inputs, ["base"]))
    assert pick(s, "grid").availability == pytest.approx(1.0)
    assert 0.9 < pick(s, "btm").availability < 1.0


def test_value_of_unserved_must_be_set_when_energy_goes_unserved(inputs):
    from dataclasses import replace

    scen = inputs.scenarios["base"].model_copy(
        update={"scenario_id": "novou", "overrides": {"global.value_of_unserved_usd_mwh": None}}
    )
    with pytest.raises(ValueError, match="value_of_unserved"):
        run_all(replace(inputs, scenarios={"novou": scen}), ["novou"])


def test_site_with_two_tariffs_is_rejected(inputs):
    from dataclasses import replace

    scen = inputs.scenarios["base"].model_copy(
        update={"scenario_id": "two", "overrides": {"sites.SITE-A.grid_paths": ["EXAMPLE-LGS-TOU", "EXAMPLE-LGS-TOU"]}}
    )
    with pytest.raises(ValueError, match="takes one"):
        run_all(replace(inputs, scenarios={"two": scen}), ["two"])
