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


def test_base_run_reconciles_and_is_within_bounds(inputs):
    result = run_all(inputs, ["base"])
    summary = validated_summary(result)
    row = summary.iloc[0]
    assert (row.site_id, row.path_id, row.scenario_id) == ("SITE-A", "grid", "base")
    assert 20 <= row.ldcop_usd_mwh <= 250
    reconcile(result.ledger, summary)


def test_ldcop_equals_independent_npv_ratio(inputs):
    result = run_all(inputs, ["base"])
    df = result.ledger.to_frame()
    cost = df.groupby("year")["usd_real"].sum().to_dict()
    mwh = df.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()
    cfg = inputs.global_cfg
    expected = sum(c / (1 + cfg.discount_rate) ** (y - cfg.base_year) for y, c in cost.items()) / sum(
        m / (1 + cfg.discount_rate) ** (y - cfg.base_year) for y, m in mwh.items()
    )
    assert summary_table(result).iloc[0].ldcop_usd_mwh == pytest.approx(expected)
    assert expected == pytest.approx(levelize(cost, mwh, cfg.discount_rate, cfg.base_year))


def test_every_line_carries_all_three_ids(inputs):
    df = run_all(inputs, ["base"]).ledger.to_frame()
    assert df[ID_KEYS].notna().all().all()


def test_tariff_redesign_raises_cost_only_after_year_one(inputs):
    summary = summary_table(run_all(inputs, ["base", "tariff_redesign"])).set_index("scenario_id")
    assert summary.loc["tariff_redesign", "ldcop_usd_mwh"] > summary.loc["base", "ldcop_usd_mwh"]
    assert summary.loc["tariff_redesign", "year1_usd_mwh"] == pytest.approx(summary.loc["base", "year1_usd_mwh"])


def test_year_one_is_the_ramp_year_and_costs_more_per_mwh_than_full_load(inputs):
    summary = summary_table(run_all(inputs, ["base"])).iloc[0]
    assert summary.year1_usd_mwh > summary.year5_usd_mwh  # fixed charges over a smaller ramped load


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
    s = summary_table(run_all(inputs2, ["base", "hi"])).set_index("scenario_id")
    assert s.loc["hi", "ldcop_usd_mwh"] != pytest.approx(s.loc["base", "ldcop_usd_mwh"])


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
