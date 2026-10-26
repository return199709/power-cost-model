"""Deterministic scenario runner (spec §7).

Override keys are dotted paths into the inputs:
``global.<field>``, ``sites.<site_id>.<field>``, ``tariffs.<tariff_id>.<field>``,
``plants.<plant_config_id>.<field>``, ``gas.<hub>.<field>``, ``upgrades.<site_id>.<field>``.
List elements are addressed by index (``tariffs.T1.riders.0.value``). A path
that does not exist is an error, and the result is re-validated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from dcop.btm import availability as avail
from dcop.btm.gas import gas_ledger_lines
from dcop.btm.plant import (
    firm_mw,
    heat_rate_mmbtu_mwh,
    plant_ledger_lines,
    unit_mw_site,
)
from dcop.grid.tariff import annual_average_monthly_load_factor, tariff_ledger_lines
from dcop.grid.upgrades import upgrade_ledger_lines
from dcop.io import Inputs
from dcop.ledger import Ledger, LineFactory
from dcop.load import mwh_by_year, site_load_kw, sum_by_year
from dcop.schemas import (
    GasSupply,
    GlobalConfig,
    LedgerLine,
    PlantConfig,
    SensitivitySpec,
    Site,
    Tariff,
    UpgradeExposure,
)
from dcop.timing import (
    PathTiming,
    btm_lead_months,
    build_timing,
    capacity_mw_series,
    grid_lead_months,
    unserved_lines,
)


@dataclass(frozen=True)
class PathRun:
    """One (site, path, scenario) result: ledger lines plus timing and energy totals."""

    site_id: str
    path_id: str
    scenario_id: str
    lines: list[LedgerLine]
    timing: PathTiming
    load_mwh: float  # what the ramp asks for
    served_mwh: float  # load within the path's timing and capacity
    delivered_mwh: float  # after availability and curtailment
    notes: tuple[str, ...] = ()  # non-fatal findings, e.g. a tariff eligibility test that fails


@dataclass(frozen=True)
class RunResult:
    ledger: Ledger
    inputs_by_scenario: dict[str, Inputs]  # effective inputs after overrides
    paths: list[PathRun]


def _set_path(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = root
    for part in parts[:-1]:
        cur = _child(cur, part, dotted)
    last = parts[-1]
    if isinstance(cur, list):
        i = int(last)
        if not -len(cur) <= i < len(cur):
            raise KeyError(f"override path {dotted!r}: index {i} out of range")
        cur[i] = value
    elif isinstance(cur, dict):
        if last not in cur:
            raise KeyError(f"override path {dotted!r}: no field {last!r}")
        cur[last] = value
    else:
        raise KeyError(f"override path {dotted!r}: cannot set inside {type(cur).__name__}")


def _child(cur: Any, part: str, dotted: str) -> Any:
    if isinstance(cur, list) and "=" in part:  # selector: riders.name=Power Supply Adjustment.value
        key, _, wanted = part.partition("=")
        hits = [el for el in cur if isinstance(el, dict) and str(el.get(key)) == wanted]
        if len(hits) != 1:
            raise KeyError(f"override path {dotted!r}: {len(hits)} list items with {part!r}")
        return hits[0]
    try:
        if isinstance(cur, list):
            return cur[int(part)]
        return cur[part]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise KeyError(f"override path {dotted!r}: nothing at {part!r}") from exc


def get_path(root: dict[str, Any], dotted: str) -> Any:
    """Value at a concrete (no wildcard) override path."""
    cur: Any = root
    for part in dotted.split("."):
        cur = _child(cur, part, dotted)
    return cur


def expand_wildcards(root: dict[str, Any], dotted: str) -> list[str]:
    """Replace each ``*`` with every list index at that position."""
    paths = [""]
    for part in dotted.split("."):
        nxt = []
        for prefix in paths:
            if part == "*":
                node = get_path(root, prefix) if prefix else root
                nxt += [f"{prefix}.{i}" for i in range(len(node))]
            else:
                nxt.append(f"{prefix}.{part}" if prefix else part)
        paths = [p.lstrip(".") for p in nxt]
    return paths


def inputs_tree(inputs: Inputs) -> dict[str, Any]:
    return {
        "global": inputs.global_cfg.model_dump(),
        "sites": {k: v.model_dump() for k, v in inputs.sites.items()},
        "tariffs": {k: v.model_dump() for k, v in inputs.tariffs.items()},
        "plants": {k: v.model_dump() for k, v in inputs.plants.items()},
        "gas": {k: v.model_dump() for k, v in inputs.gas.items()},
        "upgrades": {k: v.model_dump() for k, v in inputs.upgrades.items()},
    }


def apply_overrides(inputs: Inputs, overrides: dict[str, Any]) -> Inputs:
    if not overrides:
        return inputs
    tree = inputs_tree(inputs)
    for dotted, value in overrides.items():
        _set_path(tree, dotted, value)
    return replace(
        inputs,
        global_cfg=GlobalConfig.model_validate(tree["global"]),
        sites={k: Site.model_validate(v) for k, v in tree["sites"].items()},
        tariffs={k: Tariff.model_validate(v) for k, v in tree["tariffs"].items()},
        plants={k: PlantConfig.model_validate(v) for k, v in tree["plants"].items()},
        gas={k: GasSupply.model_validate(v) for k, v in tree["gas"].items()},
        upgrades={k: UpgradeExposure.model_validate(v) for k, v in tree["upgrades"].items()},
    )


class RunCache:
    """Reuses work that does not change between Monte Carlo draws or sensitivity runs.

    Keys are the full content of what the result depends on, so a hit is always valid:
    the load profile depends on the site and years, and the grid bill depends on the
    tariff and on the delivered load, which depends only on the first-power date.
    """

    def __init__(self) -> None:
        self._loads: dict[Any, pd.Series] = {}
        self._bills: dict[Any, tuple[list[LedgerLine], tuple[str, ...]]] = {}

    def load(self, site: Site, cfg: GlobalConfig, data_dir: Any) -> pd.Series:
        key = (site.model_dump_json(), tuple(cfg.ops_years), str(data_dir))
        if key not in self._loads:
            self._loads[key] = site_load_kw(site, cfg.ops_years, data_dir=data_dir)
        return self._loads[key]

    def grid_bill(self, key: Any, build: Any) -> tuple[list[LedgerLine], tuple[str, ...]]:
        if key not in self._bills:
            self._bills[key] = build()
        lines, notes = self._bills[key]
        return list(lines), notes


def _one(items: list[str], site: Site, what: str, override_key: str) -> str:
    if len(items) > 1:
        raise ValueError(
            f"site {site.site_id!r} lists several {what} {items}; a path takes one. "
            f"Pick one with a scenario override of {override_key}."
        )
    return items[0]


def _eligibility_notes(tariff: Tariff, delivered_kw: pd.Series) -> tuple[str, ...]:
    """Check a schedule's load factor requirement on the first full calendar year of service."""
    if tariff.min_load_factor is None:
        return ()
    for year in sorted(set(delivered_kw.index.year)):
        lf = annual_average_monthly_load_factor(delivered_kw, int(year))
        if lf is None:
            continue
        if lf < tariff.min_load_factor:
            return (f"{tariff.tariff_id}: {year} annual average monthly load factor {lf:.1%} is below "
                    f"the {tariff.min_load_factor:.0%} this schedule requires",)
        return ()
    return (f"{tariff.tariff_id}: no full year of service to test the {tariff.min_load_factor:.0%} "
            "load factor requirement",)


def _run_grid_path(
    site: Site, eff: Inputs, scenario_id: str, load_kw: pd.Series, cache: RunCache | None = None
) -> PathRun:
    cfg = eff.global_cfg
    tariff = eff.tariffs[_one(site.grid_paths, site, "tariffs", f"sites.{site.site_id}.grid_paths")]
    exposure = eff.upgrades.get(site.site_id)
    timing = build_timing(
        "grid", site, cfg.clock_start_date, grid_lead_months(exposure), np.inf, cfg.ops_years
    )
    cap_kw = capacity_mw_series(load_kw.index, timing.first_power_date, np.inf) * 1000.0
    served = np.minimum(load_kw.to_numpy(), cap_kw)
    delivered = served  # no grid availability or curtailment model yet
    delivered_kw = pd.Series(delivered, index=load_kw.index)
    mwh_year = mwh_by_year(delivered_kw)
    factory = LineFactory(site.site_id, "grid", scenario_id, mwh_year)

    def bill() -> tuple[list[LedgerLine], tuple[str, ...]]:
        return (
            tariff_ledger_lines(site.site_id, scenario_id, tariff, delivered_kw, cfg, mwh_by_year=mwh_year),
            _eligibility_notes(tariff, delivered_kw),
        )

    if cache is None:
        lines, notes = bill()
    else:
        key = (site.model_dump_json(), tariff.model_dump_json(), scenario_id,
               timing.first_power_date.value, cfg.first_ops_year, tuple(cfg.ops_years), str(eff.data_dir))
        lines, notes = cache.grid_bill(key, bill)
    if exposure is not None:
        lines += upgrade_ledger_lines(factory, exposure, cfg)
    lines += unserved_lines(
        factory, load_kw, served, delivered, timing.first_power_date, cfg.value_of_unserved_usd_mwh
    )
    run = _path_run(site, "grid", scenario_id, lines, timing, load_kw, served, delivered)
    return replace(run, notes=notes)


def _run_btm_path(
    site: Site, eff: Inputs, scenario_id: str, load_kw: pd.Series
) -> PathRun:
    cfg = eff.global_cfg
    plant = eff.plants[_one(site.btm_configs, site, "plants", f"sites.{site.site_id}.btm_configs")]
    gas = eff.gas[site.gas_hub]  # type: ignore[index]  # presence checked at load
    unit_mw = unit_mw_site(plant, site)
    cap_mw = firm_mw(plant, site)
    timing = build_timing(
        "btm", site, cfg.clock_start_date, btm_lead_months(plant, gas), cap_mw, cfg.ops_years
    )
    first_power = timing.first_power_date
    index = load_kw.index

    cap_kw = capacity_mw_series(index, first_power, cap_mw) * 1000.0
    served = np.minimum(load_kw.to_numpy(), cap_kw)
    required = avail.required_units(served / 1000.0, unit_mw)
    table = avail.availability_table(plant.n_units, avail.eaf(plant))
    delivered = served * table[required]
    delivered_kw = pd.Series(delivered, index=index)
    mwh_year = mwh_by_year(delivered_kw)
    factory = LineFactory(site.site_id, "btm", scenario_id, mwh_year)

    # fuel = delivered MWh x heat rate at the part load the running units see
    hr = heat_rate_mmbtu_mwh(plant, eff.data_dir, served / 1000.0, required, unit_mw, index, first_power)
    fuel = delivered / 1000.0 * float((index[1] - index[0]) / pd.Timedelta(hours=1)) * hr
    fuel_year = sum_by_year(fuel, index)

    lines = gas_ledger_lines(factory, gas, cfg, eff.data_dir, index, fuel, first_power)
    lines += plant_ledger_lines(
        factory, plant, cfg, first_power, mwh_year, fuel_year,
        lateral_principal_usd=gas.lateral_capex_usd,
    )
    lines += unserved_lines(
        factory, load_kw, served, delivered, first_power, cfg.value_of_unserved_usd_mwh
    )
    return _path_run(site, "btm", scenario_id, lines, timing, load_kw, served, delivered)


def _path_run(
    site: Site, path_id: str, scenario_id: str, lines: list[LedgerLine], timing: PathTiming,
    load_kw: pd.Series, served: np.ndarray, delivered: np.ndarray,
) -> PathRun:
    step_h = float((load_kw.index[1] - load_kw.index[0]) / pd.Timedelta(hours=1))
    to_mwh = step_h / 1000.0
    return PathRun(
        site_id=site.site_id,
        path_id=path_id,
        scenario_id=scenario_id,
        lines=lines,
        timing=timing,
        load_mwh=float(load_kw.sum() * to_mwh),
        served_mwh=float(served.sum() * to_mwh),
        delivered_mwh=float(delivered.sum() * to_mwh),
    )


def run_inputs(
    eff: Inputs,
    scenario_id: str,
    cache: RunCache | None = None,
    errors: list[tuple[str, str, ValueError]] | None = None,
) -> list[PathRun]:
    """Run every site and path on inputs whose overrides are already applied.

    A path the engine rejects raises, unless ``errors`` is given: then the failure is
    recorded there as (site, path, error) and the other paths still run.
    """
    cfg = eff.global_cfg
    runs: list[PathRun] = []
    for site in eff.sites.values():
        if not (site.grid_paths or site.btm_configs):
            continue
        load_kw = (
            cache.load(site, cfg, eff.data_dir) if cache else site_load_kw(site, cfg.ops_years, data_dir=eff.data_dir)
        )
        for path_id, present, run_path in (
            ("grid", site.grid_paths, lambda: _run_grid_path(site, eff, scenario_id, load_kw, cache)),
            ("btm", site.btm_configs, lambda: _run_btm_path(site, eff, scenario_id, load_kw)),
        ):
            if not present:
                continue
            try:
                runs.append(run_path())
            except ValueError as exc:
                if errors is None:
                    raise
                errors.append((site.site_id, path_id, exc))
    return runs


def run_scenario(
    inputs: Inputs, scenario_id: str, cache: RunCache | None = None
) -> tuple[list[PathRun], Inputs]:
    if scenario_id not in inputs.scenarios:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(inputs.scenarios)}")
    eff = apply_overrides(inputs, inputs.scenarios[scenario_id].overrides)
    return run_inputs(eff, scenario_id, cache), eff


def run_all(inputs: Inputs, scenario_ids: list[str] | None = None) -> RunResult:
    ids = scenario_ids or list(inputs.scenarios)
    ledger = Ledger()
    effective: dict[str, Inputs] = {}
    paths: list[PathRun] = []
    for sid in ids:
        runs, eff = run_scenario(inputs, sid)
        for run in runs:
            ledger.extend(run.lines)
        paths += runs
        effective[sid] = eff
    return RunResult(ledger=ledger, inputs_by_scenario=effective, paths=paths)


# ---- one-at-a-time sensitivities for the tornado (spec §7) ----
class _Missing(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _sensitivity_overrides(
    eff: Inputs, tree: dict[str, Any], spec: SensitivitySpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Low and high override dicts for one sensitivity across every site it applies to."""
    low: dict[str, Any] = {}
    high: dict[str, Any] = {}
    for site in eff.sites.values():
        names = {"site": site.site_id}
        if site.grid_paths:
            names["tariff"] = site.grid_paths[0]
        if site.gas_hub:
            names["hub"] = site.gas_hub
        if site.btm_configs:
            names["plant"] = site.btm_configs[0]
        try:
            pattern = spec.path.format_map(_Missing(names))
            concrete = expand_wildcards(tree, pattern)
        except KeyError:
            continue  # this site has no such input
        for path in concrete:
            base = get_path(tree, path)
            for out, value in ((low, spec.low), (high, spec.high)):
                v = {"absolute": value, "relative": base * (1 + value), "delta": base + value}[spec.mode]
                out[path] = round(v) if isinstance(base, int) and not isinstance(base, bool) else v
    return low, high


def _range_text(spec: SensitivitySpec) -> str:
    if spec.mode == "relative":
        return f"{spec.low:+.0%} / {spec.high:+.0%}"
    if spec.mode == "delta":
        return f"{spec.low:+g} / {spec.high:+g}"
    return f"{spec.low:g} / {spec.high:g}"


def run_sensitivities(
    inputs: Inputs, scenario_id: str, specs: list[SensitivitySpec], cache: RunCache | None = None
) -> pd.DataFrame:
    """LDCoP at each input's low and high value, one input at a time.

    One row per (site, path, input). ``swing`` is |LDCoP(high) - LDCoP(low)|.
    """
    from dcop.montecarlo import path_metrics  # local import: montecarlo imports this module

    cache = cache or RunCache()
    eff0 = apply_overrides(inputs, inputs.scenarios[scenario_id].overrides)
    tree = inputs_tree(eff0)

    def ldcop(eff: Inputs) -> dict[tuple[str, str], float]:
        return {(r["site_id"], r["path_id"]): r["ldcop_usd_mwh"]
                for r in (path_metrics(run, eff) for run in run_inputs(eff, scenario_id, cache))}

    base = ldcop(eff0)
    rows = []
    for spec in specs:
        low_ov, high_ov = _sensitivity_overrides(eff0, tree, spec)
        if not low_ov:
            continue
        low, high = ldcop(apply_overrides(eff0, low_ov)), ldcop(apply_overrides(eff0, high_ov))
        for (site_id, path_id), b in base.items():
            rows.append({
                "site_id": site_id, "path_id": path_id, "scenario_id": scenario_id,
                "label": spec.label, "mode": spec.mode, "range": _range_text(spec),
                "ldcop_base": b, "ldcop_low": low[(site_id, path_id)], "ldcop_high": high[(site_id, path_id)],
                "swing": abs(high[(site_id, path_id)] - low[(site_id, path_id)]),
            })
    return pd.DataFrame(rows)
