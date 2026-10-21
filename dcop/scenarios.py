"""Deterministic scenario runner (spec §7).

Override keys are dotted paths into the inputs:
``global.<field>``, ``sites.<site_id>.<field>``, ``tariffs.<tariff_id>.<field>``.
List elements are addressed by index (``tariffs.T1.riders.0.value``). A path
that does not exist is an error, and the result is re-validated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from dcop.grid.tariff import tariff_ledger_lines
from dcop.io import Inputs
from dcop.ledger import Ledger
from dcop.load import delivered_and_unserved, site_load_kw
from dcop.schemas import GlobalConfig, Site, Tariff


@dataclass(frozen=True)
class RunResult:
    ledger: Ledger
    inputs_by_scenario: dict[str, Inputs]  # effective inputs after overrides


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
    try:
        if isinstance(cur, list):
            return cur[int(part)]
        return cur[part]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise KeyError(f"override path {dotted!r}: nothing at {part!r}") from exc


def apply_overrides(inputs: Inputs, overrides: dict[str, Any]) -> Inputs:
    if not overrides:
        return inputs
    tree: dict[str, Any] = {
        "global": inputs.global_cfg.model_dump(),
        "sites": {k: v.model_dump() for k, v in inputs.sites.items()},
        "tariffs": {k: v.model_dump() for k, v in inputs.tariffs.items()},
    }
    for dotted, value in overrides.items():
        _set_path(tree, dotted, value)
    return replace(
        inputs,
        global_cfg=GlobalConfig.model_validate(tree["global"]),
        sites={k: Site.model_validate(v) for k, v in tree["sites"].items()},
        tariffs={k: Tariff.model_validate(v) for k, v in tree["tariffs"].items()},
    )


def run_scenario(inputs: Inputs, scenario_id: str) -> tuple[Ledger, Inputs]:
    if scenario_id not in inputs.scenarios:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(inputs.scenarios)}")
    eff = apply_overrides(inputs, inputs.scenarios[scenario_id].overrides)
    cfg = eff.global_cfg
    ledger = Ledger()
    for site in eff.sites.values():
        if not site.grid_paths:
            continue
        if len(site.grid_paths) > 1:
            raise ValueError(
                f"site {site.site_id!r} lists several tariffs {site.grid_paths}; the grid path "
                "takes one. Pick one with a scenario override of sites.<id>.grid_paths."
            )
        tariff = eff.tariffs[site.grid_paths[0]]
        load_kw = site_load_kw(site, cfg.ops_years, data_dir=eff.data_dir)
        # M0: no availability or curtailment model on the grid path yet
        delivered_kw, _unserved_kw = delivered_and_unserved(load_kw)
        ledger.extend(tariff_ledger_lines(site.site_id, scenario_id, tariff, delivered_kw, cfg))
    return ledger, eff


def run_all(inputs: Inputs, scenario_ids: list[str] | None = None) -> RunResult:
    ids = scenario_ids or list(inputs.scenarios)
    ledger = Ledger()
    effective: dict[str, Inputs] = {}
    for sid in ids:
        part, eff = run_scenario(inputs, sid)
        ledger.extend(part)
        effective[sid] = eff
    return RunResult(ledger=ledger, inputs_by_scenario=effective)
