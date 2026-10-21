"""Loaders, unit normalization, and real/nominal conversion (spec §2, §3).

Input files are read with ``yaml.safe_load`` only. ``*_ref`` fields resolve
inside the data directory (see SECURITY.md).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dcop.finance import to_real
from dcop.schemas import GlobalConfig, Scenario, Site, Tariff


def kw_to_mw(kw: float) -> float:
    return kw / 1000.0


def mw_to_kw(mw: float) -> float:
    return mw * 1000.0


def read_yaml(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_ref(data_dir: Path | str, ref: str) -> Path:
    """Resolve a ``*_ref`` file reference, refusing anything outside ``data_dir``."""
    base = Path(data_dir).resolve()
    if Path(ref).is_absolute():
        raise ValueError(f"reference {ref!r} must be relative to the data directory")
    target = (base / ref).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"reference {ref!r} escapes the data directory")
    return target


@dataclass(frozen=True)
class Inputs:
    global_cfg: GlobalConfig
    sites: dict[str, Site]
    tariffs: dict[str, Tariff]
    scenarios: dict[str, Scenario]
    data_dir: Path


def deflate_tariff(raw: dict[str, Any], base_year: int, inflation: float) -> dict[str, Any]:
    """Convert a tariff dict's nominal ``dollar_year`` dollars to real ``base_year`` dollars.

    ``dollar_year`` is removed from the result. Percent riders are left untouched.
    """
    raw = copy.deepcopy(raw)
    dollar_year = raw.pop("dollar_year", None)
    if dollar_year is None or dollar_year == base_year:
        return raw

    def real(v: float) -> float:
        return to_real(v, dollar_year, base_year, inflation)

    raw["customer_charge_usd_mo"] = real(raw.get("customer_charge_usd_mo", 0.0))
    for block in raw.get("energy_charges", []):
        block["usd_per_mwh"] = real(block["usd_per_mwh"])
    for charge in raw.get("demand_charges", []):
        charge["usd_per_kw_mo"] = real(charge["usd_per_kw_mo"])
    for rider in raw.get("riders", []):
        if rider["kind"] != "pct_of_bill":
            rider["value"] = real(rider["value"])
    return raw


def _by_id(items: list[Any], key: str, what: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if getattr(item, key) in out:
            raise ValueError(f"duplicate {what} id {getattr(item, key)!r}")
        out[getattr(item, key)] = item
    return out


def load_global(path: Path | str) -> GlobalConfig:
    return GlobalConfig.model_validate(read_yaml(path) or {})


def load_sites(path: Path | str) -> dict[str, Site]:
    return _by_id([Site.model_validate(r) for r in read_yaml(path) or []], "site_id", "site")


def load_tariffs(path: Path | str, cfg: GlobalConfig) -> dict[str, Tariff]:
    raws = [deflate_tariff(r, cfg.base_year, cfg.inflation_pct) for r in read_yaml(path) or []]
    return _by_id([Tariff.model_validate(r) for r in raws], "tariff_id", "tariff")


def load_scenarios(path: Path | str) -> dict[str, Scenario]:
    return _by_id(
        [Scenario.model_validate(r) for r in read_yaml(path) or []], "scenario_id", "scenario"
    )


def load_inputs(config_dir: Path | str = "config", data_dir: Path | str | None = None) -> Inputs:
    config_dir = Path(config_dir)
    cfg = load_global(config_dir / "global.yaml")
    sites = load_sites(config_dir / "sites.yaml")
    tariffs = load_tariffs(config_dir / "tariffs.yaml", cfg)
    scenarios = load_scenarios(config_dir / "scenarios.yaml")
    for site in sites.values():
        for tariff_id in site.grid_paths:
            if tariff_id not in tariffs:
                raise ValueError(f"site {site.site_id!r} references unknown tariff {tariff_id!r}")
    return Inputs(
        global_cfg=cfg,
        sites=sites,
        tariffs=tariffs,
        scenarios=scenarios,
        data_dir=Path(data_dir) if data_dir else config_dir.parent / "data",
    )
