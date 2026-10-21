"""Plant: site derates, part-load heat rate, capex and O&M lines (spec §5.3).

Simplifications, v0
- Derates use the design ambient temperature for every hour (ISO is 15 C).
- No ambient heat-rate adjustment (``ambient_hr_adj = 1``): the config has no input for it.
- Units run at equal load; the number running is the minimum that carries the load.
- Heat rate degrades linearly from the first-power year.
- Capital recovery starts in the first-power year, not prorated, and ignores
  construction-period financing.
- Fixed O&M is prorated for the part of the first-power year the plant is online.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcop.btm.gas import online_fraction
from dcop.io import resolve_ref
from dcop.ledger import LineFactory, capex_annuity_lines
from dcop.schemas import GlobalConfig, LedgerLine, PlantConfig, Site

ISO_AMBIENT_C = 15.0


def unit_mw_site(plant: PlantConfig, site: Site) -> float:
    ambient = max(0.0, plant.ambient_derate_pct_per_10c * (site.ambient_c_design - ISO_AMBIENT_C) / 10.0)
    elevation = max(0.0, plant.elevation_derate_pct_per_1000m * site.elevation_m / 1000.0)
    return plant.unit_mw_iso * (1.0 - ambient) * (1.0 - elevation)


def firm_mw(plant: PlantConfig, site: Site) -> float:
    """Capacity after holding back the redundant units: (n_units - k) x site unit rating."""
    return (plant.n_units - plant.redundancy_k) * unit_mw_site(plant, site)


def installed_kw(plant: PlantConfig) -> float:
    return plant.n_units * plant.unit_mw_iso * 1000.0


def load_part_load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """CSV with ``load_fraction`` and ``heat_rate_multiplier`` (1.0 at full load)."""
    df = pd.read_csv(path).sort_values("load_fraction")
    if not {"load_fraction", "heat_rate_multiplier"} <= set(df.columns):
        raise ValueError(f"{path}: needs columns load_fraction, heat_rate_multiplier")
    return df["load_fraction"].to_numpy(float), df["heat_rate_multiplier"].to_numpy(float)


def heat_rate_mmbtu_mwh(
    plant: PlantConfig,
    data_dir: Path,
    served_mw: np.ndarray,
    required: np.ndarray,
    unit_mw: float,
    index: pd.DatetimeIndex,
    first_power: pd.Timestamp,
) -> np.ndarray:
    fractions, multipliers = load_part_load_curve(resolve_ref(data_dir, plant.heat_rate_part_load_curve_ref))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(required > 0, served_mw / (required * unit_mw), 1.0)
    mult = np.interp(frac, fractions, multipliers)
    years_in_service = np.maximum(0, np.asarray(index.year) - first_power.year)
    return plant.heat_rate_mmbtu_mwh_hhv_iso * mult * (1.0 + plant.degradation_pct_yr * years_in_service)


def plant_ledger_lines(
    factory: LineFactory,
    plant: PlantConfig,
    cfg: GlobalConfig,
    first_power: pd.Timestamp,
    mwh_generated_by_year: dict[int, float],
    fuel_mmbtu_by_year: dict[int, float],
    lateral_principal_usd: float = 0.0,
) -> list[LedgerLine]:
    kw = installed_kw(plant)
    lines: list[LedgerLine] = []
    plant_principal = plant.capex_usd_kw * kw
    emissions_principal = plant.emissions_control_capex_usd_kw * kw
    if plant_principal:
        lines += capex_annuity_lines(factory, "plant_capex", plant_principal, cfg, first_power.year)
    if emissions_principal:
        lines += capex_annuity_lines(factory, "emissions_controls", emissions_principal, cfg, first_power.year)

    for year in cfg.ops_years:
        frac = online_fraction(year, first_power)
        if frac > 0 and plant.fixed_om_usd_kw_yr:
            lines.append(factory.line(year, "fixed_om", plant.fixed_om_usd_kw_yr * kw * frac))
        mwh = mwh_generated_by_year.get(year, 0.0)
        if mwh > 0:
            if plant.variable_om_usd_mwh:
                lines.append(factory.line(year, "variable_om", plant.variable_om_usd_mwh * mwh, "variable"))
            if plant.major_maintenance_usd_mwh:
                lines.append(factory.line(year, "variable_om", plant.major_maintenance_usd_mwh * mwh, "major_maintenance"))
        if plant.co2_price_usd_t is not None and fuel_mmbtu_by_year.get(year, 0.0) > 0:
            tonnes = fuel_mmbtu_by_year[year] * plant.emission_factor_t_co2_mmbtu
            lines.append(factory.line(year, "carbon", tonnes * plant.co2_price_usd_t, note=f"t_co2={tonnes:.0f}"))

    if cfg.btm_salvage_pct_capex:
        total = plant_principal + emissions_principal + lateral_principal_usd
        lines.append(factory.line(cfg.ops_years[-1], "residual", -cfg.btm_salvage_pct_capex * total, "salvage"))
    return lines
