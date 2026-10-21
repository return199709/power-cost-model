"""Timing engine: when firm power arrives, MW online by quarter, unserved energy (spec §5.4).

``mw_online = min(load ramp, path capacity)`` once the path's first-power date
has passed, and zero before it. The quarterly series is sampled at quarter
start for reporting. Cost calculations use the exact hourly timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dcop.ledger import LineFactory
from dcop.load import ramp_mw
from dcop.schemas import GasSupply, PlantConfig, Site, UpgradeExposure


@dataclass(frozen=True)
class PathTiming:
    path_id: str
    lead_months: int
    first_power_date: pd.Timestamp
    capacity_mw: float  # np.inf when the path is not capacity limited
    mw_online: pd.Series  # indexed by quarter start
    full_power_date: pd.Timestamp  # NaT if the path never carries the full ramp


def add_months(start: date, months: int) -> pd.Timestamp:
    return pd.Timestamp(start) + pd.DateOffset(months=months)


def grid_lead_months(exposure: UpgradeExposure | None) -> int:
    """Longest lead time among critical-path upgrade items (0 if none are listed)."""
    if exposure is None:
        return 0
    return max((i.lead_time_months for i in exposure.items if i.on_critical_path), default=0)


def btm_lead_months(plant: PlantConfig, gas: GasSupply) -> int:
    """Permitting then construction in sequence, in parallel with the gas lateral.

    The spec says "max lead time on the critical path"; permit and build cannot
    overlap, so they are summed before taking the max with the lateral.
    """
    return max(plant.permit_lead_months + plant.cod_lead_months, gas.lateral_lead_months)


def build_timing(
    path_id: str,
    site: Site,
    clock_start: date,
    lead_months: int,
    capacity_mw: float,
    years: list[int],
) -> PathTiming:
    first_power = add_months(clock_start, lead_months)
    quarters = pd.date_range(
        pd.Timestamp(year=years[0], month=1, day=1),
        pd.Timestamp(year=years[-1] + 1, month=1, day=1),
        freq="QS",
        inclusive="left",
    )
    online = np.where(
        np.asarray(quarters >= first_power), np.minimum(ramp_mw(site, quarters), capacity_mw), 0.0
    )
    full_mw = max(mw for _, mw in site.load_mw_ramp)
    reached = np.flatnonzero(online >= full_mw - 1e-9)
    return PathTiming(
        path_id=path_id,
        lead_months=lead_months,
        first_power_date=first_power,
        capacity_mw=capacity_mw,
        mw_online=pd.Series(online, index=quarters, name=f"{path_id}_mw_online"),
        full_power_date=quarters[reached[0]] if len(reached) else pd.NaT,
    )


def capacity_mw_series(index: pd.DatetimeIndex, first_power: pd.Timestamp, capacity_mw: float) -> np.ndarray:
    """Path capacity at each timestamp: zero before first power, then ``capacity_mw``."""
    return np.where(np.asarray(index >= first_power), capacity_mw, 0.0)


def unserved_lines(
    factory: LineFactory,
    load_kw: pd.Series,
    served_kw: np.ndarray,
    delivered_kw: np.ndarray,
    first_power: pd.Timestamp,
    value_of_unserved_usd_mwh: float | None,
) -> list:
    """Ledger lines pricing unserved energy (spec §5.1, §5.3).

    - ``delay``: load before first power.
    - ``unserved_energy/capacity_shortfall``: load above the path's capacity.
    - ``unserved_energy/availability``: expected loss from unit outages.
    """
    index = load_kw.index
    step_h = float((index[1] - index[0]) / pd.Timedelta(hours=1))
    load = load_kw.to_numpy()
    pre = np.asarray(index < first_power)
    parts = {
        ("delay", None): np.where(pre, load - served_kw, 0.0),
        ("unserved_energy", "capacity_shortfall"): np.where(pre, 0.0, load - served_kw),
        ("unserved_energy", "availability"): served_kw - delivered_kw,
    }
    years = np.asarray(index.year)
    lines = []
    for (category, sub), kw in parts.items():
        mwh_by_year = pd.Series(kw * step_h / 1000.0).groupby(years).sum()
        for year, mwh in mwh_by_year.items():
            if mwh <= 1e-6:
                continue
            if value_of_unserved_usd_mwh is None:
                raise ValueError(
                    "value_of_unserved_usd_mwh (assumption A2) must be set: "
                    f"{mwh:,.0f} MWh unserved in {year}"
                )
            lines.append(
                factory.line(year, category, mwh * value_of_unserved_usd_mwh, sub,
                             note=f"unserved_mwh={mwh:.1f}")
            )
    return lines
