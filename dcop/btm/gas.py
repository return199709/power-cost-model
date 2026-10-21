"""Gas supply: delivered price and fixed transport (spec §5.3).

Curves are CSV files with columns ``month`` (YYYY-MM) and ``usd_mmbtu`` in real
base-year dollars. A request after the last month holds the last value flat. A
request before the first month is an error.

    gas_delivered[m] = (HH[m] + basis[m]) / (1 - fuel_retention) + transport_commodity

``interruptible_risk_days_yr`` is not modeled in v0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcop.io import resolve_ref
from dcop.ledger import LineFactory, capex_annuity_lines
from dcop.schemas import GasSupply, GlobalConfig, LedgerLine


def load_curve(path: Path) -> pd.Series:
    """Monthly curve indexed by ym (``year * 12 + month - 1``)."""
    df = pd.read_csv(path)
    if not {"month", "usd_mmbtu"} <= set(df.columns):
        raise ValueError(f"{path}: needs columns 'month' and 'usd_mmbtu'")
    months = pd.PeriodIndex(df["month"], freq="M")
    ym = months.year * 12 + months.month - 1
    return pd.Series(df["usd_mmbtu"].to_numpy(float), index=np.asarray(ym)).sort_index()


def curve_at(curve: pd.Series, ym: np.ndarray, name: str) -> np.ndarray:
    """Exact month lookup. Gaps and months after the end hold the previous value."""
    ym = ym.astype(int)
    if ym.min() < curve.index.min():
        y, m = divmod(int(curve.index.min()), 12)
        raise ValueError(f"{name} curve starts {y}-{m + 1:02d}, after the first requested month")
    full = curve.reindex(np.arange(curve.index.min(), max(ym.max(), curve.index.max()) + 1)).ffill()
    return full.loc[ym].to_numpy(float)


def monthly_price_components(
    gas: GasSupply, data_dir: Path, ym: np.ndarray
) -> pd.DataFrame:
    """$/MMBtu delivered by component for each month in ``ym`` (sorted unique)."""
    hh = curve_at(load_curve(resolve_ref(data_dir, gas.henry_hub_curve_ref)), ym, "henry_hub")
    basis = curve_at(load_curve(resolve_ref(data_dir, gas.basis_curve_ref)), ym, "basis")
    gross = 1.0 / (1.0 - gas.fuel_retention_pct)
    return pd.DataFrame(
        {
            "henry_hub": hh * gross,
            "basis": basis * gross,
            "transport_commodity": gas.transport_commodity_usd_mmbtu,
        },
        index=ym,
    )


def online_fraction(year: int, first_power: pd.Timestamp) -> float:
    """Share of ``year`` on or after the first-power date."""
    if year < first_power.year:
        return 0.0
    if year > first_power.year:
        return 1.0
    days = (pd.Timestamp(year=year, month=12, day=31) - first_power.normalize()).days + 1
    return days / (366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365)


def gas_ledger_lines(
    factory: LineFactory,
    gas: GasSupply,
    cfg: GlobalConfig,
    data_dir: Path,
    index: pd.DatetimeIndex,
    fuel_mmbtu: np.ndarray,
    first_power: pd.Timestamp,
) -> list[LedgerLine]:
    """Fuel by price component, fixed firm transport, and lateral capital recovery."""
    ym = np.asarray(index.year) * 12 + np.asarray(index.month) - 1
    monthly_fuel = pd.Series(fuel_mmbtu).groupby(ym).sum()
    prices = monthly_price_components(gas, data_dir, monthly_fuel.index.to_numpy())
    cost = prices.mul(monthly_fuel, axis=0)
    cost["year"] = monthly_fuel.index.to_numpy() // 12
    by_year = cost.groupby("year").sum()

    lines: list[LedgerLine] = []
    for year, row in by_year.iterrows():
        for comp in ("henry_hub", "basis", "transport_commodity"):
            if row[comp]:
                lines.append(factory.line(year, "fuel", row[comp], comp))

    # Contracted MDQ (MMBtu/day) sized to the peak hour of the run
    step_h = float((index[1] - index[0]) / pd.Timedelta(hours=1))
    mdq = float(fuel_mmbtu.max() / step_h * 24.0)
    for year in cfg.ops_years:
        frac = online_fraction(year, first_power)
        if frac > 0:
            lines.append(
                factory.line(year, "transport_fixed", gas.transport_firm_usd_mmbtu * mdq * 365 * frac,
                             note=f"mdq_mmbtu_d={mdq:.0f}")
            )
    if gas.lateral_capex_usd:
        lines += capex_annuity_lines(
            factory, "lateral", gas.lateral_capex_usd, cfg, first_power.year
        )
    return lines
