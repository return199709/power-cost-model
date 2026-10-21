"""Load profile generation, ramp scaling, and delivered energy (spec §5.1).

The synthetic shape is a placeholder (assumption A3). It peaks at 1.0 once per
year (mid-July, 15:00) and its annual mean equals ``load_factor``. Supply a real
profile with ``load_profile_ref`` (CSV with a ``shape`` column of 8760 hourly
fractions of peak) once one is available.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dcop.schemas import Site

_PERIOD_RE = re.compile(r"^(\d{4})(?:-Q([1-4]))?$")


def parse_period(period: str) -> pd.Timestamp:
    """Start of ``"YYYY-Qn"`` (quarter) or ``"YYYY"`` (year)."""
    m = _PERIOD_RE.match(period)
    if not m:
        raise ValueError(f"period must look like '2028-Q1' or '2028', got {period!r}")
    quarter = int(m.group(2)) if m.group(2) else 1
    return pd.Timestamp(year=int(m.group(1)), month=3 * (quarter - 1) + 1, day=1)


def ramp_mw(site: Site, index: pd.DatetimeIndex) -> np.ndarray:
    """Cumulative energized MW at each timestamp (step function, 0 before the first step)."""
    out = np.zeros(len(index))
    for period, mw in site.load_mw_ramp:
        out[np.asarray(index >= parse_period(period))] = mw
    return out


def synthetic_shape(load_factor: float, index: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic shape in [0, 1] with max 1.0 and annual mean ``load_factor``."""
    if load_factor < 0.5:
        raise ValueError("synthetic shape needs load_factor >= 0.5; provide load_profile_ref")
    doy = np.asarray(index.dayofyear, dtype=float)
    hour = np.asarray(index.hour, dtype=float) + np.asarray(index.minute) / 60.0
    seasonal = 0.5 * (1 - np.cos(2 * np.pi * (doy - 200) / 365))  # 0 at mid-July
    diurnal = 0.5 * (1 - np.cos(2 * np.pi * (hour - 15) / 24))  # 0 at 15:00
    u = 0.7 * seasonal + 0.3 * diurnal  # in [0, 1], mean 0.5
    return 1.0 - 2.0 * (1.0 - load_factor) * u


def shape_from_csv(path: Path, index: pd.DatetimeIndex) -> np.ndarray:
    shape = pd.read_csv(path)["shape"].to_numpy(dtype=float)
    if len(shape) != 8760:
        raise ValueError(f"{path} must have 8760 hourly rows, got {len(shape)}")
    if shape.min() < 0 or shape.max() > 1 + 1e-9:
        raise ValueError(f"{path}: shape values must be fractions of peak in [0, 1]")
    pos = np.minimum((np.asarray(index.dayofyear) - 1) * 24 + np.asarray(index.hour), 8759)
    return shape[pos]


def site_load_kw(
    site: Site,
    years: Sequence[int],
    freq: str = "h",
    data_dir: Path | None = None,
) -> pd.Series:
    """Load in kW over whole calendar ``years``: shape x energized MW x 1000."""
    index = pd.date_range(
        pd.Timestamp(year=years[0], month=1, day=1),
        pd.Timestamp(year=years[-1] + 1, month=1, day=1),
        freq=freq,
        inclusive="left",
    )
    if site.load_profile_ref:
        from dcop.io import resolve_ref

        if data_dir is None:
            raise ValueError("data_dir is required when load_profile_ref is set")
        shape = shape_from_csv(resolve_ref(data_dir, site.load_profile_ref), index)
    else:
        shape = synthetic_shape(site.load_factor, index)
    return pd.Series(shape * ramp_mw(site, index) * 1000.0, index=index, name="load_kw")


def delivered_and_unserved(
    load_kw: pd.Series, availability: float | pd.Series = 1.0, curtailment: float | pd.Series = 0.0
) -> tuple[pd.Series, pd.Series]:
    """delivered = load x availability x (1 - curtailment); unserved = load - delivered."""
    delivered = load_kw * availability * (1.0 - curtailment)
    return delivered, load_kw - delivered


def mwh_by_year(kw: pd.Series) -> dict[int, float]:
    """Energy in MWh per calendar year from a regular kW series."""
    step_h = float((kw.index[1] - kw.index[0]) / pd.Timedelta(hours=1))
    return (kw.groupby(np.asarray(kw.index.year)).sum() * step_h / 1000.0).to_dict()
