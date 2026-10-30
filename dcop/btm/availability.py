"""Plant availability: EAF and binomial N+k probability (spec §5.3).

v0 assumes independent units. Common-mode gas interruption
(``interruptible_risk_days_yr``) is a v1 item and is not modeled here.
"""

from __future__ import annotations

from math import comb

import numpy as np

from dcop.schemas import PlantConfig


def eaf(plant: PlantConfig) -> float:
    """Equivalent availability: (1 - planned outage fraction) x (1 - EFOR)."""
    return (1.0 - plant.planned_outage_days_yr / 365.0) * (1.0 - plant.forced_outage_rate)


def required_units(served_mw: np.ndarray, unit_mw_site: float) -> np.ndarray:
    """Units that must run to carry ``served_mw``: ceil(load / unit capacity)."""
    return np.maximum(np.ceil(np.asarray(served_mw) / unit_mw_site - 1e-9), 0).astype(int)


def availability_table(n_units: int, unit_availability: float) -> np.ndarray:
    """``table[r] = P(at least r of n_units are available)`` for r = 0..n_units."""
    p = unit_availability
    pmf = np.array([comb(n_units, k) * p**k * (1 - p) ** (n_units - k) for k in range(n_units + 1)])
    return np.cumsum(pmf[::-1])[::-1]
