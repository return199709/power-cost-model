"""Finance primitives: annuity, NPV, levelization, escalation, real/nominal.

Timing convention: a cash flow dated calendar year Y is discounted by
``(1 + r) ** (Y - base_year)``. Base-year flows are not discounted.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def discount_factor(rate: float, t: float) -> float:
    return (1.0 + rate) ** (-t)


def npv(flows_by_year: Mapping[int, float], rate: float, base_year: int) -> float:
    """Present value in base_year of ``{calendar_year: amount}``."""
    return sum(v * discount_factor(rate, y - base_year) for y, v in flows_by_year.items())


def annuity_payment(principal: float, rate: float, n_years: int) -> float:
    """Level end-of-year payment that repays ``principal`` over ``n_years`` at ``rate``."""
    if n_years < 1:
        raise ValueError("n_years must be >= 1")
    if rate == 0:
        return principal / n_years
    return principal * rate / (1.0 - (1.0 + rate) ** (-n_years))


def annuity_schedule(
    principal: float, rate: float, n_years: int, first_payment_year: int
) -> dict[int, float]:
    """``{year: payment}`` for a capex annuity whose first payment is in ``first_payment_year``."""
    pmt = annuity_payment(principal, rate, n_years)
    return {first_payment_year + i: pmt for i in range(n_years)}


def escalate(value: float, rate: float, years: float) -> float:
    """Compound ``value`` at ``rate`` for ``years`` (used for real escalation)."""
    return value * (1.0 + rate) ** years


def to_real(value: float, dollar_year: int, base_year: int, inflation: float) -> float:
    """Convert nominal ``dollar_year`` dollars to real ``base_year`` dollars.

    A 2030 nominal dollar deflates into 2026 dollars; a 2024 nominal dollar
    inflates into 2026 dollars.
    """
    return value / (1.0 + inflation) ** (dollar_year - base_year)


def levelize(
    cost_by_year: Mapping[int, float],
    mwh_by_year: Mapping[int, float],
    rate: float,
    base_year: int,
) -> float:
    """LDCoP = NPV(cost) / NPV(delivered MWh), in $/MWh (spec §6)."""
    denom = npv(mwh_by_year, rate, base_year)
    if denom <= 0:
        raise ValueError("NPV of delivered MWh must be positive to levelize")
    return npv(cost_by_year, rate, base_year) / denom


def online_fraction(year: int, start: pd.Timestamp) -> float:
    """Share of calendar ``year`` on or after ``start``."""
    if year < start.year:
        return 0.0
    if year > start.year:
        return 1.0
    days = (pd.Timestamp(year=year, month=12, day=31) - start.normalize()).days + 1
    return days / (366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365)
