"""Tariff bill calculator (spec §4.2, §5.2).

Assumptions and limits, v0
- Timestamps are tz-naive local time, hour-beginning, on a regular interval.
- No holiday calendar. Holidays bill as the weekday/weekend they fall on.
- Months with zero delivered energy are not billed (not yet energized).
- Demand bases: ``15min`` is the peak 15-minute interval (hourly input data
  cannot see inside the hour, so it understates). ``ncp`` is the peak
  hourly-integrated demand. ``cp`` is the mean hourly demand over the
  utility-designated ``cp_hours`` in each month.
- Ratchet: billing demand = max(measured, ratchet_pct x max of measured demand
  over the trailing ``ratchet_lookback_months`` months). The ratchet looks at
  measured demand in all months, not only the months the charge applies to.
- ``usd_per_kw_mo`` riders bill on the largest billing demand among the demand
  charges that apply that month, else on the measured 15-minute peak.
- ``pct_of_bill`` riders apply to the month's subtotal (customer + energy +
  demand + per-unit riders). They do not compound on each other.
- Energy tiers bill each month's total kWh in blocks (inclining-block rates).
- Power factor: pass ``power_factor`` (rounded to the nearest hundredth). Below
  ``power_factor_min`` every measured demand is scaled by min / pf. The tariff
  rounds billed kW; this does not.
- ``Rider.annual_cap_usd`` caps one rider's charges per calendar year, prorated
  by billed months / 12. Feed whole calendar years for the cap to mean a full year,
  or pass ``service_months_per_year`` (a one-month statement passes 12 and assumes
  no cap was used earlier in the year).
- ``min_bill_trailing_months``: a month's bill is at least the highest measured kW
  over the last N months (this month included) x the demand rates, plus the
  riders marked ``in_minimum_bill``. The shortfall is a ``minimum_bill`` line.
- Minimum bill (take-or-pay, spec open question 2): a month's bill is at least
  ``min_bill_pct`` x (customer charge + applicable demand rates x contract
  demand). Contract demand defaults to the peak 15-minute demand of the series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcop.finance import escalate
from dcop.schemas import DemandCharge, GlobalConfig, LedgerLine, Tariff

CATEGORY_ENERGY = "energy"
CATEGORY_DEMAND = "demand"
CATEGORY_FEES = "riders_fees"
CATEGORY_MIN_BILL = "minimum_bill"


@dataclass(frozen=True)
class BillResult:
    """``lines`` has columns ym, year, month, category, subcategory, usd.

    ``ym`` is ``year * 12 + month - 1``. The monthly series are indexed by ``ym``
    and cover every month in the input, including unbilled ones.
    """

    lines: pd.DataFrame
    mwh_by_month: pd.Series
    billing_kw_by_month: pd.Series

    def total(self) -> float:
        return float(self.lines["usd"].sum())

    def by_category(self) -> pd.Series:
        return self.lines.groupby("category")["usd"].sum()


def _step_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("load series needs at least two timestamps")
    steps = np.unique(np.diff(index.values))
    if len(steps) != 1:
        raise ValueError("load series must have a regular time step")
    return float(steps[0] / np.timedelta64(1, "h"))


def _ym(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.year) * 12 + np.asarray(index.month) - 1


def _mask(
    index: pd.DatetimeIndex,
    months: Sequence[int] | None = None,
    hours: Sequence[int] | None = None,
    days: str = "all",
) -> np.ndarray:
    m = np.ones(len(index), dtype=bool)
    if months is not None:
        m &= np.isin(np.asarray(index.month), months)
    if hours is not None:
        m &= np.isin(np.asarray(index.hour), hours)
    if days == "weekday":
        m &= np.asarray(index.dayofweek) < 5
    elif days == "weekend":
        m &= np.asarray(index.dayofweek) >= 5
    return m


def _measured_demand(
    load_kw: pd.Series,
    basis: str,
    hours: Sequence[int] | None,
    days: str,
    cp_hours: pd.DatetimeIndex | None,
) -> pd.Series:
    """Monthly measured demand in kW under ``basis``, indexed by ym."""
    step = _step_hours(load_kw.index)
    s = load_kw
    if basis == "15min":
        if step < 0.25 - 1e-9:
            s = s.resample("15min").mean()
    elif step < 1.0 - 1e-9:
        s = s.resample("1h").mean()
    ym = _ym(s.index)

    if basis == "cp":
        if cp_hours is None:
            raise ValueError("basis 'cp' needs cp_hours (coincident peak timestamps)")
        sel = np.asarray(s.index.isin(cp_hours))
        picked = pd.Series(s.to_numpy()[sel]).groupby(ym[sel]).mean()
        missing = sorted(set(np.unique(ym)) - set(picked.index))
        if missing:
            y, m = divmod(missing[0], 12)
            raise ValueError(f"no coincident peak hour in cp_hours for {y}-{m + 1:02d}")
        return picked.sort_index()

    windowed = np.where(_mask(s.index, None, hours, days), s.to_numpy(), 0.0)
    return pd.Series(windowed).groupby(ym).max().sort_index()


def _billing_demand(
    measured: pd.Series, ratchet_pct: float, lookback: int, prior_kw: Sequence[float]
) -> pd.Series:
    vals = [*prior_kw, *measured.to_numpy()]
    n0 = len(prior_kw)
    out = []
    for i, current in enumerate(measured.to_numpy()):
        j = n0 + i
        history = vals[max(0, j - lookback) : j]
        trailing = max(history) if history else 0.0
        out.append(max(current, ratchet_pct * trailing))
    return pd.Series(out, index=measured.index)


def _power_factor_scale(tariff: Tariff, power_factor: float | None) -> float:
    if tariff.power_factor_min is None or power_factor is None:
        return 1.0
    pf = round(power_factor, 2)  # tariff rounds to the nearest hundredth
    return tariff.power_factor_min / pf if pf < tariff.power_factor_min else 1.0


def compute_monthly_bills(
    tariff: Tariff,
    load_kw: pd.Series,
    *,
    contract_demand_kw: float | None = None,
    prior_demand_kw: Sequence[float] = (),
    cp_hours: pd.DatetimeIndex | None = None,
    power_factor: float | None = None,
    service_months_per_year: int | None = None,
) -> BillResult:
    """Bill ``load_kw`` (a regular, tz-naive series covering whole months) under ``tariff``.

    ``prior_demand_kw`` is measured monthly demand preceding the series, oldest
    first, used to seed the ratchet.
    """
    if tariff.market_indexed:
        raise NotImplementedError("market-indexed tariffs need grid/market.py (not built yet)")
    index = load_kw.index
    step = _step_hours(index)
    kw = load_kw.to_numpy(dtype=float)
    if (kw < 0).any():
        raise ValueError("load must be non-negative")
    ym = _ym(index)
    months = np.unique(ym)
    if len(months) > 1 and (np.diff(months) != 1).any():
        raise ValueError("load series must cover contiguous months")
    mwh_i = kw * step / 1000.0
    mwh_m = pd.Series(mwh_i).groupby(ym).sum().sort_index()
    billed = [int(v) for v in mwh_m.index[mwh_m.to_numpy() > 0]]

    rows: list[tuple[int, str, str, float]] = []

    def add(month_key: int, category: str, sub: str, usd: float) -> None:
        rows.append((month_key, category, sub, float(usd)))

    # --- energy tiers: monthly kWh allocated across blocks ---
    for k in billed:
        lower, remaining = 0.0, float(mwh_m[k]) * 1000.0
        for i, tier in enumerate(tariff.energy_tiers, start=1):
            if remaining <= 0:
                break
            width = remaining if tier.up_to_kwh_mo is None else min(remaining, tier.up_to_kwh_mo - lower)
            label = f"tier {i} ({lower:g}-{tier.up_to_kwh_mo:g} kWh)" if tier.up_to_kwh_mo else f"tier {i} (over {lower:g} kWh)"
            add(k, CATEGORY_ENERGY, label, width * tier.usd_per_kwh)
            remaining -= width
            lower = tier.up_to_kwh_mo if tier.up_to_kwh_mo is not None else lower

    # --- energy: every interval must match exactly one block ---
    block_masks = [_mask(index, b.months, b.hours, b.days) for b in tariff.energy_charges]
    coverage = np.sum(block_masks, axis=0) if block_masks else np.ones(len(index), int)
    if (coverage != 1).any():
        bad = int(np.argmax(coverage != 1))
        raise ValueError(
            f"energy_charges must match each interval exactly once; "
            f"{index[bad]} matched {int(coverage[bad])} blocks"
        )
    for block, m in zip(tariff.energy_charges, block_masks):
        block_mwh = pd.Series(mwh_i * m).groupby(ym).sum()
        for k in billed:
            if block_mwh[k] > 0:
                add(k, CATEGORY_ENERGY, f"{block.season}/{block.period}", block_mwh[k] * block.usd_per_mwh)

    # --- demand (with ratchet) ---
    pf_factor = _power_factor_scale(tariff, power_factor)
    peak15 = _measured_demand(load_kw, "15min", None, "all", None).reindex(months, fill_value=0.0)
    peak15 = peak15 * pf_factor
    billing_kw: dict[int, float] = {}
    trailing_demand_usd: dict[int, float] = {}
    applicable: dict[int, list[DemandCharge]] = {k: [] for k in billed}
    for charge in tariff.demand_charges:
        measured = _measured_demand(load_kw, charge.basis, charge.hours, charge.days, cp_hours)
        measured = measured.reindex(months, fill_value=0.0) * pf_factor
        bd = _billing_demand(
            measured, charge.ratchet_pct, charge.ratchet_lookback_months, prior_demand_kw
        )
        trailing = (
            _billing_demand(measured, 1.0, tariff.min_bill_trailing_months - 1, prior_demand_kw)
            if tariff.min_bill_trailing_months
            else None
        )
        for k in billed:
            if (k % 12) + 1 in charge.months:
                add(k, CATEGORY_DEMAND, charge.name, bd[k] * charge.usd_per_kw_mo)
                if trailing is not None:
                    trailing_demand_usd[k] = trailing_demand_usd.get(k, 0.0) + trailing[k] * charge.usd_per_kw_mo
                billing_kw[k] = max(billing_kw.get(k, 0.0), float(bd[k]))
                applicable[k].append(charge)
    rider_kw = {k: billing_kw.get(k, float(peak15[k])) for k in billed}

    # --- customer charge and per-unit riders ---
    billed_per_year: dict[int, int] = {}
    for k in billed:
        billed_per_year[k // 12] = (
            service_months_per_year
            if service_months_per_year is not None
            else billed_per_year.get(k // 12, 0) + 1
        )
    capped_so_far: dict[tuple[int, str], float] = {}
    min_bill_riders: dict[int, float] = {k: 0.0 for k in billed}
    for k in billed:
        if tariff.customer_charge_usd_mo:
            add(k, CATEGORY_FEES, "customer_charge", tariff.customer_charge_usd_mo)
        for r in tariff.riders:
            if r.kind == "pct_of_bill" or (r.months is not None and (k % 12) + 1 not in r.months):
                continue
            base = {
                "usd_per_mwh": mwh_m[k],
                "usd_per_kwh": mwh_m[k] * 1000.0,
                "usd_per_kw_mo": rider_kw[k],
            }[r.kind]
            amount = base * r.value
            if r.annual_cap_usd is not None and amount > 0:
                room = r.annual_cap_usd * billed_per_year[k // 12] / 12.0 - capped_so_far.get((k // 12, r.name), 0.0)
                amount = max(0.0, min(amount, room))
                capped_so_far[(k // 12, r.name)] = capped_so_far.get((k // 12, r.name), 0.0) + amount
            if r.in_minimum_bill:
                min_bill_riders[k] += amount
            add(k, r.category, r.name, amount)

    # --- percent-of-bill riders on the subtotal so far ---
    subtotal: dict[int, float] = {}
    for k, _, _, usd in rows:
        subtotal[k] = subtotal.get(k, 0.0) + usd
    for k in billed:
        for r in tariff.riders:
            if r.kind == "pct_of_bill" and (r.months is None or (k % 12) + 1 in r.months):
                add(k, r.category, r.name, r.value * subtotal[k])

    # --- minimum bill true-up ---
    if tariff.min_bill_pct:
        contract = contract_demand_kw if contract_demand_kw is not None else float(peak15.max())
        computed: dict[int, float] = {}
        for k, _, _, usd in rows:
            computed[k] = computed.get(k, 0.0) + usd
        for k in billed:
            floor = tariff.min_bill_pct * (
                tariff.customer_charge_usd_mo
                + sum(c.usd_per_kw_mo for c in applicable[k]) * contract
            )
            if floor > computed[k]:
                add(k, CATEGORY_MIN_BILL, "min_bill_true_up", floor - computed[k])

    # --- trailing-demand minimum bill ---
    if tariff.min_bill_trailing_months:
        computed = {}
        for k, _, _, usd in rows:
            computed[k] = computed.get(k, 0.0) + usd
        for k in billed:
            floor = trailing_demand_usd.get(k, 0.0) + min_bill_riders[k]
            if floor > computed[k] + 1e-9:
                add(k, CATEGORY_MIN_BILL, "min_bill_trailing_demand", floor - computed[k])

    frame = pd.DataFrame(rows, columns=["ym", "category", "subcategory", "usd"])
    frame["year"] = frame["ym"] // 12
    frame["month"] = frame["ym"] % 12 + 1
    frame = frame[["ym", "year", "month", "category", "subcategory", "usd"]]
    return BillResult(
        lines=frame,
        mwh_by_month=mwh_m,
        billing_kw_by_month=pd.Series(billing_kw, dtype=float).sort_index(),
    )


def monthly_load_factors(load_kw: pd.Series) -> pd.Series:
    """Load factor per month: kWh / (peak kW x hours in the month), per the tariff glossary.

    Peak is the highest interval demand in the data. The series is indexed by ym.
    """
    step_h = _step_hours(load_kw.index)
    ym = _ym(load_kw.index)
    kwh = pd.Series(load_kw.to_numpy() * step_h).groupby(ym).sum()
    peak = pd.Series(load_kw.to_numpy()).groupby(ym).max()
    hours = pd.Series(1.0, index=load_kw.index).groupby(ym).sum() * step_h
    return (kwh / (peak * hours)).where(peak > 0)


def annual_average_monthly_load_factor(load_kw: pd.Series, year: int) -> float | None:
    """Average of the 12 monthly load factors in ``year``, or None if any month is unserved."""
    lf = monthly_load_factors(load_kw)
    months = [year * 12 + m for m in range(12)]
    vals = lf.reindex(months)
    return None if vals.isna().any() else float(vals.mean())


def tariff_ledger_lines(
    site_id: str,
    scenario_id: str,
    tariff: Tariff,
    delivered_kw: pd.Series,
    cfg: GlobalConfig,
    *,
    contract_demand_kw: float | None = None,
    cp_hours: pd.DatetimeIndex | None = None,
    mwh_by_year: Mapping[int, float] | None = None,
) -> list[LedgerLine]:
    """Annual grid ledger lines (energy, demand, riders/fees, minimum bill) for one site.

    Real tariff escalation compounds from the first operating year.
    """
    res = compute_monthly_bills(
        tariff, delivered_kw, contract_demand_kw=contract_demand_kw, cp_hours=cp_hours
    )
    mwh_year = (
        mwh_by_year
        if mwh_by_year is not None
        else res.mwh_by_month.groupby(res.mwh_by_month.index // 12).sum()
    )
    annual = res.lines.groupby(["year", "category", "subcategory"], as_index=False)["usd"].sum()
    out: list[LedgerLine] = []
    for row in annual.itertuples(index=False):
        factor = escalate(1.0, tariff.escalation_real_pct, row.year - cfg.first_ops_year)
        out.append(
            LedgerLine(
                site_id=site_id,
                path_id="grid",
                scenario_id=scenario_id,
                year=int(row.year),
                category=row.category,
                subcategory=row.subcategory,
                usd_real=float(row.usd) * factor,
                mwh_delivered=float(mwh_year[row.year]),
                note=f"tariff={tariff.tariff_id}",
            )
        )
    return out
