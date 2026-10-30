"""Itemized monthly bill statements (what a customer sees on one bill).

Built on ``compute_monthly_bills``. A statement rounds each line to the cent and
totals the rounded lines, as a printed bill does.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from dcop.grid.tariff import compute_monthly_bills
from dcop.schemas import Tariff

_ORDER = {"customer_charge": 0, "energy": 1, "demand": 2, "riders_fees": 3, "minimum_bill": 4}


def _label(category: str, sub: str) -> str:
    if category == "energy" and sub == "all/all":
        return "Energy Charge"
    if category == "energy" and ("/" in sub or sub.startswith("tier ")):
        return f"Energy {sub}"
    return sub


def round_cents(value: float) -> float:
    return float(Decimal(repr(float(value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def usage_month_load(
    kwh: float, peak_kw: float | None = None, year: int = 2026, month: int = 1, freq: str = "15min"
) -> pd.Series:
    """One billing month of load (kW) that uses ``kwh`` and, if given, peaks at ``peak_kw``.

    The load is flat except for a single peak interval.
    """
    start = pd.Timestamp(year=year, month=month, day=1)
    index = pd.date_range(start, start + pd.DateOffset(months=1), freq=freq, inclusive="left")
    step_h = float((index[1] - index[0]) / pd.Timedelta(hours=1))
    hours = len(index) * step_h
    if peak_kw is None:
        return pd.Series(kwh / hours, index=index)
    rest = (kwh - peak_kw * step_h) / (hours - step_h)
    if rest < 0 or rest > peak_kw:
        raise ValueError(
            f"peak {peak_kw} kW is inconsistent with {kwh} kWh over {hours:.0f} h "
            f"(average {kwh / hours:.2f} kW): the peak must be at least the average"
        )
    load = pd.Series(rest, index=index)
    load.iloc[len(index) // 2] = peak_kw
    return load


@dataclass(frozen=True)
class Statement:
    tariff_id: str
    kwh: float
    billed_kw: float | None
    items: pd.DataFrame  # columns: item, category, usd (rounded to cents)

    @property
    def total_usd(self) -> float:
        return round_cents(self.items["usd"].sum())

    @property
    def cents_per_kwh(self) -> float:
        return self.total_usd / self.kwh * 100.0

    def usd(self, item: str) -> float:
        """Amount of one line, matched by exact label."""
        return float(self.items.loc[self.items["item"] == item, "usd"].sum())

    def render(self) -> str:
        width = max(len(i) for i in self.items["item"]) + 2
        rows = [f"  {r.item:<{width}}{r.usd:>10,.2f}" for r in self.items.itertuples()]
        head = f"{self.tariff_id}: {self.kwh:,.0f} kWh" + (
            f", {self.billed_kw:,.1f} billed kW" if self.billed_kw else ""
        )
        return "\n".join([head, *rows, f"  {'Total':<{width}}{self.total_usd:>10,.2f}",
                          f"  {'Effective rate (cents/kWh)':<{width}}{self.cents_per_kwh:>10,.2f}"])


def monthly_statement(
    tariff: Tariff, load_kw: pd.Series, *, power_factor: float | None = None,
    contract_demand_kw: float | None = None,
) -> Statement:
    """Itemize the single billing month in ``load_kw``."""
    res = compute_monthly_bills(
        tariff, load_kw, power_factor=power_factor, contract_demand_kw=contract_demand_kw,
        service_months_per_year=12,  # a statement is one month of ongoing service
    )
    if len(res.mwh_by_month) != 1:
        raise ValueError("a statement covers exactly one month")
    lines = res.lines.copy()
    lines["order"] = lines["category"].map(_ORDER).where(lines["subcategory"] != "customer_charge", 0)
    lines = lines.sort_values("order", kind="stable")
    label = {"customer_charge": "Customer Charge"}
    items = pd.DataFrame(
        {
            "item": [
                label.get(sub, _label(cat, sub))
                for cat, sub in zip(lines["category"], lines["subcategory"])
            ],
            "category": lines["category"].to_numpy(),
            "usd": [round_cents(u) for u in lines["usd"]],
        }
    ).reset_index(drop=True)
    billed_kw = float(res.billing_kw_by_month.iloc[0]) if len(res.billing_kw_by_month) else None
    return Statement(tariff.tariff_id, float(res.mwh_by_month.iloc[0]) * 1000.0, billed_kw, items)
