"""Cost ledger container, aggregation, and reconciliation (spec §6, §9)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcop.finance import annuity_schedule
from dcop.schemas import GlobalConfig, LedgerLine

ID_KEYS = ["site_id", "path_id", "scenario_id"]
ATOMIC_KEYS = [*ID_KEYS, "year"]
_COLUMNS = list(LedgerLine.model_fields)


@dataclass(frozen=True)
class LineFactory:
    """Builds ledger lines for one (site, path, scenario), stamping each with that year's MWh.

    Years with no delivered energy (before first power) carry ``mwh_delivered = 0``.
    """

    site_id: str
    path_id: str
    scenario_id: str
    mwh_by_year: Mapping[int, float]

    def line(
        self,
        year: int,
        category: str,
        usd: float,
        subcategory: str | None = None,
        *,
        is_capex: bool = False,
        is_exposure: bool = False,
        note: str | None = None,
    ) -> LedgerLine:
        return LedgerLine(
            site_id=self.site_id,
            path_id=self.path_id,  # type: ignore[arg-type]
            scenario_id=self.scenario_id,
            year=int(year),
            category=category,
            subcategory=subcategory,
            usd_real=float(usd),
            mwh_delivered=float(self.mwh_by_year.get(int(year), 0.0)),
            is_capex=is_capex,
            is_exposure=is_exposure,
            note=note,
        )


def capex_annuity_lines(
    factory: LineFactory,
    category: str,
    principal: float,
    cfg: GlobalConfig,
    first_year: int,
    subcategory: str | None = None,
    note: str | None = None,
) -> list[LedgerLine]:
    """Capital recovery: level payments from ``first_year`` to the end of the horizon.

    Amortizing over the remaining horizon at the discount rate means a later
    in-service date raises the annual payment, which is how delay shows up in capex.
    """
    last_year = cfg.ops_years[-1]
    if first_year > last_year:
        raise ValueError(f"{category}: in service in {first_year}, after the horizon ends ({last_year})")
    n_years = last_year - first_year + 1
    sched = annuity_schedule(principal, cfg.discount_rate, n_years, first_year)
    return [
        factory.line(y, category, pmt, subcategory, is_capex=True, note=note)
        for y, pmt in sched.items()
    ]


class ReconciliationError(AssertionError):
    pass


class Ledger:
    def __init__(self, lines: Iterable[LedgerLine] = ()) -> None:
        self._lines: list[LedgerLine] = list(lines)

    def extend(self, lines: Iterable[LedgerLine]) -> None:
        self._lines.extend(lines)

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self) -> Iterator[LedgerLine]:
        return iter(self._lines)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([ln.model_dump() for ln in self._lines], columns=_COLUMNS)

    def validate_mwh(self) -> None:
        """``mwh_delivered`` must be identical across lines of a (site, path, scenario, year)."""
        df = self.to_frame()
        grp = df.groupby(ATOMIC_KEYS)["mwh_delivered"]
        spread = (grp.max() - grp.min()) / grp.max().clip(lower=1.0)
        bad = spread[spread > 1e-9]
        if len(bad):
            raise ValueError(f"inconsistent mwh_delivered within {bad.index[0]}")

    def totals(self) -> pd.DataFrame:
        """Total real USD per (site, path, scenario)."""
        df = self.to_frame()
        return df.groupby(ID_KEYS, as_index=False)["usd_real"].sum()

    def aggregate(
        self, by: list[str], kw_by_site: Mapping[str, float] | None = None
    ) -> pd.DataFrame:
        """Group by any subset of keys: ``usd``, ``usd_per_mwh``, ``usd_per_kw_yr``, ``share``.

        $/MWh and $/kW-yr divide by delivered MWh and peak kW-years of the
        (site, path, scenario, year) combinations that appear in ``by``, so a
        category that is absent in some years is not flattered. ``share`` is the
        fraction of the total within those same combinations.
        """
        self.validate_mwh()
        df = self.to_frame()
        atomic = df.drop_duplicates(ATOMIC_KEYS)[[*ATOMIC_KEYS, "mwh_delivered"]].copy()
        if kw_by_site:
            atomic["kw_years"] = atomic["site_id"].map(dict(kw_by_site)).astype(float)
        else:
            atomic["kw_years"] = np.nan

        num = df.groupby(by, dropna=False)["usd_real"].sum().rename("usd").reset_index()
        ctx = [k for k in by if k in ATOMIC_KEYS]
        if ctx:
            den = (
                atomic.groupby(ctx)
                .agg(mwh=("mwh_delivered", "sum"), kw_years=("kw_years", "sum"))
                .reset_index()
            )
            out = num.merge(den, on=ctx, how="left")
        else:
            out = num.assign(mwh=atomic["mwh_delivered"].sum(), kw_years=atomic["kw_years"].sum())

        out["usd_per_mwh"] = out["usd"] / out["mwh"].where(out["mwh"] > 0)
        out["usd_per_kw_yr"] = out["usd"] / out["kw_years"].where(out["kw_years"] > 0)
        total = out.groupby(ctx)["usd"].transform("sum") if ctx else out["usd"].sum()
        out["share"] = out["usd"] / total
        return out


def reconcile(ledger: Ledger, summary: pd.DataFrame, tol: float = 1e-6) -> None:
    """Sum of ledger lines must equal the summary total per (site, path, scenario) (spec §9)."""
    led = ledger.totals().set_index(ID_KEYS)["usd_real"]
    summ = summary.set_index(ID_KEYS)["total_cost_usd_real"]
    if set(led.index) != set(summ.index):
        raise ReconciliationError("ledger and summary cover different (site, path, scenario) keys")
    diff = (led - summ.reindex(led.index)).abs()
    scale = max(1.0, float(led.abs().max()))
    if (diff > tol * scale).any():
        key = diff.idxmax()
        raise ReconciliationError(f"ledger != summary for {key}: off by {diff.max():.6f}")
