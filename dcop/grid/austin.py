"""Austin Energy FY2026 bills: a residential branch and a commercial branch.

Residential uses the inclining-block Residential Service schedule. Commercial
uses the General Service secondary-voltage schedules, picked by the customer's
average June-September peak demand (tariff p.6). Inside city limits only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dcop.grid.bills import Statement, monthly_statement, usage_month_load
from dcop.schemas import Tariff

RESIDENTIAL_ID = "AE-RES-INSIDE"
GS_LT10_ID = "AE-GS-SEC-LT10"
GS_10_300_ID = "AE-GS-SEC-10-300"
GS_GE300_ID = "AE-GS-SEC-GE300"
PSA_ADMIN_ADJUSTMENT = "Administrative Adjustment to PSA"  # effective Dec 1, 2025


def general_service_tariff_id(summer_peak_kw_by_month: Sequence[float]) -> str:
    """Rate class from the average of the June-September monthly peak demands."""
    if not summer_peak_kw_by_month:
        raise ValueError("need at least one summer peak demand")
    avg = sum(summer_peak_kw_by_month) / len(summer_peak_kw_by_month)
    if avg < 10:
        return GS_LT10_ID
    return GS_10_300_ID if avg < 300 else GS_GE300_ID


def _without_psa_admin(tariff: Tariff) -> Tariff:
    return tariff.model_copy(
        update={"riders": [r for r in tariff.riders if r.name != PSA_ADMIN_ADJUSTMENT]}
    )


def residential_statement(
    tariffs: Mapping[str, Tariff], kwh: float, *, year: int = 2026, month: int = 1,
    include_psa_admin_adjustment: bool = True,
) -> Statement:
    """Residential bill for ``kwh`` in one month.

    The City's budget material quotes a typical bill without the Dec 1, 2025 PSA
    administrative adjustment. Pass ``include_psa_admin_adjustment=False`` to match it.
    """
    tariff = tariffs[RESIDENTIAL_ID]
    if not include_psa_admin_adjustment:
        tariff = _without_psa_admin(tariff)
    return monthly_statement(tariff, usage_month_load(kwh, None, year, month))


def commercial_statement(
    tariffs: Mapping[str, Tariff], kwh: float, peak_kw: float, *,
    power_factor: float | None = None, summer_peak_kw_by_month: Sequence[float] | None = None,
    year: int = 2026, month: int = 1, include_psa_admin_adjustment: bool = True,
) -> Statement:
    """General Service bill for ``kwh`` and a ``peak_kw`` 15-minute peak in one month.

    The rate class comes from ``summer_peak_kw_by_month`` (June-September peaks),
    defaulting to ``[peak_kw]`` when the summer history is unknown.
    """
    tariff_id = general_service_tariff_id(summer_peak_kw_by_month or [peak_kw])
    tariff = tariffs[tariff_id]
    if not include_psa_admin_adjustment:
        tariff = _without_psa_admin(tariff)
    # Customers below 10 kW have no demand charge; the peak only matters for class
    load = usage_month_load(kwh, None if tariff_id == GS_LT10_ID else peak_kw, year, month)
    return monthly_statement(tariff, load, power_factor=power_factor)
