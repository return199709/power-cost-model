"""Network upgrade cost, deterministic expected value (spec §5.2).

Each item costs ``P(trigger) x allocation x P50`` as a lump sum in the year its
lead time ends (clock start + lead_time_months). Items with ``ciac`` are
non-refundable. Others refund ``cost x pct`` at ``payment year + years`` per
``refund_schedule`` (negative lines); refunds that land after the horizon are
dropped. P90 values, probabilistic sampling and the exposure table are M3.
"""

from __future__ import annotations

from dcop.ledger import LineFactory
from dcop.schemas import GlobalConfig, LedgerLine, UpgradeExposure
from dcop.timing import add_months


def upgrade_ledger_lines(
    factory: LineFactory, exposure: UpgradeExposure, cfg: GlobalConfig
) -> list[LedgerLine]:
    lines: list[LedgerLine] = []
    for item in exposure.items:
        expected = item.probability_triggered * item.allocation_pct * item.cost_usd_p50
        if expected == 0:
            continue
        pay_year = add_months(cfg.clock_start_date, item.lead_time_months).year
        exposed = item.probability_triggered < 1.0
        note = f"p={item.probability_triggered:g} alloc={item.allocation_pct:g}"
        lines.append(
            factory.line(pay_year, "upgrade_capex", expected, item.name,
                         is_capex=True, is_exposure=exposed, note=note)
        )
        if not item.ciac and item.refund_schedule:
            last_year = cfg.ops_years[-1]
            for years_after, pct in item.refund_schedule:
                year = pay_year + years_after
                if year <= last_year and pct:
                    lines.append(
                        factory.line(year, "upgrade_refund", -expected * pct, item.name,
                                     is_exposure=exposed, note=note)
                    )
    return lines
