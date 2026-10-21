"""Summary tables and run-time validation (spec §8, §9). Charts arrive with M1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcop.finance import levelize
from dcop.ledger import ID_KEYS, reconcile
from dcop.scenarios import RunResult


class OutOfRangeError(ValueError):
    pass


def summary_table(result: RunResult) -> pd.DataFrame:
    """One row per (site, path, scenario).

    Columns still to come: exposure P50/P90 (M3), first/full power date and
    availability (M1).
    """
    df = result.ledger.to_frame()
    rows = []
    for (site_id, path_id, scenario_id), g in df.groupby(ID_KEYS):
        eff = result.inputs_by_scenario[scenario_id]
        cfg = eff.global_cfg
        cost = g.groupby("year")["usd_real"].sum().to_dict()
        mwh = g.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()

        def undiscounted(year_index: int) -> float:
            if year_index >= len(cfg.ops_years):
                return np.nan
            y = cfg.ops_years[year_index]
            return cost.get(y, 0.0) / mwh[y] if mwh.get(y, 0.0) > 0 else np.nan

        rows.append(
            {
                "site_id": site_id,
                "path_id": path_id,
                "scenario_id": scenario_id,
                "ldcop_usd_mwh": levelize(cost, mwh, cfg.discount_rate, cfg.base_year),
                "year1_usd_mwh": undiscounted(0),
                "year5_usd_mwh": undiscounted(4),
                "total_cost_usd_real": float(g["usd_real"].sum()),
                "capex_total_usd": float(g.loc[g["is_capex"], "usd_real"].sum()),
                "mwh_delivered_total": float(sum(mwh.values())),
                "peak_kw": eff.sites[site_id].load_mw_peak * 1000.0,
                "discount_rate": cfg.discount_rate,
            }
        )
    return pd.DataFrame(rows)


def check_bounds(summary: pd.DataFrame, result: RunResult) -> None:
    """Fail if any LDCoP is outside the configured bounds unless allowed (spec §9)."""
    for row in summary.itertuples(index=False):
        cfg = result.inputs_by_scenario[row.scenario_id].global_cfg
        lo, hi = cfg.ldcop_bounds_usd_mwh
        if not cfg.allow_out_of_range and not lo <= row.ldcop_usd_mwh <= hi:
            raise OutOfRangeError(
                f"{row.site_id}/{row.path_id}/{row.scenario_id}: LDCoP "
                f"{row.ldcop_usd_mwh:.2f} $/MWh is outside [{lo}, {hi}]; "
                "check inputs or set allow_out_of_range"
            )


def validated_summary(result: RunResult) -> pd.DataFrame:
    """Build the summary and run the reconciliation and sanity-bound checks."""
    summary = summary_table(result)
    reconcile(result.ledger, summary)
    check_bounds(summary, result)
    return summary
