"""Summary tables, levelized components, charts, and run-time validation (spec §6, §8, §9).

Still to come: assumptions.md and staleness flags (M4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dcop.finance import levelize, npv
from dcop.ledger import ID_KEYS, ReconciliationError, reconcile
from dcop.io import Inputs
from dcop.load import ramp_mw
from dcop.scenarios import PathRun, RunResult

# Ledger categories folded into at most eight chart groups, in fixed stack order.
CATEGORY_GROUPS: dict[str, list[str]] = {
    "Energy / fuel": ["energy", "fuel"],
    "Demand / firm transport": ["demand", "transport_fixed"],
    "Riders & fees": ["riders_fees", "minimum_bill"],
    "Capital (plant, lateral)": ["plant_capex", "emissions_controls", "lateral"],
    "Grid upgrades (net)": ["upgrade_capex", "upgrade_refund"],
    "O&M": ["fixed_om", "variable_om"],
    "Delay & unserved": ["delay", "unserved_energy"],
    "Carbon & residual": ["carbon", "residual"],
}
_GROUP_OF = {cat: grp for grp, cats in CATEGORY_GROUPS.items() for cat in cats}


class OutOfRangeError(ValueError):
    pass


def _group_of(category: str) -> str:
    if category not in _GROUP_OF:
        raise KeyError(f"ledger category {category!r} has no chart group; add it to CATEGORY_GROUPS")
    return _GROUP_OF[category]


def _path_lookup(result: RunResult) -> dict[tuple[str, str, str], PathRun]:
    return {(p.site_id, p.path_id, p.scenario_id): p for p in result.paths}


def summary_table(result: RunResult) -> pd.DataFrame:
    """One row per (site, path, scenario).

    Year 1 is the path's first year with delivered energy and year 5 is four
    years later. Both are undiscounted $/MWh. ``capex_total_usd`` sums capital
    recovery payments and upfront upgrade payments, so it includes financing.
    """
    df = result.ledger.to_frame()
    runs = _path_lookup(result)
    rows = []
    for key, g in df.groupby(ID_KEYS):
        site_id, path_id, scenario_id = key
        run = runs[key]
        eff = result.inputs_by_scenario[scenario_id]
        cfg = eff.global_cfg
        cost = g.groupby("year")["usd_real"].sum().to_dict()
        mwh = g.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()
        live_years = sorted(y for y, m in mwh.items() if m > 0)

        def undiscounted(offset: int) -> float:
            if len(live_years) <= offset:
                return np.nan
            y = live_years[offset]
            return cost.get(y, 0.0) / mwh[y]

        capex = g[g["is_capex"]]
        rows.append(
            {
                "site_id": site_id,
                "path_id": path_id,
                "scenario_id": scenario_id,
                "ldcop_usd_mwh": levelize(cost, mwh, cfg.discount_rate, cfg.base_year),
                "year1_usd_mwh": undiscounted(0),
                "year5_usd_mwh": undiscounted(4),
                "total_cost_usd_real": float(g["usd_real"].sum()),
                "capex_total_usd": float(capex["usd_real"].sum()),
                "capex_npv_usd": npv(
                    capex.groupby("year")["usd_real"].sum().to_dict(), cfg.discount_rate, cfg.base_year
                ),
                "delay_cost_usd": float(g.loc[g["category"] == "delay", "usd_real"].sum()),
                "first_power_date": run.timing.first_power_date.date(),
                "full_power_date": None if pd.isna(run.timing.full_power_date) else run.timing.full_power_date.date(),
                "availability": run.delivered_mwh / run.served_mwh if run.served_mwh else np.nan,
                "load_served_share": run.delivered_mwh / run.load_mwh if run.load_mwh else np.nan,
                "mwh_delivered_total": run.delivered_mwh,
                "unserved_mwh_total": run.load_mwh - run.delivered_mwh,
                "peak_kw": eff.sites[site_id].load_mw_peak * 1000.0,
                "discount_rate": cfg.discount_rate,
                "notes": "; ".join(run.notes),
            }
        )
    return pd.DataFrame(rows)


def levelized_components(result: RunResult) -> pd.DataFrame:
    """Levelized $/MWh by chart group: NPV(group cost) / NPV(delivered MWh).

    The groups of one (site, path, scenario) sum exactly to its LDCoP.
    """
    df = result.ledger.to_frame()
    df["group"] = df["category"].map(_group_of)
    rows = []
    for key, g in df.groupby(ID_KEYS):
        cfg = result.inputs_by_scenario[key[2]].global_cfg
        mwh = g.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()
        denom = npv(mwh, cfg.discount_rate, cfg.base_year)
        for group in CATEGORY_GROUPS:
            sub = g[g["group"] == group]
            usd = npv(sub.groupby("year")["usd_real"].sum().to_dict(), cfg.discount_rate, cfg.base_year)
            rows.append((*key, group, usd / denom))
    return pd.DataFrame(rows, columns=[*ID_KEYS, "group", "usd_per_mwh"])


def levelized_categories(result: RunResult) -> pd.DataFrame:
    """Levelized $/MWh by ledger category: NPV(category cost) / NPV(delivered MWh).

    Same arithmetic as ``levelized_components`` but before grouping for the chart.
    """
    df = result.ledger.to_frame()
    rows = []
    for key, g in df.groupby(ID_KEYS):
        cfg = result.inputs_by_scenario[key[2]].global_cfg
        mwh = g.drop_duplicates("year").set_index("year")["mwh_delivered"].to_dict()
        denom = npv(mwh, cfg.discount_rate, cfg.base_year)
        for category, sub in g.groupby("category"):
            usd = npv(sub.groupby("year")["usd_real"].sum().to_dict(), cfg.discount_rate, cfg.base_year)
            rows.append((*key, category, usd / denom))
    return pd.DataFrame(rows, columns=[*ID_KEYS, "category", "usd_per_mwh"])


def check_components(summary: pd.DataFrame, components: pd.DataFrame, tol: float = 1e-6) -> None:
    """Levelized components must add up to each LDCoP (milestone M1 acceptance)."""
    total = components.groupby(ID_KEYS)["usd_per_mwh"].sum()
    ldcop = summary.set_index(ID_KEYS)["ldcop_usd_mwh"]
    diff = (total - ldcop.reindex(total.index)).abs()
    if (diff > tol).any():
        raise ReconciliationError(f"components != LDCoP for {diff.idxmax()}: off by {diff.max():.8f}")


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
    """Build the summary and run reconciliation, component, and sanity-bound checks."""
    summary = summary_table(result)
    reconcile(result.ledger, summary)
    check_components(summary, levelized_components(result))
    check_bounds(summary, result)
    return summary


# --------------------------------------------------------------------------
# Charts (static PNG, light surface). Colors are the validated categorical
# slots in fixed order; contrast on three slots is below 3:1, so the legend and
# the components table carry identity and values.
# --------------------------------------------------------------------------
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_RAMP = "#c3c2b7"
_GRID = "#e6e5e1"
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_GROUP_COLOR = dict(zip(CATEGORY_GROUPS, _SERIES))
_PATH_COLOR = {"grid": _SERIES[0], "btm": _SERIES[1], "hybrid": _SERIES[2]}
_DPI = 100


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_INK_2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def stack_chart(result: RunResult, scenario_id: str, path: Path) -> None:
    """$/MWh levelized, stacked by group; sites on x, grid and BTM side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    comp = levelized_components(result)
    comp = comp[comp["scenario_id"] == scenario_id]
    if comp.empty:
        raise ValueError(f"no results for scenario {scenario_id!r}")
    sites = list(dict.fromkeys(comp["site_id"]))
    bars = [(s, p) for s in sites for p in ("grid", "btm", "hybrid") if ((comp.site_id == s) & (comp.path_id == p)).any()]
    xs, gap = [], 0.0
    for i, (s, _) in enumerate(bars):
        if i and bars[i - 1][0] != s:
            gap += 1.0
        xs.append(i + gap)

    fig = plt.figure(figsize=(7.4, 4.6), dpi=_DPI, facecolor=_SURFACE)
    left, right, bottom, top = 0.10, 0.70, 0.17, 0.86
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    _style_axes(ax)
    ax.set_xlim(-0.8, max(xs) + 0.8)
    px_per_unit = fig.get_figwidth() * _DPI * (right - left) / (max(xs) + 1.6)
    width = 24.0 / px_per_unit  # marks are at most 24 px thick
    gap_pt = 2.0 * 72.0 / _DPI  # 2 px surface gap between stacked segments

    pos = np.zeros(len(bars))
    neg = np.zeros(len(bars))
    for group in CATEGORY_GROUPS:
        vals = np.array([
            comp[(comp.site_id == s) & (comp.path_id == p) & (comp.group == group)]["usd_per_mwh"].sum()
            for s, p in bars
        ])
        base = np.where(vals >= 0, pos, neg)
        ax.bar(xs, vals, width, bottom=base, color=_GROUP_COLOR[group], edgecolor=_SURFACE, linewidth=gap_pt)
        pos += np.where(vals >= 0, vals, 0.0)
        neg += np.where(vals < 0, vals, 0.0)
    totals = pos + neg
    for x, top_y, total in zip(xs, pos, totals):
        ax.text(x, top_y + 1.5, f"${total:,.1f}", ha="center", va="bottom", fontsize=9, color=_INK)

    ax.set_xticks(xs, [p.upper() for _, p in bars])
    for site in sites:
        site_x = [x for x, (s, _) in zip(xs, bars) if s == site]
        ax.text(np.mean(site_x), -0.13, site, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=9, color=_INK_2)
    ax.set_ylabel("Levelized delivered cost, real $/MWh", fontsize=9, color=_INK_2)
    ax.set_ylim(min(neg.min() * 1.2, 0), pos.max() * 1.15)

    fig.text(left, 0.94, "Delivered cost of power by component", fontsize=12, color=_INK,
             fontweight="bold", ha="left")
    fig.text(left, 0.895, f"Scenario: {scenario_id}. Components sum to the levelized cost (LDCoP).",
             fontsize=9, color=_INK_2, ha="left")
    handles = [Patch(facecolor=_GROUP_COLOR[g], label=g) for g in reversed(list(CATEGORY_GROUPS))]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(right + 0.02, 0.5),
                     frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(_INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_SURFACE)
    plt.close(fig)


def timing_chart(result: RunResult, scenario_id: str, path: Path) -> None:
    """MW online by quarter for each path against the load ramp, one panel per site."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = [r for r in result.paths if r.scenario_id == scenario_id]
    if not runs:
        raise ValueError(f"no results for scenario {scenario_id!r}")
    sites = list(dict.fromkeys(r.site_id for r in runs))
    fig, axes = plt.subplots(len(sites), 1, figsize=(7.4, 1.0 + 2.8 * len(sites)), dpi=_DPI,
                             facecolor=_SURFACE, squeeze=False, sharex=True)
    for ax, site_id in zip(axes[:, 0], sites):
        _style_axes(ax)
        site = result.inputs_by_scenario[scenario_id].sites[site_id]
        quarters = runs[0].timing.mw_online.index
        # the ramp is a wide pale band underneath so a path that tracks it stays visible
        ax.step(quarters, ramp_mw(site, quarters), where="post", color=_RAMP, linewidth=11, zorder=1)
        # paths often coincide, so draw them as nested stripes: widest at the bottom, narrowest on top
        stripe = {"hybrid": 7.0, "btm": 4.5, "grid": 2.0}
        for run in sorted((r for r in runs if r.site_id == site_id), key=lambda r: -stripe[r.path_id]):
            online = run.timing.mw_online
            ax.step(online.index, online.to_numpy(), where="post", color=_PATH_COLOR[run.path_id],
                    linewidth=stripe[run.path_id], zorder=2 + (8 - stripe[run.path_id]) / 10)
        ax.set_ylabel(f"{site_id}\nMW online", fontsize=9, color=_INK_2)
        ax.set_ylim(0, site.load_mw_peak * 1.1)
    # zoom to the ramp period: from the first quarter to two years past the last ramp or power event
    quarters = runs[0].timing.mw_online.index
    events = [r.timing.first_power_date for r in runs] + [
        pd.Timestamp(year=int(p[:4]), month=1, day=1)
        for s in {r.site_id for r in runs}
        for p, _ in result.inputs_by_scenario[scenario_id].sites[s].load_mw_ramp
    ]
    axes[-1, 0].set_xlim(quarters[0], max(events) + pd.DateOffset(years=2))
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    names = {"grid": "Grid online", "btm": "BTM online", "hybrid": "Hybrid online"}
    present = [p for p in ("grid", "btm", "hybrid") if any(r.path_id == p for r in runs)]
    handles = [Line2D([], [], color=_PATH_COLOR[p], linewidth=4, label=names[p]) for p in present]
    handles.append(Line2D([], [], color=_RAMP, linewidth=9, label="Load ramp"))
    top = 1.0 - 0.9 / fig.get_figheight()
    fig.text(0.10, top + 0.045, "MW online by quarter", fontsize=12, color=_INK, fontweight="bold", ha="left")
    fig.text(0.10, top - 0.005, f"Scenario: {scenario_id}. Online = lesser of the load ramp and the path's capacity.",
             fontsize=9, color=_INK_2, ha="left")
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(_INK_2)
    fig.subplots_adjust(left=0.10, right=0.96, top=top - 0.05, bottom=0.14 + 0.1 / len(sites), hspace=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_SURFACE)
    plt.close(fig)


def exposure_table(eff: Inputs, include_confirmed: bool = False) -> pd.DataFrame:
    """Upgrade items by site (spec §8, output 6).

    By default only unconfirmed items (probability below 1) are listed. The expected
    allocated cost is probability x allocation x P50.
    """
    rows = []
    for site_id, exposure in eff.upgrades.items():
        for item in exposure.items:
            if item.probability_triggered >= 1.0 and not include_confirmed:
                continue
            rows.append({
                "site_id": site_id, "item": item.name, "study_source": exposure.study_source,
                "probability_triggered": item.probability_triggered,
                "cost_usd_p10": item.cost_usd_p10_effective, "cost_usd_p50": item.cost_usd_p50,
                "cost_usd_p90": item.cost_usd_p90, "allocation_pct": item.allocation_pct,
                "expected_allocated_usd": item.probability_triggered * item.allocation_pct * item.cost_usd_p50,
                "ciac": item.ciac, "lead_time_months": item.lead_time_months,
                "on_critical_path": item.on_critical_path,
                "confirmed": item.probability_triggered >= 1.0,
            })
    return pd.DataFrame(rows)


def tornado_chart(table: pd.DataFrame, site_id: str, path: Path, top: int = 12) -> None:
    """Tornado for one site: LDCoP at each input's low and high value, grid and BTM side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    site = table[(table["site_id"] == site_id) & (table["swing"] > 1e-9)]
    paths = [p for p in ("grid", "btm", "hybrid") if (site["path_id"] == p).any()]
    if not paths:
        raise ValueError(f"no sensitivities with an effect for site {site_id!r}")
    rows = max(min(top, (site["path_id"] == p).sum()) for p in paths)
    fig, axes = plt.subplots(1, len(paths), figsize=(6.2 * len(paths) + 1.0, 1.9 + 0.42 * rows), dpi=_DPI,
                             facecolor=_SURFACE, squeeze=False)
    for ax, path_id in zip(axes[0], paths):
        _style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=_GRID, linewidth=1, linestyle="-")
        d = site[site["path_id"] == path_id].nlargest(top, "swing").iloc[::-1]  # biggest at the top
        base = float(d["ldcop_base"].iloc[0])
        y = np.arange(len(d))
        height = 0.6
        low_w = d["ldcop_low"].to_numpy() - base
        high_w = d["ldcop_high"].to_numpy() - base
        # where both ends move the same way, draw the longer bar first so the shorter one shows on top
        for yi, lo_w, hi_w in zip(y, low_w, high_w):
            for width_, color in sorted(((lo_w, _SERIES[0]), (hi_w, _SERIES[1])), key=lambda t: -abs(t[0])):
                ax.barh(yi, width_, height, left=base, color=color, edgecolor=_SURFACE, linewidth=1.44)
        ax.axvline(base, color=_INK_2, linewidth=1)
        ax.set_yticks(y, [f"{lbl}\n{rng}" for lbl, rng in zip(d["label"], d["range"])], fontsize=8)
        ax.tick_params(axis="y", colors=_INK_2)
        ax.yaxis.set_tick_params(pad=6)
        # label only the three biggest bars at their tips; the table view carries the rest
        for yi, lo, hi in list(zip(y, d["ldcop_low"], d["ldcop_high"]))[-3:]:
            for v in (lo, hi):
                if abs(v - base) < 1e-9:
                    continue
                ax.text(v + (0.25 if v >= base else -0.25), yi, f"{v:,.1f}", va="center",
                        ha="left" if v >= base else "right", fontsize=8, color=_INK)
        span = float(max(d[["ldcop_low", "ldcop_high"]].max().max() - base, base - d[["ldcop_low", "ldcop_high"]].min().min()))
        ax.set_xlim(base - span * 1.35, base + span * 1.35)
        ax.set_xlabel("Levelized cost, real $/MWh", fontsize=9, color=_INK_2)
        ax.set_title(f"{path_id.upper()}: base {base:,.1f} $/MWh", fontsize=10, color=_INK, loc="left", fontweight="bold")
    handles = [Patch(facecolor=_SERIES[0], label="Input at its low value"),
               Patch(facecolor=_SERIES[1], label="Input at its high value")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(_INK_2)
    scenario = site["scenario_id"].iloc[0]
    fig.suptitle(f"What moves the cost: {site_id}, scenario {scenario}", x=0.01, ha="left", fontsize=12,
                 color=_INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_SURFACE)
    plt.close(fig)
