"""Monte Carlo sampling and P-values (spec §7).

Every draw is a set of overrides applied to the effective inputs, then the
deterministic engine runs on them. All draws are sampled up front from one seeded
generator, so results do not depend on the number of workers.

Distributions (``config/distributions.yaml``)
- Gas basis: an additive, persistent shock in $/MMBtu, ``scale x (F - 1)`` where F is
  lognormal with mean 1 and sigma ``basis_volatility``. Mean zero and right-skewed.
  A multiplier on the basis curve would not work because basis can be negative.
- Upgrade cost: piecewise-linear through the item's P10, P50 and P90, extended past
  P10 and P90 at the adjacent slope (floored at zero), so the sampled percentiles match
  the inputs. Each item is also triggered with its probability.
- Interconnection date and BTM permit date: discrete months of slip added to the
  critical-path upgrade lead times and to the permit lead time.
- EFOR: beta with mean equal to the plant's EFOR.

Not sampled in v0: Henry Hub (covered by the tornado), tariff changes, and the
correlations the spec lists for v1.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dcop.finance import levelize
from dcop.io import Inputs
from dcop.scenarios import PathRun, RunCache, apply_overrides, run_inputs
from dcop.schemas import DiscreteDist, Distributions

QUANTILES = (0.10, 0.50, 0.90)


# ---- samplers ----
def quantile_sample(u: np.ndarray, p10: float, p50: float, p90: float) -> np.ndarray:
    """Map uniform ``u`` through a piecewise-linear quantile function with these P10/P50/P90."""
    lo = max(0.0, p10 - 0.25 * (p50 - p10))  # the slope between P10 and P50 carried down to u = 0
    hi = p90 + 0.25 * (p90 - p50)
    return np.interp(u, [0.0, 0.1, 0.5, 0.9, 1.0], [lo, p10, p50, p90, hi])


def basis_shock(z: np.ndarray, sigma: float, scale: float) -> np.ndarray:
    """``scale x (F - 1)`` with F lognormal, E[F] = 1."""
    return scale * (np.exp(sigma * z - sigma**2 / 2.0) - 1.0)


def sample_efor(rng: np.random.Generator, efor: float, concentration: float, n: int) -> np.ndarray:
    if efor <= 0.0 or efor >= 1.0:
        return np.full(n, efor)
    return rng.beta(efor * concentration, (1.0 - efor) * concentration, size=n)


def sample_slip(rng: np.random.Generator, dist: DiscreteDist, n: int) -> np.ndarray:
    return rng.choice(np.asarray(dist.values), size=n, p=np.asarray(dist.probs))


@dataclass(frozen=True)
class Draws:
    overrides: list[dict[str, Any]]
    upgrade_exposure_usd: dict[str, np.ndarray]  # gross allocated upgrade cost per draw, by site


def sample_draws(eff: Inputs, dist: Distributions, n: int, seed: int) -> Draws:
    rng = np.random.default_rng(seed)
    overrides: list[dict[str, Any]] = [{} for _ in range(n)]
    exposure: dict[str, np.ndarray] = {}

    if dist.basis is not None:
        for hub in sorted({s.gas_hub for s in eff.sites.values() if s.btm_configs and s.gas_hub}):
            gas = eff.gas[hub]
            shock = basis_shock(rng.standard_normal(n), gas.basis_volatility, dist.basis.scale_usd_mmbtu)
            for k in range(n):
                overrides[k][f"gas.{hub}.basis_adder_usd_mmbtu"] = gas.basis_adder_usd_mmbtu + float(shock[k])

    for pid in sorted({p for s in eff.sites.values() for p in s.btm_configs}):
        plant = eff.plants[pid]
        if dist.efor is not None:
            efor = sample_efor(rng, plant.forced_outage_rate, dist.efor.concentration, n)
            for k in range(n):
                overrides[k][f"plants.{pid}.forced_outage_rate"] = float(efor[k])
        if dist.permit_slip_months is not None:
            slip = sample_slip(rng, dist.permit_slip_months, n)
            for k in range(n):
                overrides[k][f"plants.{pid}.permit_lead_months"] = plant.permit_lead_months + int(slip[k])

    for site_id in sorted(eff.upgrades):
        if site_id not in eff.sites or not eff.sites[site_id].grid_paths:
            continue
        items = eff.upgrades[site_id].items
        slip = sample_slip(rng, dist.interconnection_slip_months, n) if dist.interconnection_slip_months else None
        total = np.zeros(n)
        for i, item in enumerate(items):
            base = f"upgrades.{site_id}.items.{i}"
            if dist.upgrade_cost:
                cost = quantile_sample(rng.random(n), item.cost_usd_p10_effective, item.cost_usd_p50, item.cost_usd_p90)
                triggered = rng.random(n) < item.probability_triggered
                total += triggered * item.allocation_pct * cost
                for k in range(n):
                    overrides[k][f"{base}.cost_usd_sampled"] = float(cost[k])
                    overrides[k][f"{base}.probability_triggered"] = 1.0 if triggered[k] else 0.0
            else:
                total += item.probability_triggered * item.allocation_pct * item.cost_usd_p50
            if slip is not None and item.on_critical_path:
                for k in range(n):
                    overrides[k][f"{base}.lead_time_months"] = item.lead_time_months + int(slip[k])
        exposure[site_id] = total
    return Draws(overrides, exposure)


# ---- evaluation ----
def path_metrics(run: PathRun, eff: Inputs) -> dict[str, Any]:
    cfg = eff.global_cfg
    cost: dict[int, float] = {}
    mwh: dict[int, float] = {}
    delay = 0.0
    for ln in run.lines:
        cost[ln.year] = cost.get(ln.year, 0.0) + ln.usd_real
        mwh[ln.year] = ln.mwh_delivered
        if ln.category == "delay":
            delay += ln.usd_real
    try:
        ldcop = levelize(cost, mwh, cfg.discount_rate, cfg.base_year)
    except ValueError:
        ldcop = float("nan")
    return {
        "site_id": run.site_id,
        "path_id": run.path_id,
        "ldcop_usd_mwh": ldcop,
        "first_power_ordinal": run.timing.first_power_date.toordinal(),
        "delay_cost_usd": delay,
        "availability": run.delivered_mwh / run.served_mwh if run.served_mwh else float("nan"),
    }


def _failed_row(site_id: str, path_id: str) -> dict[str, Any]:
    nan = float("nan")
    return {"site_id": site_id, "path_id": path_id, "ldcop_usd_mwh": nan,
            "first_power_ordinal": nan, "delay_cost_usd": nan, "availability": nan}


def evaluate_draw(eff0: Inputs, overrides: dict[str, Any], cache: RunCache) -> list[dict[str, Any]]:
    """Per-path metrics for one draw.

    A path the engine rejects (for example power after the horizon) is NaN and the other
    paths of the same draw are kept. Inputs that fail validation make every path NaN.
    """
    try:
        eff = apply_overrides(eff0, overrides)
    except ValueError:
        return [_failed_row(s.site_id, p) for s in eff0.sites.values()
                for p, present in (("grid", s.grid_paths), ("btm", s.btm_configs)) if present]
    errors: list[tuple[str, str, ValueError]] = []
    rows = [path_metrics(run, eff) for run in run_inputs(eff, "mc", cache, errors)]
    return rows + [_failed_row(site_id, path_id) for site_id, path_id, _ in errors]


_WORKER: dict[str, Any] = {}


def _init_worker(eff0: Inputs) -> None:
    _WORKER["eff0"] = eff0
    _WORKER["cache"] = RunCache()


def _eval_chunk(chunk: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [evaluate_draw(_WORKER["eff0"], ov, _WORKER["cache"]) for ov in chunk]


# ---- run and summarize ----
@dataclass(frozen=True)
class McResult:
    scenario_id: str
    n_draws: int
    seed: int
    draws: pd.DataFrame  # one row per (draw, site, path)
    base: pd.DataFrame  # the deterministic run, one row per (site, path)
    summary: pd.DataFrame  # P-values per (site, path)
    upgrade_exposure: pd.DataFrame  # one row per site with upgrades


def default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def run_monte_carlo(
    inputs: Inputs,
    dist: Distributions,
    scenario_id: str = "base",
    n: int | None = None,
    seed: int | None = None,
    workers: int = 1,
) -> McResult:
    if scenario_id not in inputs.scenarios:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(inputs.scenarios)}")
    n = dist.n_draws if n is None else n
    seed = dist.seed if seed is None else seed
    eff0 = apply_overrides(inputs, inputs.scenarios[scenario_id].overrides)
    draws = sample_draws(eff0, dist, n, seed)

    base_rows = evaluate_draw(eff0, {}, RunCache())
    if workers <= 1 or n < 2 * workers:
        cache = RunCache()
        results = [evaluate_draw(eff0, ov, cache) for ov in draws.overrides]
    else:
        size = max(1, -(-n // (workers * 4)))
        chunks = [draws.overrides[i : i + size] for i in range(0, n, size)]
        # spawn, not fork: the parent has numpy and pandas threads running, and forking those can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(eff0,)) as pool:
            results = [row for part in pool.map(_eval_chunk, chunks) for row in part]

    frame = pd.DataFrame(
        [{"draw": k, **row} for k, rows in enumerate(results) for row in rows]
    )
    frame["first_power_date"] = pd.to_datetime(frame["first_power_ordinal"].map(
        lambda o: pd.Timestamp.fromordinal(int(o)) if o == o else pd.NaT))
    base = pd.DataFrame(base_rows)
    return McResult(
        scenario_id=scenario_id, n_draws=n, seed=seed,
        draws=frame.drop(columns="first_power_ordinal"),
        base=base,
        summary=_summarize(frame, base, eff0, scenario_id),
        upgrade_exposure=_exposure_summary(eff0, draws),
    )


def _quantile_dates(ordinals: pd.Series) -> list[pd.Timestamp]:
    vals = ordinals.dropna().to_numpy()
    if len(vals) == 0:
        return [pd.NaT, pd.NaT, pd.NaT]
    return [pd.Timestamp.fromordinal(int(round(np.quantile(vals, q)))) for q in QUANTILES]


def _summarize(frame: pd.DataFrame, base: pd.DataFrame, eff0: Inputs, scenario_id: str) -> pd.DataFrame:
    rows = []
    base_by = base.set_index(["site_id", "path_id"])
    for (site_id, path_id), g in frame.groupby(["site_id", "path_id"]):
        ok = g["ldcop_usd_mwh"].dropna()
        p10, p50, p90 = (float(np.quantile(ok, q)) if len(ok) else float("nan") for q in QUANTILES)
        d10, d50, d90 = _quantile_dates(g["first_power_ordinal"])
        b = base_by.loc[(site_id, path_id)]
        rows.append({
            "site_id": site_id, "path_id": path_id, "scenario_id": scenario_id,
            "n_draws": len(g), "n_failed": int(g["ldcop_usd_mwh"].isna().sum()),
            "ldcop_base": float(b["ldcop_usd_mwh"]), "ldcop_mean": float(ok.mean()) if len(ok) else float("nan"),
            "ldcop_p10": p10, "ldcop_p50": p50, "ldcop_p90": p90,
            "first_power_base": pd.Timestamp.fromordinal(int(b["first_power_ordinal"])),
            "first_power_p10": d10, "first_power_p50": d50, "first_power_p90": d90,
            "delay_cost_p50_usd": float(g["delay_cost_usd"].quantile(0.5)),
            "delay_cost_p90_usd": float(g["delay_cost_usd"].quantile(0.9)),
            "availability_p50": float(g["availability"].quantile(0.5)),
        })
    out = pd.DataFrame(rows)
    out["p_grid_cheaper"] = np.nan
    for site_id in out["site_id"].unique():
        pivot = frame[frame["site_id"] == site_id].pivot(index="draw", columns="path_id", values="ldcop_usd_mwh")
        if {"grid", "btm"} <= set(pivot.columns):
            both = pivot.dropna()
            out.loc[out["site_id"] == site_id, "p_grid_cheaper"] = float((both["grid"] < both["btm"]).mean())
    return out


def unconfirmed_share(items: list) -> float:
    """Share of the P50 allocated exposure that sits in items not yet confirmed (probability below 1)."""
    total = sum(i.allocation_pct * i.cost_usd_p50 for i in items)
    open_ = sum(i.allocation_pct * i.cost_usd_p50 for i in items if i.probability_triggered < 1.0)
    return open_ / total if total else 0.0


def _exposure_summary(eff0: Inputs, draws: Draws) -> pd.DataFrame:
    rows = []
    for site_id, total in draws.upgrade_exposure_usd.items():
        peak_kw = eff0.sites[site_id].load_mw_peak * 1000.0
        p50, p90 = (float(np.quantile(total, q)) for q in (0.5, 0.9))
        rows.append({
            "site_id": site_id,
            "upgrade_exposure_p50_usd": p50, "upgrade_exposure_p90_usd": p90,
            "upgrade_exposure_p50_usd_per_kw": p50 / peak_kw, "upgrade_exposure_p90_usd_per_kw": p90 / peak_kw,
            "upgrade_unconfirmed_share": unconfirmed_share(eff0.upgrades[site_id].items),
            "tariff_redesign_risk": eff0.sites[site_id].tariff_docket_open,
        })
    return pd.DataFrame(rows)
