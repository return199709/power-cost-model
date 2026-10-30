"""Pydantic models for every input and output type (spec §4, §6).

Conventions
- Units live in field names.
- Every ``*_pct`` field and every rate is a *fraction* (0.08 means 8%).
- Costs are positive, refunds and credits are negative.
- Timestamps are tz-naive local time, hour-beginning.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PathId = Literal["grid", "btm", "hybrid"]
Days = Literal["all", "weekday", "weekend"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sourced(_Model):
    """Source tagging for audit and staleness checks (spec §9)."""

    source: str | None = None
    as_of: date | None = None


# --------------------------------------------------------------------------
# Global config
# --------------------------------------------------------------------------
class GlobalConfig(Sourced):
    base_year: int = 2026
    discount_rate: float = Field(0.08, ge=0, lt=1)
    horizon_years: int = Field(15, ge=1)
    ops_start_year: int | None = None
    # Date the project clock starts; lead times run from here. Defaults to Jan 1 of base_year + 1.
    clock_start: date | None = None
    btm_salvage_pct_capex: float = Field(0.0, ge=0, le=1)  # residual value at horizon end
    inflation_pct: float = Field(0.025, ge=-0.1, lt=1)
    value_of_unserved_usd_mwh: float | None = Field(None, ge=0)
    allow_out_of_range: bool = False
    ldcop_bounds_usd_mwh: tuple[float, float] = (20.0, 250.0)
    stale_days: int = Field(365, ge=1)

    @property
    def first_ops_year(self) -> int:
        return self.ops_start_year if self.ops_start_year is not None else self.base_year + 1

    @property
    def clock_start_date(self) -> date:
        return self.clock_start if self.clock_start is not None else date(self.base_year + 1, 1, 1)

    @property
    def ops_years(self) -> list[int]:
        return list(range(self.first_ops_year, self.first_ops_year + self.horizon_years))

    @model_validator(mode="after")
    def _check_years(self) -> GlobalConfig:
        if self.first_ops_year <= self.base_year:
            raise ValueError("ops_start_year must be after base_year")
        return self


class HybridConfig(_Model):
    """BTM bridge to grid: gas serves the load until the grid is energized, then the grid
    serves it and the gas plant stays as backup (spec §12, open question 1).

    Capital recovery is sized so its present value equals the outlay, so the schedule alone
    does not change the levelized cost. What does: keeping the plant (default, capital is
    recovered over the horizon) or selling it at the handover (``sale_at_handover_pct``).
    """

    # Share of the firm gas transport kept after the grid takes over
    backup_transport_pct: float = Field(0.0, ge=0, le=1)
    # Share of plant fixed O&M that continues after the handover
    backup_fixed_om_pct: float = Field(1.0, ge=0, le=1)
    # Share of the plant, emissions-control and lateral capex recovered by selling them at the
    # handover. Capital is then recovered only up to the handover.
    sale_at_handover_pct: float = Field(0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _sold_plant_is_not_backup(self) -> HybridConfig:
        if self.sale_at_handover_pct > 0 and (self.backup_fixed_om_pct > 0 or self.backup_transport_pct > 0):
            raise ValueError("a plant sold at the handover cannot also keep backup O&M or gas transport")
        return self


# --------------------------------------------------------------------------
# 4.1 Site
# --------------------------------------------------------------------------
class Site(Sourced):
    site_id: str
    name: str
    region: str
    utility: str
    load_mw_peak: float = Field(gt=0)
    # [(period, cumulative MW)], period is "YYYY-Qn" or "YYYY"
    load_mw_ramp: list[tuple[str, float]]
    load_factor: float = Field(gt=0, le=1)
    load_profile_ref: str | None = None
    ambient_c_design: float
    elevation_m: float
    # BTM-only fields are optional until the BTM path lands (M1)
    gas_hub: str | None = None
    gas_pipeline: str | None = None
    lateral_km: float | None = Field(None, ge=0)
    tariff_docket_open: bool = False  # manual flag: utility has an open large-load tariff docket
    hybrid: HybridConfig | None = None  # set to add the hybrid path; needs grid_paths and btm_configs
    grid_paths: list[str] = Field(default_factory=list)  # tariff_ids
    btm_configs: list[str] = Field(default_factory=list)  # plant_config_ids

    @field_validator("site_id")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("ids must not contain '.' (used by scenario override paths)")
        return v

    @model_validator(mode="after")
    def _check_hybrid(self) -> Site:
        if self.hybrid is not None and not (self.grid_paths and self.btm_configs and self.gas_hub):
            raise ValueError("a hybrid path needs grid_paths, btm_configs and gas_hub")
        return self

    @model_validator(mode="after")
    def _check_ramp(self) -> Site:
        from dcop.load import parse_period  # local import avoids a cycle

        if not self.load_mw_ramp:
            raise ValueError("load_mw_ramp must have at least one step")
        starts = [parse_period(p) for p, _ in self.load_mw_ramp]
        mws = [mw for _, mw in self.load_mw_ramp]
        if starts != sorted(starts):
            raise ValueError("load_mw_ramp periods must be in chronological order")
        if any(b < a for a, b in zip(mws, mws[1:])) or any(m < 0 for m in mws):
            raise ValueError("load_mw_ramp is cumulative MW and cannot decrease")
        if max(mws) > self.load_mw_peak + 1e-9:
            raise ValueError("load_mw_ramp exceeds load_mw_peak")
        return self


# --------------------------------------------------------------------------
# 4.2 Grid: tariff
# --------------------------------------------------------------------------
class TOUBlock(_Model):
    season: str
    period: str
    months: list[int] = Field(min_length=1)
    hours: list[int] | None = None  # hour-beginning 0-23; None means all hours
    days: Days = "all"
    usd_per_mwh: float

    @field_validator("months")
    @classmethod
    def _months(cls, v: list[int]) -> list[int]:
        if any(m < 1 or m > 12 for m in v):
            raise ValueError("months must be 1-12")
        return v

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(h < 0 or h > 23 for h in v):
            raise ValueError("hours must be 0-23")
        return v


class DemandCharge(_Model):
    name: str
    season: str
    months: list[int] = Field(min_length=1)
    usd_per_kw_mo: float
    # 15min: peak 15-minute interval; ncp: peak hourly-integrated demand;
    # cp: demand in utility-designated coincident peak hours
    basis: Literal["15min", "ncp", "cp"] = "15min"
    hours: list[int] | None = None  # demand window, None means all hours
    days: Days = "all"
    ratchet_pct: float = Field(0.0, ge=0, le=1)
    ratchet_lookback_months: int = Field(11, ge=1)

    @field_validator("months")
    @classmethod
    def _months(cls, v: list[int]) -> list[int]:
        if any(m < 1 or m > 12 for m in v):
            raise ValueError("months must be 1-12")
        return v


class EnergyTier(_Model):
    """Inclining-block energy rate on monthly usage (e.g. residential)."""

    up_to_kwh_mo: float | None = Field(None, gt=0)  # cumulative upper bound, None means no limit
    usd_per_kwh: float


class Rider(_Model):
    name: str
    kind: Literal["usd_per_mwh", "usd_per_kwh", "usd_per_kw_mo", "pct_of_bill"]
    value: float
    months: list[int] | None = None  # None means every month
    # Ledger category: the power supply adjustment is energy cost, most others are fees
    category: Literal["energy", "riders_fees"] = "riders_fees"
    # Cap on this rider's charges per calendar year, prorated by billed months / 12
    annual_cap_usd: float | None = Field(None, gt=0)
    # Counts toward the trailing-demand minimum bill (e.g. the power supply adjustment)
    in_minimum_bill: bool = False


class Tariff(Sourced):
    tariff_id: str
    utility: str
    name: str
    effective_date: date
    energy_charges: list[TOUBlock] = Field(default_factory=list)
    energy_tiers: list[EnergyTier] = Field(default_factory=list)
    demand_charges: list[DemandCharge] = Field(default_factory=list)
    # When the power factor at peak is below this, billed kW = metered kW x min / pf
    power_factor_min: float | None = Field(None, gt=0, le=1)
    customer_charge_usd_mo: float = 0.0
    riders: list[Rider] = Field(default_factory=list)
    min_bill_pct: float | None = Field(None, ge=0, le=1)
    # Minimum bill = highest billed kW in the last N months x demand rates + in_minimum_bill riders
    min_bill_trailing_months: int | None = Field(None, ge=1)
    # Annual average monthly load factor needed to take this schedule (high-load-factor contracts)
    min_load_factor: float | None = Field(None, gt=0, le=1)
    contract_term_years: int | None = None
    collateral_requirements: str | None = None
    market_indexed: bool = False
    escalation_real_pct: float = Field(0.0, ge=-0.5, le=0.5)

    @field_validator("tariff_id")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("ids must not contain '.' (used by scenario override paths)")
        return v

    @model_validator(mode="after")
    def _check_energy(self) -> Tariff:
        if not self.energy_charges and not self.energy_tiers:
            raise ValueError("a tariff needs energy_charges or energy_tiers")
        bounds = [t.up_to_kwh_mo for t in self.energy_tiers]
        if self.energy_tiers and (
            bounds[-1] is not None
            or any(b is None for b in bounds[:-1])
            or bounds[:-1] != sorted(bounds[:-1])
        ):
            raise ValueError("energy_tiers need ascending bounds and a final tier with no limit")
        return self


# --------------------------------------------------------------------------
# 4.3 Grid: network upgrade exposure (consumed from M3)
# --------------------------------------------------------------------------
class UpgradeItem(_Model):
    name: str
    cost_usd_p10: float | None = Field(None, ge=0)  # default: mirror of P90 around P50, floored at 0
    cost_usd_p50: float = Field(ge=0)
    cost_usd_p90: float = Field(ge=0)
    probability_triggered: float = Field(ge=0, le=1)
    allocation_pct: float = Field(ge=0, le=1)
    ciac: bool
    refund_schedule: list[tuple[int, float]] | None = None  # [(year, pct refunded)]
    lead_time_months: int = Field(ge=0)
    on_critical_path: bool
    # Set by Monte Carlo draws: replaces the P50 in the cost lines
    cost_usd_sampled: float | None = Field(None, ge=0)

    @property
    def cost_usd_p10_effective(self) -> float:
        if self.cost_usd_p10 is not None:
            return self.cost_usd_p10
        return max(0.0, 2.0 * self.cost_usd_p50 - self.cost_usd_p90)

    @model_validator(mode="after")
    def _check(self) -> UpgradeItem:
        if self.cost_usd_p90 < self.cost_usd_p50:
            raise ValueError("cost_usd_p90 must be >= cost_usd_p50")
        if self.cost_usd_p10 is not None and self.cost_usd_p10 > self.cost_usd_p50:
            raise ValueError("cost_usd_p10 must be <= cost_usd_p50")
        if self.refund_schedule:
            if any(y < 0 or not 0 <= pct <= 1 for y, pct in self.refund_schedule):
                raise ValueError("refund_schedule entries are (years after payment >= 0, pct 0-1)")
            if sum(pct for _, pct in self.refund_schedule) > 1 + 1e-9:
                raise ValueError("refund_schedule cannot refund more than 100%")
        return self


class UpgradeExposure(Sourced):
    site_id: str
    # "feasibility study", "SIS", "analog", "rule-of-thumb"
    study_source: str
    items: list[UpgradeItem]


# --------------------------------------------------------------------------
# 4.4 BTM: gas supply (consumed from M1)
# --------------------------------------------------------------------------
class GasSupply(Sourced):
    hub: str  # matches Site.gas_hub
    henry_hub_curve_ref: str
    basis_curve_ref: str
    basis_volatility: float = Field(ge=0)  # lognormal sigma of the basis shock (Monte Carlo)
    basis_adder_usd_mmbtu: float = 0.0  # added to the basis curve; Monte Carlo and sensitivities set it
    henry_hub_multiplier: float = Field(1.0, gt=0)  # scales the Henry Hub curve
    transport_firm_usd_mmbtu: float = Field(ge=0)
    transport_commodity_usd_mmbtu: float = Field(ge=0)
    fuel_retention_pct: float = Field(ge=0, lt=1)
    lateral_capex_usd: float = Field(ge=0)
    lateral_lead_months: int = Field(ge=0)
    interruptible_risk_days_yr: float = Field(ge=0)

    @field_validator("hub")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("ids must not contain '.' (used by scenario override paths)")
        return v


# --------------------------------------------------------------------------
# 4.5 BTM: plant (consumed from M1)
# --------------------------------------------------------------------------
class PlantConfig(Sourced):
    plant_config_id: str
    technology: Literal["aeroderivative", "frame", "recip", "CCGT"]
    unit_mw_iso: float = Field(gt=0)
    n_units: int = Field(ge=1)
    redundancy: str = Field(pattern=r"^N(\+\d+)?$")  # "N", "N+1", "N+2"
    heat_rate_mmbtu_mwh_hhv_iso: float = Field(gt=0)
    heat_rate_part_load_curve_ref: str
    ambient_derate_pct_per_10c: float
    elevation_derate_pct_per_1000m: float
    degradation_pct_yr: float = Field(ge=0)
    capex_usd_kw: float = Field(ge=0)
    emissions_control_capex_usd_kw: float = Field(ge=0)
    fixed_om_usd_kw_yr: float = Field(ge=0)
    variable_om_usd_mwh: float = Field(ge=0)
    major_maintenance_usd_mwh: float = Field(ge=0)
    planned_outage_days_yr: float = Field(ge=0, le=365)
    forced_outage_rate: float = Field(ge=0, le=1)  # EFOR
    cod_lead_months: int = Field(ge=0)
    permit_lead_months: int = Field(ge=0)
    nox_limit_compliance: str
    co2_price_usd_t: float | None = None
    emission_factor_t_co2_mmbtu: float = 0.0531

    @field_validator("plant_config_id")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("ids must not contain '.' (used by scenario override paths)")
        return v

    @property
    def redundancy_k(self) -> int:
        return 0 if self.redundancy == "N" else int(self.redundancy.split("+")[1])


# --------------------------------------------------------------------------
# 4.6 Scenario
# --------------------------------------------------------------------------
class Scenario(_Model):
    scenario_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


# --------------------------------------------------------------------------
# 7 Monte Carlo and sensitivity settings (config/distributions.yaml)
# --------------------------------------------------------------------------
class DiscreteDist(_Model):
    """Discrete months of slip, added to a lead time."""

    values: list[int]
    probs: list[float]

    @model_validator(mode="after")
    def _check(self) -> DiscreteDist:
        if len(self.values) != len(self.probs) or not self.values:
            raise ValueError("values and probs must be the same non-zero length")
        if any(v < 0 for v in self.values) or any(p < 0 for p in self.probs):
            raise ValueError("values and probs must be non-negative")
        if abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError("probs must sum to 1")
        return self


class BasisDist(_Model):
    scale_usd_mmbtu: float = Field(ge=0)  # shock = scale x (lognormal factor with mean 1, minus 1)


class EforDist(_Model):
    concentration: float = Field(gt=0)  # beta with mean = the plant's EFOR and alpha + beta = this


class SensitivitySpec(_Model):
    """One-at-a-time input range for the tornado.

    ``path`` is an override path that may use ``{site}``, ``{tariff}``, ``{hub}``,
    ``{plant}`` and ``*`` for every element of a list. ``absolute`` sets the value,
    ``relative`` scales it by (1 + low) and (1 + high), ``delta`` adds low and high.
    """

    label: str
    path: str
    mode: Literal["absolute", "relative", "delta"]
    low: float
    high: float


class Distributions(Sourced):
    n_draws: int = Field(2000, ge=1)
    seed: int = 0
    basis: BasisDist | None = None
    upgrade_cost: bool = True  # sample item costs from P10/P50/P90 and trigger items by probability
    interconnection_slip_months: DiscreteDist | None = None  # added to critical-path upgrade lead times
    permit_slip_months: DiscreteDist | None = None  # added to the BTM permit lead time
    efor: EforDist | None = None
    sensitivities: list[SensitivitySpec] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 11 Assumptions register and data-file sources (config/assumptions.yaml, data/sources.yaml)
# --------------------------------------------------------------------------
class AssumptionEntry(_Model):
    """One row of the spec's assumptions register.

    ``path`` is an override-style path to the value, and may use ``{site}``, ``{tariff}``,
    ``{hub}`` and ``{plant}`` to expand once per site. A structural assumption, one that is
    built into the model rather than an input, has no path.
    """

    id: str
    name: str
    path: str | None = None
    kind: Literal["input", "structural"] = "input"
    description: str = ""  # for structural entries: what the model does
    source: str | None = None
    as_of: date | None = None
    owner: str = "TBD"
    confidence: Literal["low", "medium", "high"] | None = None


class DataSource(_Model):
    """Provenance of a data file referenced by a ``*_ref`` field."""

    path: str  # relative to the data directory, as written in the ref
    source: str | None = None
    as_of: date | None = None


# --------------------------------------------------------------------------
# 6 Ledger line
# --------------------------------------------------------------------------
class LedgerLine(_Model):
    site_id: str
    path_id: PathId
    scenario_id: str
    year: int  # calendar year
    category: str  # "energy", "demand", "riders_fees", "minimum_bill", "fuel", ...
    subcategory: str | None = None
    usd_real: float  # real USD in base_year dollars
    mwh_delivered: float = Field(ge=0)  # same for all lines in (site, path, scenario, year)
    is_capex: bool = False
    is_exposure: bool = False
    note: str | None = None
