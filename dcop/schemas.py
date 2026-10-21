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

PathId = Literal["grid", "btm"]
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
class GlobalConfig(_Model):
    base_year: int = 2026
    discount_rate: float = Field(0.08, ge=0, lt=1)
    horizon_years: int = Field(15, ge=1)
    ops_start_year: int | None = None
    inflation_pct: float = Field(0.025, ge=-0.1, lt=1)
    value_of_unserved_usd_mwh: float | None = Field(None, ge=0)
    allow_out_of_range: bool = False
    ldcop_bounds_usd_mwh: tuple[float, float] = (20.0, 250.0)
    stale_days: int = Field(365, ge=1)

    @property
    def first_ops_year(self) -> int:
        return self.ops_start_year if self.ops_start_year is not None else self.base_year + 1

    @property
    def ops_years(self) -> list[int]:
        return list(range(self.first_ops_year, self.first_ops_year + self.horizon_years))

    @model_validator(mode="after")
    def _check_years(self) -> GlobalConfig:
        if self.first_ops_year <= self.base_year:
            raise ValueError("ops_start_year must be after base_year")
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
    grid_paths: list[str] = Field(default_factory=list)  # tariff_ids
    btm_configs: list[str] = Field(default_factory=list)  # plant_config_ids

    @field_validator("site_id")
    @classmethod
    def _no_dots(cls, v: str) -> str:
        if "." in v:
            raise ValueError("ids must not contain '.' (used by scenario override paths)")
        return v

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


class Rider(_Model):
    name: str
    kind: Literal["usd_per_mwh", "usd_per_kw_mo", "pct_of_bill"]
    value: float
    months: list[int] | None = None  # None means every month


class Tariff(Sourced):
    tariff_id: str
    utility: str
    name: str
    effective_date: date
    energy_charges: list[TOUBlock]
    demand_charges: list[DemandCharge] = Field(default_factory=list)
    customer_charge_usd_mo: float = 0.0
    riders: list[Rider] = Field(default_factory=list)
    min_bill_pct: float | None = Field(None, ge=0, le=1)
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


# --------------------------------------------------------------------------
# 4.3 Grid: network upgrade exposure (consumed from M3)
# --------------------------------------------------------------------------
class UpgradeItem(_Model):
    name: str
    cost_usd_p50: float = Field(ge=0)
    cost_usd_p90: float = Field(ge=0)
    probability_triggered: float = Field(ge=0, le=1)
    allocation_pct: float = Field(ge=0, le=1)
    ciac: bool
    refund_schedule: list[tuple[int, float]] | None = None  # [(year, pct refunded)]
    lead_time_months: int = Field(ge=0)
    on_critical_path: bool

    @model_validator(mode="after")
    def _p90(self) -> UpgradeItem:
        if self.cost_usd_p90 < self.cost_usd_p50:
            raise ValueError("cost_usd_p90 must be >= cost_usd_p50")
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
    hub: str
    henry_hub_curve_ref: str
    basis_curve_ref: str
    basis_volatility: float = Field(ge=0)
    transport_firm_usd_mmbtu: float = Field(ge=0)
    transport_commodity_usd_mmbtu: float = Field(ge=0)
    fuel_retention_pct: float = Field(ge=0, lt=1)
    lateral_capex_usd: float = Field(ge=0)
    lateral_lead_months: int = Field(ge=0)
    interruptible_risk_days_yr: float = Field(ge=0)


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
