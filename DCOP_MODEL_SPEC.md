# Delivered Cost of Power (DCoP) Model — Build Spec

> Scope: a site-by-site, path-by-path model of the all-in cost of delivering power to the load, in $/MWh and $/kW-yr, with timing and risk. Two paths per site: **Grid** (utility interconnect) and **BTM** (behind-the-meter gas generation). Owner: _TBD_. Status: v0 draft.

---

## 1. Purpose

Answer, for every candidate site and each path:

1. What is the levelized delivered cost of power ($/MWh) over the planning horizon, and how is it composed?
2. What is the exposure to costs we don't control yet (network upgrade allocation, gas basis blowout, tariff redesign)?
3. When does firm power actually arrive (MW online by quarter), and what does delay cost?
4. Which inputs drive the answer (sensitivities / tornado), and where is our confidence low?

Non-goals for v0: dispatch optimization, hourly market simulation, carbon accounting beyond a price adder, financing structure optimization.

---

## 2. Conventions

| Item | Convention |
|---|---|
| Currency | USD, **real** dollars in `BASE_YEAR` (default 2026). Nominal inputs are deflated on load. |
| Discount rate | Real WACC, single `discount_rate` in config (default 8%). Levelization uses NPV of cost / NPV of delivered MWh. |
| Horizon | `horizon_years` (default 15), annual steps; monthly granularity for demand charges and gas; hourly only where tariff/TOU requires. |
| Energy units | MW, MWh (AC at the meter). Gas in MMBtu, HHV. Heat rate in MMBtu/MWh HHV. |
| Power vs delivered | "Delivered" = MWh actually available to the load after availability/curtailment/derates. All $/MWh are per **delivered** MWh. |
| Sign | Costs positive. Refunds / credits negative. |
| IDs | `site_id`, `path_id ∈ {grid, btm}`, `scenario_id`. Every output row carries all three. |

---

## 3. Architecture

```
inputs (yaml/csv/parquet)
   │
   ├─ load profile  ──────────────┐
   ├─ site facts  ────────────────┤
   ├─ grid: tariffs, upgrades ────┼──►  path engines  ──►  finance (levelize) ──►  outputs
   ├─ btm: gas, plant, avail. ────┤        │                                       (tables,
   └─ scenarios / distributions ──┘        └── timing engine (MW online by period)  charts,
                                                                                   sensitivities)
```

Each path engine produces the same **cost ledger** shape (§6), so finance and outputs are path-agnostic.

### Repo layout

```
dcop/
├── README.md                  # this file → trimmed
├── pyproject.toml
├── config/
│   ├── global.yaml            # base_year, discount_rate, horizon, escalation
│   ├── sites.yaml
│   ├── scenarios.yaml
│   └── distributions.yaml     # for Monte Carlo
├── data/
│   ├── raw/                   # tariff sheets, interconnection study PDFs→csv, EIA/hub curves
│   └── processed/
├── dcop/
│   ├── schemas.py             # pydantic models for every input/output type
│   ├── io.py                  # loaders, unit normalization, real/nominal
│   ├── load.py                # load profile generation / scaling / ramp
│   ├── timing.py              # MW online by period, delay cost
│   ├── grid/
│   │   ├── tariff.py          # bill calculator
│   │   ├── upgrades.py        # network upgrade exposure
│   │   └── market.py          # LMP/capacity-indexed components (if applicable)
│   ├── btm/
│   │   ├── gas.py             # commodity + basis + transport → $/MMBtu delivered
│   │   ├── plant.py           # capex, O&M, heat rate, derates
│   │   └── availability.py    # EAF, N+k, firming
│   ├── finance.py             # annuity, NPV, levelization, escalation
│   ├── ledger.py              # cost ledger dataclass + aggregation
│   ├── scenarios.py           # deterministic scenario runner
│   ├── montecarlo.py          # sampling + P-values
│   └── reports.py             # summary tables, tornado, stack charts
├── tests/
│   ├── test_tariff.py         # replicate a known utility bill to the dollar
│   ├── test_gas.py
│   ├── test_finance.py        # annuity/NPV against hand calc
│   └── fixtures/
└── notebooks/                 # exploration only; nothing production lives here
```

---

## 4. Input schemas

Sketched as pydantic; keep units in field names.

### 4.1 Site

```python
class Site(BaseModel):
    site_id: str
    name: str
    region: str                  # ISO/RTO or utility territory
    utility: str
    load_mw_peak: float
    load_mw_ramp: list[tuple[str, float]]   # [(period, cumulative MW), ...]
    load_factor: float           # 0–1; or point to a profile file
    load_profile_ref: str | None
    ambient_c_design: float      # for turbine derate
    elevation_m: float
    gas_hub: str                 # pricing point for basis
    gas_pipeline: str
    lateral_km: float
    grid_paths: list[str]        # tariff_ids applicable
    btm_configs: list[str]       # plant_config_ids applicable
```

### 4.2 Grid — tariff

```python
class Tariff(BaseModel):
    tariff_id: str
    utility: str
    name: str                    # e.g. "LGS-TOU", "Large Load Rider X"
    effective_date: date
    energy_charges: list[TOUBlock]       # $/MWh by season/period
    demand_charges: list[DemandCharge]   # $/kW-mo by season; ratchet_pct; billing basis (15-min, coincident, NCP)
    customer_charge_usd_mo: float
    riders: list[Rider]                  # fuel adj, transmission, DSM, etc. $/MWh or $/kW-mo or % of bill
    min_bill_pct: float | None           # take-or-pay / minimum demand
    contract_term_years: int | None
    collateral_requirements: str | None
    market_indexed: bool                 # if True, energy = LMP + adder (see market.py)
    escalation_real_pct: float           # expected real rate growth
```

### 4.3 Grid — network upgrade exposure

```python
class UpgradeExposure(BaseModel):
    site_id: str
    source: str                          # "feasibility study", "SIS", "analog", "rule-of-thumb"
    items: list[UpgradeItem]

class UpgradeItem(BaseModel):
    name: str                            # "Substation 345/138", "Line rebuild 12 mi"
    cost_usd_p50: float
    cost_usd_p90: float
    probability_triggered: float         # 0–1 for items not yet confirmed
    allocation_pct: float                # share borne by us (cluster studies)
    ciac: bool                           # contribution in aid of construction (non-refundable)?
    refund_schedule: list[tuple[int, float]] | None   # [(year, pct refunded)]
    lead_time_months: int
    on_critical_path: bool
```

### 4.4 BTM — gas supply

```python
class GasSupply(BaseModel):
    hub: str
    henry_hub_curve_ref: str             # forward curve file
    basis_curve_ref: str                 # $/MMBtu vs HH by month
    basis_volatility: float              # for MC
    transport_firm_usd_mmbtu: float      # FT demand charge
    transport_commodity_usd_mmbtu: float
    fuel_retention_pct: float            # pipeline fuel/lost-and-unaccounted
    lateral_capex_usd: float
    lateral_lead_months: int
    interruptible_risk_days_yr: float    # if IT instead of FT
```

### 4.5 BTM — plant

```python
class PlantConfig(BaseModel):
    plant_config_id: str
    technology: str                      # "aeroderivative", "frame", "recip", "CCGT"
    unit_mw_iso: float
    n_units: int
    redundancy: str                      # "N", "N+1", "N+2"
    heat_rate_mmbtu_mwh_hhv_iso: float   # at ISO, full load
    heat_rate_part_load_curve_ref: str
    ambient_derate_pct_per_10c: float
    elevation_derate_pct_per_1000m: float
    degradation_pct_yr: float
    capex_usd_kw: float                  # installed, incl. BOP, grid-independence gear
    emissions_control_capex_usd_kw: float
    fixed_om_usd_kw_yr: float
    variable_om_usd_mwh: float
    major_maintenance_usd_mwh: float     # LTSA
    planned_outage_days_yr: float
    forced_outage_rate: float            # EFOR
    cod_lead_months: int
    permit_lead_months: int
    nox_limit_compliance: str
    co2_price_usd_t: float | None
    emission_factor_t_co2_mmbtu: float   # 0.0531 for gas
```

### 4.6 Scenario

```python
class Scenario(BaseModel):
    scenario_id: str
    overrides: dict[str, Any]            # dotted-path overrides, e.g. "gas.basis_curve_ref"
    description: str
```

---

## 5. Path methodology

### 5.1 Common: load and delivered energy

```
load_mwh[t]        = profile(site, t) scaled to ramp schedule
delivered_mwh[t]   = load_mwh[t] × availability[t] × (1 − curtailment[t])
unserved_mwh[t]    = load_mwh[t] − delivered_mwh[t]
```

`unserved_mwh` carries a `value_of_unserved_usd_mwh` (config; proxy for lost revenue / backup cost) so paths with lower availability or later energization are penalized consistently.

### 5.2 Grid path

Annual cost ledger lines:

| Line | Formula | Module |
|---|---|---|
| Energy | Σ_h load_h × rate(season, TOU period) or Σ_h load_h × (LMP_h + adder) | `tariff.py` / `market.py` |
| Demand | Σ_m max(billing_demand_m, ratchet_m) × $/kW-mo | `tariff.py` |
| Riders & fees | per rider: $/MWh × MWh, $/kW-mo × kW, or % × subtotal | `tariff.py` |
| Minimum bill | max(0, min_bill − computed bill) | `tariff.py` |
| Network upgrades | Σ_i P(trigger_i) × alloc_i × cost_i → capex at year(lead_i); refunds as negative lines per schedule | `upgrades.py` |
| Interconnection facilities | our-side substation, metering, line: capex → annuity | `upgrades.py` |
| Curtailment / NWA | expected curtailed MWh × value_of_unserved | `market.py` |
| Delay | unserved MWh before energization × value_of_unserved | `timing.py` |

Exposure metrics (reported separately, not just in the mean):
- `upgrade_exposure_p50`, `upgrade_exposure_p90` ($, and $/kW of peak load)
- `upgrade_unconfirmed_share` = share of P50 that is still `probability_triggered < 1`
- `tariff_redesign_risk` = flagged if utility has open large-load tariff docket (manual input)

Billing demand: implement 15-min max, NCP, and CP bases; ratchet as `max(current, pct × trailing-11-month max)`.

### 5.3 BTM path

Delivered gas price:

```
gas_delivered[m] = (HH[m] + basis[m]) / (1 − fuel_retention) + transport_commodity
transport_fixed[yr] = FT demand charge × contracted MDQ × 365
```

Plant output and fuel:

```
unit_mw_site     = unit_mw_iso × (1 − ambient_derate) × (1 − elevation_derate)
firm_mw          = (n_units − k) × unit_mw_site            # k from redundancy string
heat_rate[yr]    = HR_iso × part_load_factor × (1 + degradation × yr) × ambient_hr_adj
fuel_mmbtu[t]    = delivered_mwh[t] × heat_rate[t]
```

Availability:

```
EAF              = (1 − planned_outage_days/365) × (1 − EFOR)
availability[t]  = P(≥ required units available | n_units, EAF)   # binomial; required = ceil(load/unit_mw_site)
```
v0 uses binomial independence; v1 can add common-mode (gas supply interruption → `interruptible_risk_days_yr`).

Ledger lines:

| Line | Formula |
|---|---|
| Fuel commodity | Σ fuel_mmbtu × gas_delivered |
| Transport fixed | FT demand charge, annual |
| Lateral | capex → annuity |
| Plant capex | capex_usd_kw × n_units × unit_mw_iso × 1000 → annuity (note: redundancy inflates $/delivered MWh) |
| Emissions controls | capex → annuity |
| Fixed O&M | $/kW-yr × installed kW |
| Variable O&M + LTSA | $/MWh × generated MWh |
| Carbon | fuel_mmbtu × EF × co2_price (if set) |
| Delay | unserved before COD × value_of_unserved |
| Residual | − salvage at horizon end (config, default 0) |

### 5.4 Timing engine

For each path: `mw_online[period]` = min(load ramp, path capacity online), where capacity online is gated by max(lead times on critical path). Outputs `first_power_date`, `full_power_date`, and `delay_cost_usd` (already in ledger). This is the piece that most often flips Grid vs BTM, so keep it explicit and separately reported.

---

## 6. Cost ledger (shared output shape)

```python
class LedgerLine(BaseModel):
    site_id: str
    path_id: Literal["grid", "btm"]
    scenario_id: str
    year: int
    category: str          # "energy", "demand", "upgrade_capex", "fuel", ...
    subcategory: str | None
    usd_real: float
    mwh_delivered: float   # same for all lines in a (site, path, scenario, year)
    is_capex: bool
    is_exposure: bool      # True for probabilistic / unconfirmed items
    note: str | None
```

Aggregation (`ledger.py`): group by any subset of keys → `$`, `$/MWh`, `$/kW-yr`, share of total.

Levelized delivered cost:

```
LDCoP = Σ_t cost_t / (1+r)^t  ÷  Σ_t delivered_mwh_t / (1+r)^t
```

Also report **undiscounted year-1 and year-5 $/MWh**, since LDCoP hides ramp effects.

---

## 7. Scenarios and uncertainty

- **Deterministic scenarios** (`scenarios.yaml`): `base`, `high_gas`, `low_gas`, `upgrade_p90`, `tariff_redesign`, `delayed_interconnect`, `btm_permit_slip`. Each is a dict of overrides; runner applies them and re-runs.
- **Monte Carlo** (`montecarlo.py`): sample from `distributions.yaml` (basis: lognormal; upgrade cost: triangular P10/P50/P90; interconnection date: discrete; EFOR: beta). N=2,000 default. Report P10/P50/P90 of LDCoP and of first_power_date.
- **Sensitivity**: one-at-a-time ±1σ on top ~12 inputs → tornado per site/path.
- Correlations to encode in v1: HH ↔ regional LMP; interconnection delay ↔ upgrade cost.

---

## 8. Outputs

1. `summary.parquet` — one row per (site, path, scenario): LDCoP, year-1 $/MWh, year-5 $/MWh, capex total, exposure P50/P90, first/full power date, availability.
2. `ledger.parquet` — full ledger.
3. `stack_chart` — $/MWh stacked by category, sites on x-axis, grid/btm side by side.
4. `tornado` — per site/path.
5. `timing_chart` — MW online vs quarter, both paths vs load ramp.
6. `exposure_table` — unconfirmed upgrade items with probability, P50/P90, critical-path flag.
7. `assumptions.md` — auto-generated dump of every input and its source, per run (for audit).

---

## 9. Validation

- **Tariff bill replication**: for ≥1 utility per region, reproduce a published or sample bill to within 1% (`test_tariff.py`).
- **Unit tests**: annuity/NPV against spreadsheet; heat rate derates against vendor curve; binomial availability against closed form.
- **Sanity bounds**: fail the run if any path lands outside `[20, 250] $/MWh` without an explicit `allow_out_of_range` flag.
- **Reconciliation**: Σ ledger lines == summary total, per (site, path, scenario).
- **Source tagging**: every input carries `source` and `as_of`; reports flag inputs older than `stale_days`.

---

## 10. Build plan

| Milestone | Deliverable | Done when |
|---|---|---|
| M0 | Schemas, loaders, finance primitives, ledger, 1 site × grid path with a hand-entered tariff | Bill replication test passes |
| M1 | BTM path for same site; timing engine; stack chart | Grid vs BTM $/MWh for one site, components reconcile |
| M2 | All candidate sites, both paths, deterministic scenarios | `summary.parquet` + charts for all sites |
| M3 | Upgrade exposure module with probabilistic items; MC; tornado | P-values and exposure table |
| M4 | Assumptions dump, staleness flags, docs | A reviewer can trace any number to a source |

Suggested stack: Python 3.12, pydantic v2, pandas/polars, numpy, pyyaml, plotly or matplotlib, pytest. Keep it a plain package + CLI (`dcop run --scenario base`) before any dashboard.

---

## 11. Assumptions register (fill as you go)

| # | Assumption | Value | Source | Owner | Confidence |
|---|---|---|---|---|---|
| A1 | Real discount rate | 8% | | | |
| A2 | Value of unserved energy | $/MWh | | | |
| A3 | Load factor / profile shape | | | | |
| A4 | Gas pricing hub per site | | | | |
| A5 | BTM excess energy has no sale value | | | | |
| A6 | Carbon price | none in base | | | |

## 12. Open questions

1. Is a **hybrid path** (BTM bridge → grid at energization, BTM retained as backup) in scope? If yes, model as a third `path_id` that stitches the two ledgers with a transition date.
2. Does the grid path assume a **standard tariff or a negotiated large-load contract**? The latter needs a contract-term object (take-or-pay, exit fees).
3. Which sites have **real interconnection studies** vs analogs? Drives `UpgradeExposure.source` and how wide the P10–P90 is.
4. Is **firm transport (FT)** actually available on the relevant pipelines, or is IT the realistic case? Changes availability materially.
5. Horizon: 15 vs 20 years changes how much BTM capex is amortized vs how much grid tariff escalation bites.
6. Do we need **hourly** resolution anywhere beyond TOU tariffs (e.g., market-indexed sites in ERCOT)?
