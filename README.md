# power-cost-model

Delivered Cost of Power (DCoP) model: the all-in cost of delivering power to a site
load in $/MWh, for a grid path and a behind-the-meter (BTM) gas path. Design:
[DCOP_MODEL_SPEC.md](DCOP_MODEL_SPEC.md). Sourcing and audit:
[docs/SOURCING.md](docs/SOURCING.md). Security: [SECURITY.md](SECURITY.md).

## Status

All five milestones in the spec are built, and so is the hybrid path from the spec's open
questions. The inputs are not real yet.

| Milestone | State |
|---|---|
| M0 schemas, loaders, finance, ledger, grid path | built |
| M1 BTM path, timing engine, stack chart | built |
| M2 all sites, all paths, deterministic scenarios | runner and the seven spec scenarios built; one placeholder site |
| M3 upgrade exposure, Monte Carlo, tornado | built |
| M4 assumptions dump, staleness flags, docs | built |

**Real data:** the Austin Energy FY2026 tariff, transcribed from the tariff PDF
(effective Nov 1, 2025). Its residential bill reproduces the City's published typical
bill of $117.07 for 860 kWh, from a search summary of the City's documents. The
commercial and large-load bills are worked by hand from the tariff.

**Placeholders:** the site, load profile, plant, gas supply and curves, upgrade items,
value of unserved energy, discount rate, and every Monte Carlo and tornado range. Results
show the model works, not what a real site would cost. `dcop audit` lists every one.

## Paths

Every site gets a **grid** path and a **BTM** (gas on site) path from its `grid_paths` and
`btm_configs`. A site with a `hybrid` setting also gets a **hybrid** path: gas serves the load
until the grid is energized, then the grid serves it and the gas plant stays as backup. The
setting says how much backup gas transport and fixed O&M continue, and what share of the
plant's capex is recovered by selling it at the handover. Capital recovery is sized to match
the outlay's present value, so only keeping versus selling the plant moves the cost, not
the payment schedule. If the grid is energized no later than gas could be, no bridge plant is
built. The Austin Energy high-load-factor contract is exclusive except for on-site backup
generation, which is why the hybrid is the form that fits it.

## Run

```
pip install -e ".[dev]"
pytest
dcop run [--scenario base]        # costs, charts, assumptions.md; all scenarios if omitted
dcop mc [--draws 2000]            # Monte Carlo P-values, upgrade exposure, tornado
dcop audit [--strict]             # assumptions.md only; --strict fails on placeholders
dcop bill residential --kwh 860   # itemized Austin Energy bill; also commercial and large
```

Outputs go to `outputs/` (git-ignored): `summary`, `ledger`, `components` and Monte Carlo
parquet files, stack, timing and tornado charts, and `assumptions.md`.

## Layout

```
config/       inputs: global, sites, tariffs, plants, gas, upgrades, scenarios,
              distributions, and the assumptions register
data/         curves and profiles, with sources.yaml recording where each came from
dcop/         the package: schemas, io, load, timing, finance, ledger, scenarios,
              montecarlo, reports, assumptions; grid/ and btm/ path engines
tests/        hand-worked expectations; run them before committing
docs/         SOURCING.md
```

## Conventions

Real dollars in the base year, real WACC, annual steps, all rates as fractions, every
output row carrying `site_id`, `path_id` and `scenario_id`. Details are in spec section 2.
