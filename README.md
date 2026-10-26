# power-cost-model

Delivered Cost of Power (DCoP) model: the all-in cost of delivering power to a site
load in $/MWh, for a grid path and a behind-the-meter (BTM) gas path. Design:
[DCOP_MODEL_SPEC.md](DCOP_MODEL_SPEC.md). Sourcing and audit:
[docs/SOURCING.md](docs/SOURCING.md). Security: [SECURITY.md](SECURITY.md).

## Status

All five milestones in the spec are built. The inputs are not real yet.

| Milestone | State |
|---|---|
| M0 schemas, loaders, finance, ledger, grid path | built |
| M1 BTM path, timing engine, stack chart | built |
| M2 all sites, both paths, deterministic scenarios | runner and the seven spec scenarios built; one placeholder site |
| M3 upgrade exposure, Monte Carlo, tornado | built |
| M4 assumptions dump, staleness flags, docs | built |

**Real data:** the Austin Energy FY2026 tariff, transcribed from the tariff PDF
(effective Nov 1, 2025). Its residential bill reproduces the City's published typical
bill of $117.07 for 860 kWh, from a search summary of the City's documents. The
commercial and large-load bills are worked by hand from the tariff.

**Placeholders:** the site, load profile, plant, gas supply and curves, upgrade items,
value of unserved energy, discount rate, and every Monte Carlo and tornado range. Results
show the model works, not what a real site would cost. `dcop audit` lists every one.

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
