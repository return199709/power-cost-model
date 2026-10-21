# power-cost-model

Delivered Cost of Power (DCoP) model: the all-in cost of delivering power to a
site load in $/MWh, for a grid path and a behind-the-meter gas path. The full
design is in [DCOP_MODEL_SPEC.md](DCOP_MODEL_SPEC.md). Report security issues
as described in [SECURITY.md](SECURITY.md).

## Status

| Milestone | State |
|---|---|
| M0 schemas, loaders, finance, ledger, grid path with a hand-entered tariff | built, with one open item below |
| M1 BTM path, timing engine, stack chart | built, on placeholder inputs |
| M2 all sites, both paths, deterministic scenarios | scenario runner and all seven spec scenarios exist; real sites pending |
| M3 upgrade exposure, Monte Carlo, tornado | not started |
| M4 assumptions dump, staleness flags | not started |

M0 bill replication: the Austin Energy FY2026 residential bill reproduces the City
of Austin's published typical bill (860 kWh, $117.07). That figure came from a
search summary of the city's budget documents, because the city site was not
reachable from the build environment, so check it against the source. The
commercial bills are worked by hand from the tariff, since no published
commercial sample bill was found.

## Run

```
pip install -e ".[dev]"
pytest
dcop run --scenario base          # all scenarios if --scenario is omitted
```

Austin Energy FY2026 bills (inside city limits, secondary voltage):

```
dcop bill residential --kwh 860
dcop bill commercial --kwh 20000 --kw 50 [--pf 0.87]
dcop bill large --kwh 65000000 --kw 100000 --voltage transmission [--hlf]
```

The model's site takes the high-load-factor transmission schedule. That schedule
needs a multi-year contract with an exclusive-supplier clause (on-site backup
generation is allowed) and an 85% annual average monthly load factor. The summary
`notes` column flags a site that fails the load factor test.

Add `--no-psa-admin` to leave out the PSA administrative adjustment that took effect
Dec 1, 2025. The City's quoted typical bill predates it.

`dcop run` writes `summary`, `ledger` and `components` parquet files plus a
stack chart and a timing chart per scenario to `outputs/`.

Every input under `config/` and `data/processed/` is a placeholder, not real
data: the site, tariff, plant, gas supply and curves, upgrade items, and the
value of unserved energy. Results show that the model works, not what a real
site would cost.
