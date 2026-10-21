# power-cost-model

Delivered Cost of Power (DCoP) model: the all-in cost of delivering power to a
site load in $/MWh, for a grid path and a behind-the-meter gas path. The full
design is in [DCOP_MODEL_SPEC.md](DCOP_MODEL_SPEC.md). Report security issues
as described in [SECURITY.md](SECURITY.md).

## Status

| Milestone | State |
|---|---|
| M0 schemas, loaders, finance, ledger, grid path with a hand-entered tariff | built, with one open item below |
| M1 BTM path, timing engine, stack chart | not started |
| M2 all sites, scenarios, charts | not started |
| M3 upgrade exposure, Monte Carlo, tornado | not started |
| M4 assumptions dump, staleness flags | not started |

Open M0 item: the bill replication test uses a synthetic tariff worked by hand.
The spec asks for a real utility's published bill reproduced within 1%.

## Run

```
pip install -e ".[dev]"
pytest
dcop run --scenario base          # writes outputs/summary.parquet and ledger.parquet
```

The site in `config/sites.yaml` and the tariff in `config/tariffs.yaml` are
placeholders, not real data.
