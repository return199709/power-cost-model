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

Open M0 item: the bill replication test uses a synthetic tariff worked by hand.
The spec asks for a real utility's published bill reproduced within 1%.

## Run

```
pip install -e ".[dev]"
pytest
dcop run --scenario base          # all scenarios if --scenario is omitted
```

`dcop run` writes `summary`, `ledger` and `components` parquet files plus a
stack chart and a timing chart per scenario to `outputs/`.

Every input under `config/` and `data/processed/` is a placeholder, not real
data: the site, tariff, plant, gas supply and curves, upgrade items, and the
value of unserved energy. Results show that the model works, not what a real
site would cost.
