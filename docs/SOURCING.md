# Sourcing and audit

The aim: a reviewer can take any number in an output and trace it to a source.

## Tracing a number

1. Run `dcop run` (or `dcop audit`). It writes `outputs/assumptions.md`.
2. Find the number's cost line in "Where each cost line comes from". Each row gives the
   levelized $/MWh of one ledger category and the inputs that drive it. The levelized lines
   of a path add up to its LDCoP.
3. Look those inputs up under "Inputs used". Each object shows its source, its date, and
   its flags. The paths are the same ones a scenario override takes.
4. Check "Assumptions register" for the reasoning behind the input, its owner and its
   confidence, and "Data files" for the exact file, with a SHA-256 hash.
5. "Scenario overrides" shows what each scenario changed from the base value.

The file also records the run date, model version, git commit and a hash of every config
file, so a run can be reproduced.

## Source tags

Every input object carries `source` and `as_of`: sites, tariffs, plants, gas supplies,
upgrade studies, the global settings and the Monte Carlo settings. Each data file has an
entry in `data/sources.yaml`. Finer-grained assumptions go in `config/assumptions.yaml`,
which is the spec's section 11 register.

An item is flagged when it is:

| Flag | Meaning |
|---|---|
| placeholder | the source contains "placeholder" or "synthetic" |
| stale | `as_of` is more than `stale_days` before the run date (365 by default) |
| no source, no as_of | the tag is missing |
| unresolved | a register entry points at a field that does not exist |
| missing file | a referenced data file is not on disk |
| owner TBD, low confidence | register entries still waiting on a person |

The first five are problems for anything you present. `dcop run --strict`, `dcop mc --strict`
and `dcop audit --strict` exit with status 2 when any is present. The last two are
reminders and never fail a run.

## Rules for adding inputs

- Give every new input a `source` and an `as_of`. Name the document, the page, and who
  supplied it. Use "PLACEHOLDER:" at the start of the source for anything you made up.
- Add a data file to `data/sources.yaml` when you add it.
- Add an assumption to `config/assumptions.yaml` when it needs an owner or a reason. A path
  may use `{site}`, `{tariff}`, `{hub}` and `{plant}`.
- A new ledger category needs an entry in `COST_LINE_INPUTS` in `dcop/assumptions.py`. A test
  fails until it has one.

## Refreshing inputs

- **Tariffs:** when a new tariff takes effect, add it with its effective date and page
  references, point the site at it, and rerun the bill checks in `tests/test_austin.py` and
  `tests/test_austin_large.py`. Austin Energy's FY2027 tariff is due Nov 1, 2026.
- **Curves and profiles:** replace the file, update its `sources.yaml` entry, and the hash in
  the next `assumptions.md` shows the change.
- **Stale inputs:** the audit lists them. Confirm each is still current and update `as_of`, or
  replace it.
