"""Command line: ``dcop run --scenario base``."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from dcop.io import load_inputs
from dcop.reports import levelized_components, stack_chart, timing_chart, validated_summary
from dcop.scenarios import run_all


def _audit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strict", action="store_true",
                   help="exit with status 2 if any input is a placeholder, stale, untagged or broken")
    p.add_argument("--today", type=date.fromisoformat, help="run date for staleness checks (default today)")


def _write_audit(args: argparse.Namespace, inputs, scenario_ids: list[str], *, result=None, dist=None) -> int:
    from dcop.assumptions import build_audit, render_markdown

    audit = build_audit(inputs, scenario_ids, today=args.today, config_dir=args.config_dir, dist=dist, result=result)
    path = args.out / "assumptions.md"
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(audit), encoding="utf-8")
    c = audit.counts()
    print(f"\nassumptions: {c['items']} items checked, {c['placeholder']} placeholder, {c['stale']} stale, "
          f"{c['no source'] + c['no as_of']} untagged, {c['unresolved'] + c['missing file']} broken -> {path}")
    problems = audit.strict_problems()
    if args.strict and problems:
        for where, flags in problems:
            print(f"  strict: {where}: {', '.join(flags)}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcop", description="Delivered Cost of Power model")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run scenarios and write summary.parquet and ledger.parquet")
    run.add_argument("--scenario", action="append", help="scenario id (repeatable); default all")
    run.add_argument("--config-dir", default="config", type=Path)
    run.add_argument("--out", default="outputs", type=Path)
    _audit_args(run)
    audit = sub.add_parser("audit", help="write assumptions.md: every input, its source, and flags")
    audit.add_argument("--scenario", action="append", help="scenario id (repeatable); default all")
    audit.add_argument("--config-dir", default="config", type=Path)
    audit.add_argument("--out", default="outputs", type=Path)
    _audit_args(audit)
    mc = sub.add_parser("mc", help="Monte Carlo P-values, upgrade exposure and tornado")
    mc.add_argument("--scenario", default="base")
    mc.add_argument("--draws", type=int, help="number of draws (default from distributions.yaml)")
    mc.add_argument("--seed", type=int)
    mc.add_argument("--workers", type=int, help="processes (default up to 4)")
    mc.add_argument("--config-dir", default="config", type=Path)
    mc.add_argument("--out", default="outputs", type=Path)
    _audit_args(mc)
    bill = sub.add_parser("bill", help="itemized Austin Energy FY2026 bill for one month")
    bill.add_argument("customer", choices=["residential", "commercial", "large"])
    bill.add_argument("--voltage", choices=["primary", "transmission"], help="large customers")
    bill.add_argument("--hlf", action="store_true", help="high-load-factor contract (large customers)")
    bill.add_argument("--kwh", type=float, required=True, help="energy used in the month")
    bill.add_argument("--kw", type=float, help="peak 15-minute demand (commercial)")
    bill.add_argument("--pf", type=float, help="power factor at peak, e.g. 0.87")
    bill.add_argument("--no-psa-admin", action="store_true",
                      help="leave out the Dec 1, 2025 PSA administrative adjustment")
    bill.add_argument("--config-dir", default="config", type=Path)
    args = parser.parse_args(argv)

    if args.command == "mc":
        return _run_mc(args)

    if args.command == "audit":
        inputs = load_inputs(args.config_dir)
        return _write_audit(args, inputs, args.scenario or list(inputs.scenarios))

    if args.command == "bill":
        from dcop.grid.austin import commercial_statement, large_load_statement, residential_statement

        tariffs = load_inputs(args.config_dir).tariffs
        keep = not args.no_psa_admin
        if args.customer == "residential":
            statement = residential_statement(tariffs, args.kwh, include_psa_admin_adjustment=keep)
        elif args.customer == "large":
            if args.kw is None or args.voltage is None:
                parser.error("large bills need --kw and --voltage")
            statement = large_load_statement(tariffs, args.kwh, args.kw, args.voltage,
                                             high_load_factor=args.hlf, power_factor=args.pf,
                                             include_psa_admin_adjustment=keep)
        else:
            if args.kw is None:
                parser.error("commercial bills need --kw")
            statement = commercial_statement(tariffs, args.kwh, args.kw, power_factor=args.pf,
                                             include_psa_admin_adjustment=keep)
        print(statement.render())
        return 0

    inputs = load_inputs(args.config_dir)
    result = run_all(inputs, args.scenario)
    summary = validated_summary(result)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.out / "summary.parquet", index=False)
    result.ledger.to_frame().to_parquet(args.out / "ledger.parquet", index=False)

    components = levelized_components(result)
    components.to_parquet(args.out / "components.parquet", index=False)
    for sid in result.inputs_by_scenario:
        stack_chart(result, sid, args.out / f"stack_chart_{sid}.png")
        timing_chart(result, sid, args.out / f"timing_chart_{sid}.png")

    with pd.option_context("display.width", 140, "display.float_format", "{:,.2f}".format):
        cols = ["site_id", "path_id", "scenario_id", "ldcop_usd_mwh", "year1_usd_mwh", "year5_usd_mwh",
                "first_power_date", "full_power_date", "availability", "delay_cost_usd"]
        print(summary[cols].to_string(index=False))
        for row in summary[summary["notes"] != ""].itertuples():
            print(f"NOTE {row.site_id}/{row.path_id}/{row.scenario_id}: {row.notes}")
        print("\nLevelized $/MWh by component (table view of the stack chart):")
        table = components.pivot_table(index=["scenario_id", "site_id", "path_id"], columns="group",
                                       values="usd_per_mwh", sort=False)
        print(table.to_string())
    print(f"\nwrote summary, ledger and components parquet files and charts to {args.out}/")
    return _write_audit(args, inputs, list(result.inputs_by_scenario), result=result)


def _run_mc(args: argparse.Namespace) -> int:
    from dcop.io import load_distributions
    from dcop.montecarlo import default_workers, run_monte_carlo
    from dcop.reports import exposure_table, tornado_chart
    from dcop.scenarios import RunCache, apply_overrides, run_sensitivities

    inputs = load_inputs(args.config_dir)
    dist = load_distributions(args.config_dir / "distributions.yaml")
    workers = args.workers or default_workers()
    result = run_monte_carlo(inputs, dist, args.scenario, n=args.draws, seed=args.seed, workers=workers)
    eff = apply_overrides(inputs, inputs.scenarios[args.scenario].overrides)
    exposure = exposure_table(eff)
    sens = run_sensitivities(inputs, args.scenario, dist.sensitivities, RunCache())

    args.out.mkdir(parents=True, exist_ok=True)
    result.draws.to_parquet(args.out / "mc_draws.parquet", index=False)
    result.summary.to_parquet(args.out / "mc_summary.parquet", index=False)
    result.upgrade_exposure.to_parquet(args.out / "upgrade_exposure.parquet", index=False)
    exposure.to_parquet(args.out / "exposure_table.parquet", index=False)
    sens.to_parquet(args.out / "tornado.parquet", index=False)
    for site_id in sorted(set(sens["site_id"])) if len(sens) else []:
        tornado_chart(sens, site_id, args.out / f"tornado_{args.scenario}_{site_id}.png")

    with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
        print(f"Monte Carlo: scenario {args.scenario}, {result.n_draws} draws, seed {result.seed}\n")
        cols = ["site_id", "path_id", "ldcop_base", "ldcop_p10", "ldcop_p50", "ldcop_p90", "n_failed", "p_grid_cheaper"]
        print(result.summary[cols].to_string(index=False))
        print("\nFirst power (base, P10, P50, P90):")
        for r in result.summary.itertuples():
            print(f"  {r.site_id}/{r.path_id}: {r.first_power_base:%Y-%m-%d}  {r.first_power_p10:%Y-%m-%d}  "
                  f"{r.first_power_p50:%Y-%m-%d}  {r.first_power_p90:%Y-%m-%d}")
        print("\nUpgrade exposure (gross allocated $, before refunds):")
        print(result.upgrade_exposure.to_string(index=False))
        print("\nUnconfirmed upgrade items:")
        print(exposure.to_string(index=False) if len(exposure) else "  none")
        print("\nTornado, largest swings (table view of the chart):")
        top = sens[sens["swing"] > 1e-9].sort_values("swing", ascending=False).groupby(["site_id", "path_id"]).head(6)
        print(top[["site_id", "path_id", "label", "range", "ldcop_low", "ldcop_base", "ldcop_high"]].to_string(index=False))
    print(f"\nwrote Monte Carlo, exposure and tornado outputs to {args.out}/")
    from dcop.scenarios import run_all

    return _write_audit(args, inputs, [args.scenario], result=run_all(inputs, [args.scenario]), dist=dist)
