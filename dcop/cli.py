"""Command line: ``dcop run --scenario base``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dcop.io import load_inputs
from dcop.reports import levelized_components, stack_chart, timing_chart, validated_summary
from dcop.scenarios import run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcop", description="Delivered Cost of Power model")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run scenarios and write summary.parquet and ledger.parquet")
    run.add_argument("--scenario", action="append", help="scenario id (repeatable); default all")
    run.add_argument("--config-dir", default="config", type=Path)
    run.add_argument("--out", default="outputs", type=Path)
    bill = sub.add_parser("bill", help="itemized Austin Energy FY2026 bill for one month")
    bill.add_argument("customer", choices=["residential", "commercial"])
    bill.add_argument("--kwh", type=float, required=True, help="energy used in the month")
    bill.add_argument("--kw", type=float, help="peak 15-minute demand (commercial)")
    bill.add_argument("--pf", type=float, help="power factor at peak, e.g. 0.87")
    bill.add_argument("--no-psa-admin", action="store_true",
                      help="leave out the Dec 1, 2025 PSA administrative adjustment")
    bill.add_argument("--config-dir", default="config", type=Path)
    args = parser.parse_args(argv)

    if args.command == "bill":
        from dcop.grid.austin import commercial_statement, residential_statement

        tariffs = load_inputs(args.config_dir).tariffs
        keep = not args.no_psa_admin
        if args.customer == "residential":
            statement = residential_statement(tariffs, args.kwh, include_psa_admin_adjustment=keep)
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
        print("\nLevelized $/MWh by component (table view of the stack chart):")
        table = components.pivot_table(index=["scenario_id", "site_id", "path_id"], columns="group",
                                       values="usd_per_mwh", sort=False)
        print(table.to_string())
    print(f"\nwrote summary, ledger and components parquet files and charts to {args.out}/")
    return 0
