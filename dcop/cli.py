"""Command line: ``dcop run --scenario base``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dcop.io import load_inputs
from dcop.reports import validated_summary
from dcop.scenarios import run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcop", description="Delivered Cost of Power model")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run scenarios and write summary.parquet and ledger.parquet")
    run.add_argument("--scenario", action="append", help="scenario id (repeatable); default all")
    run.add_argument("--config-dir", default="config", type=Path)
    run.add_argument("--out", default="outputs", type=Path)
    args = parser.parse_args(argv)

    inputs = load_inputs(args.config_dir)
    result = run_all(inputs, args.scenario)
    summary = validated_summary(result)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.out / "summary.parquet", index=False)
    result.ledger.to_frame().to_parquet(args.out / "ledger.parquet", index=False)

    with pd.option_context("display.width", 140, "display.float_format", "{:,.2f}".format):
        print(summary.drop(columns=["discount_rate"]).to_string(index=False))
    print(f"\nwrote {args.out / 'summary.parquet'} and {args.out / 'ledger.parquet'}")
    return 0
