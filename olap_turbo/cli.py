"""Command line interface: ``python -m olap_turbo --data-dir <dir> --query <sql>``."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .engine import execute, explain, render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="olap_turbo",
        description="Query CSV tables in a directory with a small SQL subset.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="directory containing <table>.csv files",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="SQL query (SELECT ... FROM ... [INNER JOIN ... ON ...] "
        "[WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT n])",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print only the query plan as JSON (column pruning, predicate "
        "pushdown and execution order); no data rows are read",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.explain:
        plan = explain(args.data_dir, args.query)
    else:
        plan = execute(args.data_dir, args.query)
    sys.stdout.write(render(plan) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
