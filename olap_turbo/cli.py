"""命令行入口：``python -m olap_turbo --data-dir <目录> --query <SQL>``。"""

import argparse
import json
import sys

from .engine import run_query


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="olap_turbo",
        description="对目录中的 CSV 表执行简单 SQL 查询，结果以 JSON 输出到 stdout。",
    )
    parser.add_argument("--data-dir", required=True, help="存放 CSV 表的目录")
    parser.add_argument("--query", required=True, help="要执行的 SQL 查询")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    result = run_query(args.data_dir, args.query)
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
