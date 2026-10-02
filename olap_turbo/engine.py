"""Columnar execution engine.

The scanner reads each CSV table in record batches of at most
:data:`BATCH_SIZE` rows, materializing only the columns referenced by the
SELECT list or the WHERE clause (column pruning). Within a batch the
predicate is evaluated column-at-a-time *before* projection (predicate
pushdown), and aggregates roll forward batch by batch.

All arithmetic and numeric comparison go through :class:`decimal.Decimal`
so results are deterministic and free of binary-float artifacts.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .sql import Comparison, Query, SelectItem, parse_sql

BATCH_SIZE = 1024


def _decimal(field: str, column: str) -> Decimal:
    try:
        value = Decimal(field)
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"column {column!r} contains non-numeric value {field!r}"
        )
    if not value.is_finite():
        raise ValueError(
            f"column {column!r} contains non-numeric value {field!r}"
        )
    return value


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "=":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    return left <= right  # "<="


def _column_vector(rows: Sequence[Sequence[str]], idx: int) -> List[str]:
    """Extract one column from a row batch; short rows yield empty fields."""
    vec: List[str] = []
    for row in rows:
        vec.append(row[idx] if idx < len(row) else "")
    return vec


def _evaluate_predicate(
    rows: List[Sequence[str]],
    col_index: Dict[str, int],
    where: Sequence[Comparison],
) -> List[bool]:
    """Vectorized conjunction; runs on raw columns, ahead of projection."""
    mask = [True] * len(rows)
    for comp in where:
        idx = col_index[comp.column]
        vec = _column_vector(rows, idx)
        if comp.quoted:
            target_text = comp.value_text
            for i, field in enumerate(vec):
                if mask[i]:
                    mask[i] = _compare(field, comp.op, target_text)
        else:
            target = Decimal(comp.value_text)
            for i, field in enumerate(vec):
                if mask[i]:
                    mask[i] = _compare(_decimal(field, comp.column), comp.op, target)
    return mask


def _project_value(field: str) -> Any:
    """Infer a projected cell: decimal-able text becomes a JSON number."""
    try:
        value = Decimal(field)
    except (InvalidOperation, ValueError):
        return field
    if not value.is_finite():
        return field
    if value == value.to_integral_value():
        return int(value)
    return value


def _number(value: Decimal) -> Any:
    """Render an aggregate sum as int when integral, else keep Decimal."""
    if value == value.to_integral_value():
        return int(value)
    return value


def _aggregate_result(query: Query, count: int, sums: Dict[str, Decimal]) -> Dict[str, Any]:
    row: List[Any] = []
    for item in query.select:
        if item.kind == "count_star":
            row.append(count)
        else:
            row.append(_number(sums[item.column]))
    return {
        "columns": [item.text for item in query.select],
        "rows": [row],
        "row_count": 1,
    }


def _run_batches(
    reader: csv.reader,
    query: Query,
    header: Sequence[str],
) -> Dict[str, Any]:
    col_index = {name: header.index(name) for name in query.required_columns()}

    projected_rows: List[List[Any]] = []
    count = 0
    sums: Dict[str, Decimal] = {
        item.column: Decimal(0)
        for item in query.select
        if item.kind == "sum"
    }
    select_cols: Tuple[SelectItem, ...] = query.select

    while True:
        batch = [row for _, row in zip(range(BATCH_SIZE), reader)]
        if not batch:
            break
        mask = _evaluate_predicate(batch, col_index, query.where)

        if query.is_aggregate:
            for i, matched in enumerate(mask):
                if not matched:
                    continue
                count += 1
                row = batch[i]
                for item in select_cols:
                    if item.kind == "sum":
                        field = row[col_index[item.column]] if col_index[item.column] < len(row) else ""
                        sums[item.column] += _decimal(field, item.column)
        else:
            indices = [col_index[item.column] for item in select_cols]
            for i, matched in enumerate(mask):
                if not matched:
                    continue
                row = batch[i]
                projected_rows.append(
                    [
                        _project_value(row[idx] if idx < len(row) else "")
                        for idx in indices
                    ]
                )

    if query.is_aggregate:
        return _aggregate_result(query, count, sums)
    return {
        "columns": [item.text for item in select_cols],
        "rows": projected_rows,
        "row_count": len(projected_rows),
    }


def _dumps(obj: Any) -> str:
    """Deterministic JSON encoder; Decimal values render as decimal numbers."""

    def encode(value: Any) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        if isinstance(value, bool):
            return "false"
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return _json_string(value)
        if isinstance(value, list):
            return "[" + ",".join(encode(v) for v in value) + "]"
        if isinstance(value, dict):
            return (
                "{"
                + ",".join(
                    f"{_json_string(k)}:{encode(v)}" for k, v in value.items()
                )
                + "}"
            )
        raise TypeError(f"cannot serialize {type(value).__name__}")

    return encode(obj)


def _json_string(text: str) -> str:
    out = ['"']
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def execute(data_dir: str, sql: str) -> Dict[str, Any]:
    """Run *sql* against CSV tables in *data_dir*; return a result dict.

    Raises:
        FileNotFoundError: the table's CSV file does not exist.
        KeyError: the query references a column absent from the header.
        ValueError: the SQL is malformed/unsupported, or a required field
            cannot be read as a decimal number.
    """
    query: Query = parse_sql(sql)
    table_path = Path(data_dir) / f"{query.table}.csv"
    # Built-in open raises FileNotFoundError, preserving the required type.
    with open(table_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        for name in query.required_columns():
            if name not in header:
                raise KeyError(
                    f"column {name!r} does not exist in table {query.table!r}"
                )
        return _run_batches(reader, query, header)


def render(result: Dict[str, Any]) -> str:
    """Serialize a query result as a single JSON object."""
    return _dumps(result)
