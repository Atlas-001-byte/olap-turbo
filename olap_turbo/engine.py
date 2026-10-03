"""Columnar execution engine.

The scanner reads each CSV table in record batches of at most
:data:`BATCH_SIZE` rows, materializing only the columns referenced by the
SELECT list, the GROUP BY key or the WHERE clause (column pruning).
Within a batch the predicate — an AND/OR/parentheses boolean tree of
column-op-value comparisons — is evaluated column-at-a-time *before*
projection or grouping (predicate pushdown), and aggregates — either a
single global row or per-group hash tables keyed by one GROUP BY column
or a tuple of several GROUP BY columns — roll forward batch by batch.

Every cell read by a numeric comparison is parsed eagerly for every row
of a batch, even on rows whose other OR branches already settle the
outcome: OR never short-circuits cell validation, so a non-numeric cell
anywhere a numeric predicate reaches fails the whole query.

All arithmetic and numeric comparison go through :class:`decimal.Decimal`
so results are deterministic and free of binary-float artifacts.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .sql import Comparison, Query, SelectItem, WhereExpr, parse_sql

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


def _comparison_vector(
    rows: Sequence[Sequence[str]],
    col_index: Dict[str, int],
    comp: Comparison,
) -> List[bool]:
    """Evaluate one comparison over every row of the batch.

    The column is parsed for *every* row up front, including rows whose
    other OR branches already decide the row: OR does not short-circuit
    cell validation.
    """
    idx = col_index[comp.column]
    vec = _column_vector(rows, idx)
    result: List[bool] = [False] * len(vec)
    if comp.quoted:
        target_text = comp.value_text
        for i, field in enumerate(vec):
            result[i] = _compare(field, comp.op, target_text)
    else:
        target = Decimal(comp.value_text)
        for i, field in enumerate(vec):
            result[i] = _compare(_decimal(field, comp.column), comp.op, target)
    return result


def _combine(op: str, vectors: Sequence[Sequence[bool]]) -> List[bool]:
    """Combine per-row boolean vectors with AND or OR."""
    if op == "AND":
        mask = [True] * len(vectors[0])
        for vec in vectors:
            for i, value in enumerate(vec):
                mask[i] = mask[i] and value
        return mask
    mask = [False] * len(vectors[0])
    for vec in vectors:
        for i, value in enumerate(vec):
            mask[i] = mask[i] or value
    return mask


def _eval_node(
    node: WhereExpr,
    rows: Sequence[Sequence[str]],
    col_index: Dict[str, int],
) -> List[bool]:
    """Vectorized boolean-tree evaluation on raw columns."""
    if isinstance(node, Comparison):
        return _comparison_vector(rows, col_index, node)
    vectors = [_eval_node(operand, rows, col_index) for operand in node.operands]
    return _combine(node.op, vectors)


def _evaluate_predicate(
    rows: List[Sequence[str]],
    col_index: Dict[str, int],
    where: WhereExpr,
) -> List[bool]:
    """Vectorized predicate tree; runs on raw columns, ahead of projection."""
    return _eval_node(where, rows, col_index)


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


def _group_key(field: str) -> Any:
    """Normalize a grouping cell.

    Finite decimal text groups by exact decimal value (so ``1`` and
    ``1.0`` share a group); everything else groups by raw text.
    """
    try:
        value = Decimal(field)
    except (InvalidOperation, ValueError):
        return field
    if not value.is_finite():
        return field
    return value


def _key_output(key: Any) -> Any:
    """Render a grouping key for JSON: numeric keys follow projection rules."""
    if isinstance(key, Decimal):
        return _number(key)
    return key


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


def _grouped_result(
    query: Query,
    order: List[Tuple[Any, ...]],
    counts: Dict[Tuple[Any, ...], int],
    sums: Dict[Tuple[Any, ...], Dict[str, Decimal]],
) -> Dict[str, Any]:
    group_cols = query.group_by
    rows: List[List[Any]] = []
    for key in order:
        row: List[Any] = []
        for pos, item in enumerate(query.select):
            if pos < len(group_cols):
                # The leading SELECT items are the GROUP BY columns, in order.
                row.append(_key_output(key[pos]))
            elif item.kind == "count_star":
                row.append(counts[key])
            else:
                row.append(_number(sums[key][item.column]))
        rows.append(row)
    return {
        "columns": [item.text for item in query.select],
        "rows": rows,
        "row_count": len(rows),
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

    if query.is_grouped:
        # First-appearance order of composite group keys over the
        # filtered row stream.
        order: List[Tuple[Any, ...]] = []
        group_counts: Dict[Tuple[Any, ...], int] = {}
        group_sums: Dict[Tuple[Any, ...], Dict[str, Decimal]] = {}
        group_idxs = [col_index[name] for name in query.group_by]
        sum_items = [item for item in select_cols if item.kind == "sum"]

        while True:
            batch = [row for _, row in zip(range(BATCH_SIZE), reader)]
            if not batch:
                break
            mask = (
                [True] * len(batch)
                if query.where is None
                else _evaluate_predicate(batch, col_index, query.where)
            )
            for i, matched in enumerate(mask):
                if not matched:
                    continue
                row = batch[i]
                key = tuple(
                    _group_key(row[idx] if idx < len(row) else "")
                    for idx in group_idxs
                )
                if key not in group_counts:
                    group_counts[key] = 0
                    group_sums[key] = {item.column: Decimal(0) for item in sum_items}
                    order.append(key)
                group_counts[key] += 1
                for item in sum_items:
                    idx = col_index[item.column]
                    sum_field = row[idx] if idx < len(row) else ""
                    group_sums[key][item.column] += _decimal(sum_field, item.column)

        return _grouped_result(query, order, group_counts, group_sums)

    while True:
        batch = [row for _, row in zip(range(BATCH_SIZE), reader)]
        if not batch:
            break
        mask = (
            [True] * len(batch)
            if query.where is None
            else _evaluate_predicate(batch, col_index, query.where)
        )

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
