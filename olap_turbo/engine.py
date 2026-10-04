"""Columnar execution engine.

The scanner reads each CSV table in record batches of at most
:data:`BATCH_SIZE` rows, materializing only the columns referenced by the
SELECT list, the GROUP BY key or the WHERE clause (column pruning).
Within a batch the predicate — an AND/OR/parentheses boolean tree of
column-op-value comparisons — is evaluated column-at-a-time *before*
projection or grouping (predicate pushdown), and aggregates — either a
single global row or per-group hash tables keyed by one GROUP BY column
or a tuple of several GROUP BY columns — roll forward batch by batch.
Besides ``count(*)``, the column aggregates are ``sum``, ``avg``, ``min``
and ``max``: sums and average accumulators add every matched row's cell,
min/max keep the extreme value seen so far. On an empty filtered input the
global ``avg``/``min``/``max`` results are null (and compare false in
HAVING) while ``sum`` is 0; empty inputs produce no groups at all.
HAVING, when present, is evaluated only after the aggregates have been
exactly computed: it drops the global row or individual groups, never
individual input rows.

ORDER BY, when present, sorts the surviving result rows — after WHERE,
aggregation and HAVING — by one or more SELECT result columns, each with
its own ASC (default) or DESC direction. Finite-decimal values compare by
exact numeric value (``1`` equals ``1.0``); everything else compares by
raw text, never coerced to a number. Multi-item keys compare left to
right; rows equal on every key keep their incoming order (CSV row order
for projections, first-appearance order for groups). LIMIT, when present,
truncates the sorted rows last.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sql import (
    Comparison,
    HavingComparison,
    HavingExpr,
    Query,
    SelectItem,
    SortItem,
    WhereExpr,
    parse_sql,
)

BATCH_SIZE = 1024

# Column aggregates over finite decimals, besides count(*).
_AGG_KINDS = ("sum", "avg", "min", "max")

# Aggregate state and final values are keyed by (kind, column); a state's
# value is a running Decimal for sum/avg, the extreme Decimal so far for
# min/max (None until a row matches).
_AggKey = Tuple[str, str]


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


def _eval_having(
    node: HavingExpr, count: int, values: Dict[_AggKey, Optional[Decimal]]
) -> bool:
    """Evaluate a HAVING tree against one fully computed aggregate row.

    Left operands are always numeric (an integer row count or an exactly
    computed Decimal aggregate); textual aggregate results are never
    coerced, and the right operands are the parser-validated finite decimal
    literals. A null aggregate — avg/min/max over an empty global input —
    makes its comparison false.
    """
    if isinstance(node, HavingComparison):
        if node.kind == "count_star":
            left: Decimal = Decimal(count)
        else:
            value = values[(node.kind, node.column)]
            if value is None:
                return False
            left = value
        return _compare(left, node.op, Decimal(node.value_text))
    if node.op == "AND":
        return all(_eval_having(operand, count, values) for operand in node.operands)
    return any(_eval_having(operand, count, values) for operand in node.operands)


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
    """Render an aggregate value as int when integral, else keep Decimal."""
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


def _new_agg_state(agg_items: Sequence[SelectItem]) -> Dict[_AggKey, Optional[Decimal]]:
    """Fresh per-row-stream aggregate state.

    sum/avg start at a zero running sum; min/max start empty (None) until
    the first matched row supplies a value.
    """
    state: Dict[_AggKey, Optional[Decimal]] = {}
    for item in agg_items:
        key = (item.kind, item.column)
        state[key] = None if item.kind in ("min", "max") else Decimal(0)
    return state


def _accumulate(
    state: Dict[_AggKey, Optional[Decimal]],
    agg_items: Sequence[SelectItem],
    row: Sequence[str],
    col_index: Dict[str, int],
) -> None:
    """Roll one matched row into the aggregate state."""
    for item in agg_items:
        idx = col_index[item.column]
        field = row[idx] if idx < len(row) else ""
        value = _decimal(field, item.column)
        key = (item.kind, item.column)
        current = state[key]
        if item.kind in ("sum", "avg"):
            state[key] = current + value
        elif item.kind == "min":
            if current is None or value < current:
                state[key] = value
        else:  # "max"
            if current is None or value > current:
                state[key] = value


def _final_values(
    agg_items: Sequence[SelectItem],
    count: int,
    state: Dict[_AggKey, Optional[Decimal]],
) -> Dict[_AggKey, Optional[Decimal]]:
    """Finalize one aggregate state into output values.

    avg divides its exact running sum by the exact row count (Decimal
    division, free of binary-float artifacts); min/max stay as recorded.
    On an empty input avg/min/max are None (JSON null) while sum is 0.
    """
    final: Dict[_AggKey, Optional[Decimal]] = {}
    for item in agg_items:
        key = (item.kind, item.column)
        value = state[key]
        if item.kind == "avg":
            final[key] = None if count == 0 else value / count
        else:
            final[key] = value
    return final


def _aggregate_result(
    query: Query, count: int, values: Dict[_AggKey, Optional[Decimal]]
) -> Dict[str, Any]:
    row: List[Any] = []
    for item in query.select:
        if item.kind == "count_star":
            row.append(count)
        else:
            value = values[(item.kind, item.column)]
            row.append(None if value is None else _number(value))
    return {
        "columns": [item.text for item in query.select],
        "rows": [row],
        "row_count": 1,
    }


def _grouped_result(
    query: Query,
    order: List[Tuple[Any, ...]],
    counts: Dict[Tuple[Any, ...], int],
    values: Dict[Tuple[Any, ...], Dict[_AggKey, Optional[Decimal]]],
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
                value = values[key][(item.kind, item.column)]
                row.append(None if value is None else _number(value))
        rows.append(row)
    return {
        "columns": [item.text for item in query.select],
        "rows": rows,
        "row_count": len(rows),
    }


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Order key for one result cell.

    Finite decimals (already int/Decimal after projection, key rendering or
    aggregation) compare by exact numeric value, so ``1`` and ``1.0`` tie;
    everything else compares by raw text and is never coerced to a number.
    Numbers order before text so mixed columns stay deterministic.
    """
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        return (0, Decimal(value))
    return (1, str(value))


def _apply_order_by(
    result: Dict[str, Any],
    order_by: Tuple[SortItem, ...],
    select: Tuple[SelectItem, ...],
) -> None:
    """Sort result rows in place, one stable pass per item, last item first.

    Each sort item was validated against the SELECT list at parse time, so
    its column position always resolves. Stability keeps rows that tie on
    every key in their incoming order (CSV row order for projections,
    first-appearance order for groups).
    """
    rows = result["rows"]
    for item in reversed(order_by):
        idx = next(
            pos
            for pos, si in enumerate(select)
            if si.kind == item.kind and si.column == item.column
        )
        rows.sort(key=lambda row, i=idx: _sort_key(row[i]), reverse=item.descending)


def _apply_limit(result: Dict[str, Any], limit: int) -> None:
    """Truncate result rows in place; LIMIT 0 keeps columns, empties rows."""
    del result["rows"][limit:]
    result["row_count"] = len(result["rows"])


def _run_batches(
    reader: csv.reader,
    query: Query,
    header: Sequence[str],
) -> Dict[str, Any]:
    col_index = {name: header.index(name) for name in query.required_columns()}

    projected_rows: List[List[Any]] = []
    count = 0
    select_cols: Tuple[SelectItem, ...] = query.select
    agg_items = [item for item in select_cols if item.kind in _AGG_KINDS]
    state = _new_agg_state(agg_items)

    if query.is_grouped:
        # First-appearance order of composite group keys over the
        # filtered row stream.
        order: List[Tuple[Any, ...]] = []
        group_counts: Dict[Tuple[Any, ...], int] = {}
        group_states: Dict[Tuple[Any, ...], Dict[_AggKey, Optional[Decimal]]] = {}
        group_idxs = [col_index[name] for name in query.group_by]

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
                    group_states[key] = _new_agg_state(agg_items)
                    order.append(key)
                group_counts[key] += 1
                _accumulate(group_states[key], agg_items, row, col_index)

        # Finalize every group exactly before HAVING or output; groups
        # always hold at least one row, so their avg/min/max are non-null.
        group_values = {
            key: _final_values(agg_items, group_counts[key], group_states[key])
            for key in order
        }
        if query.having is not None:
            # HAVING runs after every group has been exactly computed and
            # only drops whole groups; first-appearance order is preserved.
            order = [
                key
                for key in order
                if _eval_having(query.having, group_counts[key], group_values[key])
            ]
        return _grouped_result(query, order, group_counts, group_values)

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
                _accumulate(state, agg_items, batch[i], col_index)
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
        values = _final_values(agg_items, count, state)
        result = _aggregate_result(query, count, values)
        if query.having is not None and not _eval_having(
            query.having, count, values
        ):
            # A global aggregate always computes one row; HAVING failing
            # leaves the same columns but no rows.
            result["rows"] = []
            result["row_count"] = 0
        return result
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
        result = _run_batches(reader, query, header)
        # ORDER BY sorts the fully computed result rows; LIMIT truncates last.
        if query.order_by:
            _apply_order_by(result, query.order_by, query.select)
        if query.limit is not None:
            _apply_limit(result, query.limit)
        return result


def render(result: Dict[str, Any]) -> str:
    """Serialize a query result as a single JSON object."""
    return _dumps(result)
