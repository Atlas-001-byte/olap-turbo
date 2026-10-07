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

Join queries (exactly one INNER JOIN) take a separate, simpler path in
:func:`_run_join`: both tables are read in full, the right table is
hashed by its normalized ON key and each left row (in left CSV order)
expands to all matching right rows in right CSV order. ON keys are
finite decimals on *both* sides match by exact decimal value (``1``
equals ``1.0``); every other pairing matches by raw text, so a number
never joins to text. WHERE — the same AND/OR/parentheses predicate —
runs only after the join has fully expanded. The filtered join rows then
feed either a projection or an aggregate: a global ``count(*)`` row with
optional ``sum``/``avg``/``min``/``max`` over joined columns, or per-group
hash tables keyed by the GROUP BY columns (bare or ``table.column``),
with HAVING, ORDER BY and LIMIT applied exactly as for single-table
queries.

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
    JoinOn,
    Query,
    SelectItem,
    SortItem,
    WhereExpr,
    parse_sql,
)

BATCH_SIZE = 1024

# Column aggregates over finite decimals, besides count(*).
_AGG_KINDS = ("sum", "avg", "min", "max")

# Aggregate state and final values are keyed by (kind, table, column); a
# state's value is a running Decimal for sum/avg, the extreme Decimal so far
# for min/max (None until a row matches). The table qualifier is None for
# single-table queries and bare join references alike.
_AggKey = Tuple[str, Optional[str], Optional[str]]


def _agg_key(item: SelectItem) -> _AggKey:
    """The state/output key of one SELECT aggregate, qualifier included."""
    return (item.kind, item.table, item.column)


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
            value = values[(node.kind, node.table, node.column)]
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
        key = _agg_key(item)
        state[key] = None if item.kind in ("min", "max") else Decimal(0)
    return state


def _roll(
    state: Dict[_AggKey, Optional[Decimal]],
    key: _AggKey,
    kind: str,
    value: Decimal,
) -> None:
    """Roll one matched row's value into one aggregate accumulator."""
    current = state[key]
    if kind in ("sum", "avg"):
        state[key] = current + value
    elif kind == "min":
        if current is None or value < current:
            state[key] = value
    else:  # "max"
        if current is None or value > current:
            state[key] = value


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
        _roll(state, _agg_key(item), item.kind, value)


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
        key = _agg_key(item)
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
            value = values[_agg_key(item)]
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
                value = values[key][_agg_key(item)]
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


def _read_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV table into its header and raw data rows (CSV order)."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        rows = [list(row) for row in reader]
    return header, rows


def _join_key(field: str) -> Any:
    """Normalize one ON cell.

    Finite decimal text joins by exact decimal value (so ``1`` and ``1.0``
    match); everything else joins by raw text. A normalized number never
    equals a text key, so mixed numeric/text values never match.
    """
    try:
        value = Decimal(field)
    except (InvalidOperation, ValueError):
        return field
    if not value.is_finite():
        return field
    return value


def _resolve_join_column(
    table: Optional[str],
    name: str,
    left_table: str,
    right_table: str,
    left_header: Sequence[str],
    right_header: Sequence[str],
) -> Tuple[int, int]:
    """Resolve a (possibly bare) reference to ``(side, column_index)``.

    Side 0 is the left/FROM table, side 1 the JOIN table. A bare name present
    in both headers is ambiguous (ValueError); a missing column is KeyError.
    """
    if table is not None and table not in (left_table, right_table):
        raise ValueError(f"unknown table qualifier {table!r} for column {name!r}")
    in_left = name in left_header
    in_right = name in right_header
    if table is None:
        if in_left and in_right:
            raise ValueError(
                f"column {name!r} is ambiguous: it exists in both "
                f"{left_table!r} and {right_table!r}; qualify it with the "
                "table name"
            )
        if not in_left and not in_right:
            raise KeyError(
                f"column {name!r} does not exist in either {left_table!r} "
                f"or {right_table!r}"
            )
        if in_left:
            return 0, left_header.index(name)
        return 1, right_header.index(name)
    if table == left_table:
        if not in_left:
            raise KeyError(
                f"column {name!r} does not exist in table {left_table!r}"
            )
        return 0, left_header.index(name)
    if not in_right:
        raise KeyError(
            f"column {name!r} does not exist in table {right_table!r}"
        )
    return 1, right_header.index(name)


def _resolve_on(
    on: JoinOn,
    left_table: str,
    right_table: str,
    left_header: Sequence[str],
    right_header: Sequence[str],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Resolve the ON equality into one (side, index) pair per table.

    The two sides must come from different tables, one each; a same-table
    pair (after bare resolution) is a ValueError.
    """
    first = _resolve_join_column(
        on.left.table, on.left.name,
        left_table, right_table, left_header, right_header,
    )
    second = _resolve_join_column(
        on.right.table, on.right.name,
        left_table, right_table, left_header, right_header,
    )
    if first[0] == second[0]:
        raise ValueError(
            "ON must compare one column from each table, not two columns of "
            "the same table"
        )
    if first[0] == 0:
        return first, second
    return second, first


def _join_comparison_vector(
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
    position: Tuple[int, int],
    comp: Comparison,
) -> List[bool]:
    """Evaluate one WHERE comparison over every joined row.

    As with the single-table scan, the column is parsed for every joined row
    up front: OR never short-circuits numeric cell validation.
    """
    side, idx = position
    vec = [
        pair[side][idx] if idx < len(pair[side]) else "" for pair in pairs
    ]
    result: List[bool] = [False] * len(vec)
    if comp.quoted:
        target_text = comp.value_text
        for i, field in enumerate(vec):
            result[i] = _compare(field, comp.op, target_text)
    else:
        target = Decimal(comp.value_text)
        label = comp.column if comp.table is None else f"{comp.table}.{comp.column}"
        for i, field in enumerate(vec):
            result[i] = _compare(_decimal(field, label), comp.op, target)
    return result


def _resolve_where_positions(
    node: WhereExpr,
    positions: Dict[int, Tuple[int, int]],
    left_table: str,
    right_table: str,
    left_header: Sequence[str],
    right_header: Sequence[str],
) -> None:
    if isinstance(node, Comparison):
        positions[id(node)] = _resolve_join_column(
            node.table, node.column,
            left_table, right_table, left_header, right_header,
        )
        return
    for operand in node.operands:
        _resolve_where_positions(
            operand, positions,
            left_table, right_table, left_header, right_header,
        )


def _eval_join_where(
    node: WhereExpr,
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
    positions: Dict[int, Tuple[int, int]],
) -> List[bool]:
    """Vectorized boolean-tree evaluation on the joined rows."""
    if isinstance(node, Comparison):
        return _join_comparison_vector(pairs, positions[id(node)], node)
    vectors = [
        _eval_join_where(operand, pairs, positions) for operand in node.operands
    ]
    return _combine(node.op, vectors)


def _join_cell(
    pair: Tuple[Sequence[str], Sequence[str]], position: Tuple[int, int]
) -> str:
    """One raw cell of a joined row; short rows yield empty fields."""
    side, idx = position
    row = pair[side]
    return row[idx] if idx < len(row) else ""


def _qualified_label(table: Optional[str], column: str) -> str:
    return column if table is None else f"{table}.{column}"


def _resolve_join_select(
    query: Query, resolve
) -> Tuple[
    List[Tuple],
    List[Tuple[int, int]],
    List[Tuple[SelectItem, Tuple[int, int]]],
    List[Tuple[int, int]],
]:
    """Resolve every SELECT/GROUP BY column of a join query.

    Returns the canonical key of each SELECT item (plain columns and
    aggregates alike, so differently written references to the same column
    compare equal), the plain-column positions, the (item, position)
    aggregate entries and the grouping positions. Duplicate aggregates by
    resolved column, duplicate grouping columns and aggregates over a
    grouping column are ValueErrors.
    """
    select_keys: List[Tuple] = []
    column_positions: List[Tuple[int, int]] = []
    agg_entries: List[Tuple[SelectItem, Tuple[int, int]]] = []
    for item in query.select:
        if item.kind == "column":
            position = resolve(item.table, item.column)
            column_positions.append(position)
            select_keys.append(("column", position))
        elif item.kind == "count_star":
            select_keys.append(("agg", "count_star", None))
        else:
            position = resolve(item.table, item.column)
            agg_entries.append((item, position))
            select_keys.append(("agg", item.kind, position))
    seen_aggs: set = set()
    for item, position in agg_entries:
        key = (item.kind, position)
        if key in seen_aggs:
            raise ValueError(f"duplicate aggregate {item.text}")
        seen_aggs.add(key)
    group_positions = [resolve(ref.table, ref.name) for ref in query.group_refs]
    if len(set(group_positions)) != len(group_positions):
        raise ValueError("GROUP BY may not list the same column twice")
    group_set = set(group_positions)
    for item, position in agg_entries:
        if position in group_set:
            raise ValueError(
                f"GROUP BY column {_qualified_label(item.table, item.column)} "
                "may not be aggregated"
            )
    return select_keys, column_positions, agg_entries, group_positions


def _resolve_join_order_by(
    query: Query, resolve, select_keys: List[Tuple]
) -> List[Tuple[int, bool]]:
    """Map each ORDER BY item of a join query to its SELECT position.

    Items match SELECT entries by resolved column, so bare and qualified
    references to the same column are identical; duplicate items and
    references absent from the SELECT list are ValueErrors.
    """
    order: List[Tuple[int, bool]] = []
    seen: set = set()
    for sort_item in query.order_by:
        if sort_item.kind == "column":
            key = ("column", resolve(sort_item.table, sort_item.column))
        elif sort_item.kind == "count_star":
            key = ("agg", "count_star", None)
        else:
            key = ("agg", sort_item.kind, resolve(sort_item.table, sort_item.column))
        if key in seen:
            raise ValueError(f"duplicate ORDER BY item {sort_item.text!r}")
        seen.add(key)
        try:
            idx = select_keys.index(key)
        except ValueError:
            if sort_item.kind == "column":
                raise ValueError(
                    f"ORDER BY column {sort_item.text!r} is not in the "
                    "SELECT list"
                )
            raise ValueError(
                f"ORDER BY aggregate {sort_item.text} is not in the "
                "SELECT list"
            )
        order.append((idx, sort_item.descending))
    return order


def _accumulate_join(
    state: Dict[_AggKey, Optional[Decimal]],
    agg_entries: Sequence[Tuple[SelectItem, Tuple[int, int]]],
    pair: Tuple[Sequence[str], Sequence[str]],
) -> None:
    """Roll one matched joined row into the aggregate state."""
    for item, position in agg_entries:
        field = _join_cell(pair, position)
        value = _decimal(field, _qualified_label(item.table, item.column))
        _roll(state, _agg_key(item), item.kind, value)


def _join_aggregate_result(
    query: Query,
    pairs: List[Tuple[Sequence[str], Sequence[str]]],
    agg_entries: List[Tuple[SelectItem, Tuple[int, int]]],
    group_positions: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Aggregate the filtered joined rows, globally or per group.

    Mirrors the single-table scan: groups appear in first-appearance order
    of their keys over the filtered join rows, HAVING is evaluated only
    after every aggregate has been exactly computed, and an empty global
    input yields count 0, sum 0 and null avg/min/max.
    """
    agg_items = [item for item, _ in agg_entries]
    if query.is_grouped:
        # First-appearance order of composite group keys over the filtered
        # join row stream.
        order: List[Tuple[Any, ...]] = []
        group_counts: Dict[Tuple[Any, ...], int] = {}
        group_states: Dict[Tuple[Any, ...], Dict[_AggKey, Optional[Decimal]]] = {}
        for pair in pairs:
            key = tuple(
                _group_key(_join_cell(pair, position))
                for position in group_positions
            )
            if key not in group_counts:
                group_counts[key] = 0
                group_states[key] = _new_agg_state(agg_items)
                order.append(key)
            group_counts[key] += 1
            _accumulate_join(group_states[key], agg_entries, pair)
        group_values = {
            key: _final_values(agg_items, group_counts[key], group_states[key])
            for key in order
        }
        if query.having is not None:
            order = [
                key
                for key in order
                if _eval_having(query.having, group_counts[key], group_values[key])
            ]
        return _grouped_result(query, order, group_counts, group_values)

    count = 0
    state = _new_agg_state(agg_items)
    for pair in pairs:
        count += 1
        _accumulate_join(state, agg_entries, pair)
    values = _final_values(agg_items, count, state)
    result = _aggregate_result(query, count, values)
    if query.having is not None and not _eval_having(query.having, count, values):
        # A global aggregate always computes one row; HAVING failing leaves
        # the same columns but no rows.
        result["rows"] = []
        result["row_count"] = 0
    return result


def _run_join(data_dir: str, query: Query) -> Dict[str, Any]:
    """Execute the single supported INNER JOIN query end to end."""
    left_table = query.table
    right_table = query.join_table
    # Built-in open raises FileNotFoundError, preserving the required type;
    # the left (FROM) table is opened before the right (JOIN) table.
    left_path = Path(data_dir) / f"{left_table}.csv"
    right_path = Path(data_dir) / f"{right_table}.csv"
    left_header, left_rows = _read_table(left_path)
    right_header, right_rows = _read_table(right_path)

    def resolve(table: Optional[str], name: str) -> Tuple[int, int]:
        return _resolve_join_column(
            table, name, left_table, right_table, left_header, right_header
        )

    # ON first: it uniquely defines the cross-table column pair.
    left_pos, right_pos = _resolve_on(
        query.join_on, left_table, right_table, left_header, right_header
    )

    # Hash the right table by normalized join key; appending in CSV order
    # keeps each bucket in right-table CSV order.
    right_buckets: Dict[Any, List[Sequence[str]]] = {}
    right_idx = right_pos[1]
    for row in right_rows:
        field = row[right_idx] if right_idx < len(row) else ""
        right_buckets.setdefault(_join_key(field), []).append(row)

    # SELECT, GROUP BY and aggregate columns, resolved up front so
    # unknown/ambiguous columns fail before any row is emitted.
    select_keys, column_positions, agg_entries, group_positions = (
        _resolve_join_select(query, resolve)
    )

    where_positions: Dict[int, Tuple[int, int]] = {}
    if query.where is not None:
        _resolve_where_positions(
            query.where, where_positions,
            left_table, right_table, left_header, right_header,
        )

    pairs: List[Tuple[Sequence[str], Sequence[str]]] = []
    left_idx = left_pos[1]
    # Every left row is visited in left CSV order; each left row expands to
    # all of its matching right rows in right CSV order.
    for lrow in left_rows:
        field = lrow[left_idx] if left_idx < len(lrow) else ""
        for rrow in right_buckets.get(_join_key(field), ()):  # inner join
            pairs.append((lrow, rrow))

    # WHERE runs only after the join has fully expanded the rows.
    if query.where is not None:
        mask = _eval_join_where(query.where, pairs, where_positions)
        pairs = [pair for pair, matched in zip(pairs, mask) if matched]

    if query.is_aggregate:
        result = _join_aggregate_result(
            query, pairs, agg_entries, group_positions
        )
    else:
        projected_rows = []
        for pair in pairs:
            projected_rows.append(
                [
                    _project_value(_join_cell(pair, position))
                    for position in column_positions
                ]
            )
        result = {
            "columns": [item.text for item in query.select],
            "rows": projected_rows,
            "row_count": len(projected_rows),
        }

    # ORDER BY sorts the fully joined/filtered/aggregated rows; stability
    # keeps the incoming order on ties (join row order for projections,
    # first-appearance order for groups). LIMIT truncates last.
    if query.order_by:
        order = _resolve_join_order_by(query, resolve, select_keys)
        for idx, descending in reversed(order):
            result["rows"].sort(
                key=lambda row, i=idx: _sort_key(row[i]), reverse=descending
            )
    if query.limit is not None:
        _apply_limit(result, query.limit)
    return result


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


def _read_header(path: Path) -> List[str]:
    """Read only a CSV table's header (first row); data rows are untouched."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def _collect_where_columns_single(
    node: WhereExpr, cols: List[str], seen: set
) -> None:
    """First-appearance order of the bare columns a single-table WHERE reads."""
    if isinstance(node, Comparison):
        if node.column not in seen:
            seen.add(node.column)
            cols.append(node.column)
        return
    for operand in node.operands:
        _collect_where_columns_single(operand, cols, seen)


def _aggregates_plan(query: Query) -> List[Dict[str, Any]]:
    """SELECT aggregates in SELECT order; count(*) has a null column."""
    aggregates: List[Dict[str, Any]] = []
    for item in query.select:
        if item.kind == "count_star":
            aggregates.append(
                {"function": "count", "column": None, "text": item.text}
            )
        elif item.kind in _AGG_KINDS:
            aggregates.append(
                {
                    "function": item.kind,
                    "column": _qualified_label(item.table, item.column),
                    "text": item.text,
                }
            )
    return aggregates


def _order_by_plan(query: Query) -> List[Dict[str, str]]:
    return [
        {
            "expression": item.text,
            "direction": "DESC" if item.descending else "ASC",
        }
        for item in query.order_by
    ]


def _query_type(query: Query) -> str:
    if query.is_join:
        return "join"
    if query.is_grouped:
        return "grouped_aggregate"
    if query.is_aggregate:
        return "aggregate"
    return "projection"


def _explain_single(data_dir: str, query: Query) -> Dict[str, Any]:
    """Plan a single-table query; only the CSV header is read."""
    table_path = Path(data_dir) / f"{query.table}.csv"
    # Built-in open raises FileNotFoundError, preserving the required type.
    header = _read_header(table_path)
    required = query.required_columns()
    for name in required:
        if name not in header:
            raise KeyError(
                f"column {name!r} does not exist in table {query.table!r}"
            )

    # The scanner materializes exactly the required columns, in CSV header
    # order; count(*) reads no column.
    required_set = set(required)
    scan_columns: List[str] = []
    seen_scan: set = set()
    for name in header:
        if name in required_set and name not in seen_scan:
            seen_scan.add(name)
            scan_columns.append(name)

    where_columns: List[str] = []
    if query.where is not None:
        _collect_where_columns_single(query.where, where_columns, set())

    # SELECT selection originals, first appearance kept (aggregates are
    # reported separately in "aggregates").
    select_columns: List[str] = []
    seen_select: set = set()
    for item in query.select:
        if item.kind == "column" and item.text not in seen_select:
            seen_select.add(item.text)
            select_columns.append(item.text)

    return {
        "query_type": _query_type(query),
        "tables": [query.table],
        "scan_columns": scan_columns,
        "where_columns": where_columns,
        "select_columns": select_columns,
        "group_columns": list(query.group_by),
        "aggregates": _aggregates_plan(query),
        "join": None,
        "order_by": _order_by_plan(query),
        "limit": query.limit,
    }


def _collect_where_positions_ordered(
    node: WhereExpr,
    resolve,
    positions: List[Tuple[int, int]],
    seen: set,
) -> None:
    """Resolve WHERE columns in written order, each physical column once."""
    if isinstance(node, Comparison):
        position = resolve(node.table, node.column)
        if position not in seen:
            seen.add(position)
            positions.append(position)
        return
    for operand in node.operands:
        _collect_where_positions_ordered(operand, resolve, positions, seen)


def _explain_join(data_dir: str, query: Query) -> Dict[str, Any]:
    """Plan the single INNER JOIN query; only the two CSV headers are read."""
    left_table = query.table
    right_table = query.join_table
    # The left (FROM) table is opened before the right (JOIN) table, just as
    # in _run_join; built-in open raises FileNotFoundError.
    left_header = _read_header(Path(data_dir) / f"{left_table}.csv")
    right_header = _read_header(Path(data_dir) / f"{right_table}.csv")

    def resolve(table: Optional[str], name: str) -> Tuple[int, int]:
        return _resolve_join_column(
            table, name, left_table, right_table, left_header, right_header
        )

    # ON, SELECT, GROUP BY, aggregates and WHERE are resolved exactly as
    # during execution, so an unknown/ambiguous column raises
    # KeyError/ValueError without a row read.
    left_pos, right_pos = _resolve_on(
        query.join_on, left_table, right_table, left_header, right_header
    )
    select_keys, column_positions, agg_entries, group_positions = (
        _resolve_join_select(query, resolve)
    )
    where_positions: List[Tuple[int, int]] = []
    if query.where is not None:
        _collect_where_positions_ordered(
            query.where, resolve, where_positions, set()
        )

    # Mirror the execution-time ORDER BY validation: duplicates by resolved
    # column and sort items absent from the SELECT list are ValueErrors.
    _resolve_join_order_by(query, resolve, select_keys)

    headers = (left_header, right_header)
    tables = (left_table, right_table)

    def qualified(position: Tuple[int, int]) -> str:
        side, idx = position
        return f"{tables[side]}.{headers[side][idx]}"

    # Every column the join actually materializes: the two ON keys, the
    # SELECT columns, the grouping and aggregate columns and the WHERE
    # columns, left table first in each header's column order, then the
    # right table.
    scan_set = {left_pos, right_pos}
    scan_set.update(column_positions)
    scan_set.update(position for _, position in agg_entries)
    scan_set.update(group_positions)
    scan_set.update(where_positions)
    scan_columns = [
        qualified((side, idx))
        for side in (0, 1)
        for idx in sorted(p[1] for p in scan_set if p[0] == side)
    ]

    where_columns = [qualified(position) for position in where_positions]

    # SELECT selection originals, first appearance kept (aggregates are
    # reported separately in "aggregates").
    select_columns: List[str] = []
    seen_select: set = set()
    for item in query.select:
        if item.kind == "column" and item.text not in seen_select:
            seen_select.add(item.text)
            select_columns.append(item.text)

    return {
        "query_type": "join",
        "tables": [left_table, right_table],
        "scan_columns": scan_columns,
        "where_columns": where_columns,
        "select_columns": select_columns,
        "group_columns": [ref.text for ref in query.group_refs],
        "aggregates": _aggregates_plan(query),
        "join": {
            "left_table": left_table,
            "right_table": right_table,
            "left_column": left_header[left_pos[1]],
            "right_column": right_header[right_pos[1]],
        },
        "order_by": _order_by_plan(query),
        "limit": query.limit,
    }


def explain(data_dir: str, sql: str) -> Dict[str, Any]:
    """Return the query plan for *sql* as a JSON-serializable dict.

    Only the CSV headers are read, never data rows: column pruning, predicate
    pushdown and the execution order are described statically. Non-numeric
    cells never fail.

    Raises:
        FileNotFoundError: a table's CSV file does not exist.
        KeyError: the query references a column absent from a header.
        ValueError: the SQL is malformed/unsupported.
    """
    query: Query = parse_sql(sql)
    if query.is_join:
        return _explain_join(data_dir, query)
    return _explain_single(data_dir, query)


def execute(data_dir: str, sql: str) -> Dict[str, Any]:
    """Run *sql* against CSV tables in *data_dir*; return a result dict.

    Raises:
        FileNotFoundError: the table's CSV file does not exist.
        KeyError: the query references a column absent from the header.
        ValueError: the SQL is malformed/unsupported, or a required field
            cannot be read as a decimal number.
    """
    query: Query = parse_sql(sql)
    if query.is_join:
        return _run_join(data_dir, query)
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
