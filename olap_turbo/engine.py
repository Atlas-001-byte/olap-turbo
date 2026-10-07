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
runs only after the join has fully expanded. From there a join query
follows the same shapes as a single-table query: the filtered joined
rows are either projected directly, rolled into one global aggregate
row (``count(*)`` plus optional ``sum``/``avg``/``min``/``max`` over
columns of either table, qualified or uniquely resolvable bare names),
or grouped by one or more columns and aggregated per group. Grouping
keys normalize exactly as on a single table; groups emit in their
first-appearance order over the filtered join stream. HAVING compares
only aggregates already selected and drops the global row or whole
groups, then projection-equivalent ORDER BY (stable, preserving join
row or first-appearance group order) and LIMIT apply exactly as for
single-table queries. Every join reference — SELECT, ON, WHERE,
GROUP BY, aggregate arguments, HAVING and ORDER BY — is resolved
against the two headers up front, shared by execute and explain, so
ambiguity, unknown columns and unselected HAVING/ORDER BY items fail
before a single data row is read.

All arithmetic and numeric comparison go through :class:`decimal.Decimal`
so results are deterministic and free of binary-float artifacts.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sql import (
    BoolOp,
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


# One resolved join aggregate: its SELECT index, aggregate kind and physical
# ``(side, column_index)`` argument position (and its label for errors).
_JoinAggSpec = Tuple[int, str, Tuple[int, int], str]


class _JoinResolution:
    """Every join-query reference resolved against the two CSV headers.

    Built once from the parsed query and shared by execution and explain so
    both fail on the same unknown/ambiguous columns and membership errors.
    """

    def __init__(
        self,
        left_pos: Tuple[int, int],
        right_pos: Tuple[int, int],
        select_positions: List[Optional[Tuple[int, int]]],
        group_positions: List[Tuple[int, int]],
        agg_specs: List[_JoinAggSpec],
        count_star_index: Optional[int],
        where_ordered: List[Tuple[int, int]],
        where_by_id: Dict[int, Tuple[int, int]],
        order_specs: List[Tuple[int, bool]],
    ) -> None:
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.select_positions = select_positions
        self.group_positions = group_positions
        self.agg_specs = agg_specs
        self.count_star_index = count_star_index
        self.where_ordered = where_ordered
        self.where_by_id = where_by_id
        # Each ORDER BY item already mapped to its SELECT-list position.
        self.order_specs = order_specs


def _join_column_label(table: Optional[str], name: str) -> str:
    return name if table is None else f"{table}.{name}"


def _resolve_join_query(
    query: Query,
    left_table: str,
    right_table: str,
    left_header: Sequence[str],
    right_header: Sequence[str],
) -> _JoinResolution:
    """Resolve SELECT/ON/WHERE/GROUP BY/HAVING/ORDER BY against the headers.

    Runs before any data row is materialized, so unknown columns (KeyError),
    ambiguous/unknown-table references, duplicate aggregates or grouping
    columns, aggregates over a grouping column, and HAVING/ORDER BY items
    absent from the SELECT list all fail identically during execute and
    explain.
    """

    def resolve(table: Optional[str], name: str) -> Tuple[int, int]:
        return _resolve_join_column(
            table, name, left_table, right_table, left_header, right_header
        )

    left_pos, right_pos = _resolve_on(
        query.join_on, left_table, right_table, left_header, right_header
    )

    select = query.select
    select_positions: List[Optional[Tuple[int, int]]] = [None] * len(select)
    agg_specs: List[_JoinAggSpec] = []
    count_star_index: Optional[int] = None
    for idx, item in enumerate(select):
        if item.kind == "count_star":
            count_star_index = idx
        elif item.kind == "column":
            select_positions[idx] = resolve(item.table, item.column)
        else:  # sum/avg/min/max
            position = resolve(item.table, item.column)
            select_positions[idx] = position
            label = _join_column_label(item.table, item.column)
            agg_specs.append((idx, item.kind, position, label))

    group_positions: List[Tuple[int, int]] = []
    if query.is_grouped:
        for table, name in zip(query.group_by_tables, query.group_by):
            group_positions.append(resolve(table, name))
        if len(set(group_positions)) != len(group_positions):
            raise ValueError("GROUP BY may not list the same column twice")
        # The leading SELECT items are the grouping columns in order; they
        # must name the same physical columns as GROUP BY.
        for pos, resolved in zip(group_positions, select_positions):
            if resolved != pos:
                raise ValueError(
                    "SELECT grouping columns must lead the SELECT list in "
                    "the same order as GROUP BY"
                )
        group_set = set(group_positions)
        for _idx, kind, position, _label in agg_specs:
            if position in group_set:
                headers = (left_header, right_header)
                side, col_idx = position
                raise ValueError(
                    f"GROUP BY column {headers[side][col_idx]!r} may not be "
                    "aggregated"
                )

    # Every (kind, physical argument) pair is one distinct aggregate; two
    # SELECT items resolving to the same one are duplicates.
    seen_aggs: set = set()
    for _idx, kind, position, label in agg_specs:
        key = (kind, position)
        if key in seen_aggs:
            raise ValueError(f"duplicate aggregate {kind}({label})")
        seen_aggs.add(key)
    agg_keys = {(kind, position) for _i, kind, position, _l in agg_specs}

    # WHERE columns are resolved (and deduplicated by physical column) in the
    # written order; the id map feeds tree evaluation.
    where_ordered: List[Tuple[int, int]] = []
    where_by_id: Dict[int, Tuple[int, int]] = {}
    if query.where is not None:
        _collect_where_positions_ordered(
            query.where, resolve, where_ordered, set()
        )
        _resolve_where_positions(
            query.where,
            where_by_id,
            left_table,
            right_table,
            left_header,
            right_header,
        )

    if query.having is not None:
        _validate_join_having(query.having, resolve, agg_keys, count_star_index)

    order_specs = _resolve_join_order_by(
        query,
        resolve,
        select_positions,
        group_positions,
        agg_keys,
        count_star_index,
    )

    return _JoinResolution(
        left_pos,
        right_pos,
        select_positions,
        group_positions,
        agg_specs,
        count_star_index,
        where_ordered,
        where_by_id,
        order_specs,
    )


def _validate_join_having(
    node: HavingExpr,
    resolve,
    agg_keys: set,
    count_star_index: Optional[int],
) -> None:
    """Every join HAVING leaf must be an aggregate already in SELECT."""
    if isinstance(node, BoolOp):
        for operand in node.operands:
            _validate_join_having(operand, resolve, agg_keys, count_star_index)
        return
    if node.kind == "count_star":
        if count_star_index is None:
            raise ValueError("HAVING aggregate count(*) is not in the SELECT list")
        return
    position = resolve(node.table, node.column)
    if (node.kind, position) not in agg_keys:
        raise ValueError(f"HAVING aggregate {node.text} is not in the SELECT list")


def _resolve_join_order_by(
    query: Query,
    resolve,
    select_positions: Sequence[Optional[Tuple[int, int]]],
    group_positions: Sequence[Tuple[int, int]],
    agg_keys: set,
    count_star_index: Optional[int],
) -> List[Tuple[int, bool]]:
    """Resolve ORDER BY items to SELECT-list positions for a join query.

    Projection joins match a plain column against the projected positions;
    aggregated joins match it against the grouping columns. Aggregate items
    and count(*) must resolve to the same aggregate selected in SELECT.
    """
    grouped = bool(group_positions)
    aggregate = query.is_aggregate
    specs: List[Tuple[int, bool]] = []
    seen_targets: set = set()
    for sort_item in query.order_by:
        if sort_item.kind == "count_star":
            if count_star_index is None:
                raise ValueError(
                    "ORDER BY aggregate count(*) is not in the SELECT list"
                )
            target: Any = ("count",)
            select_idx = count_star_index
        elif sort_item.kind in _AGG_KINDS:
            position = resolve(sort_item.table, sort_item.column)
            key = (sort_item.kind, position)
            if key not in agg_keys:
                raise ValueError(
                    f"ORDER BY aggregate {sort_item.text} is not in the "
                    "SELECT list"
                )
            target = ("agg",) + key
            select_idx = next(
                i
                for i, item in enumerate(query.select)
                if item.kind == sort_item.kind
                and select_positions[i] == position
            )
        else:  # plain column
            position = resolve(sort_item.table, sort_item.column)
            if aggregate:
                # In an aggregate query the only plain result columns are
                # the grouping columns.
                if position not in group_positions:
                    raise ValueError(
                        f"ORDER BY column {sort_item.text!r} is not in the "
                        "SELECT list"
                    )
            elif position not in select_positions:
                raise ValueError(
                    f"ORDER BY column {sort_item.text!r} is not in the "
                    "SELECT list"
                )
            target = ("column", position)
            select_idx = select_positions.index(position)
        if target in seen_targets:
            raise ValueError(f"duplicate ORDER BY item {sort_item.text!r}")
        seen_targets.add(target)
        specs.append((select_idx, sort_item.descending))
    return specs


def _new_join_agg_state(agg_specs: List[_JoinAggSpec]) -> Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]]:
    """Fresh aggregate state over a joined row stream, keyed per physical col."""
    return {
        (kind, position): (Decimal(0) if kind in ("sum", "avg") else None)
        for _idx, kind, position, _label in agg_specs
    }


def _accumulate_join(
    state: Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]],
    agg_specs: List[_JoinAggSpec],
    pair: Tuple[Sequence[str], Sequence[str]],
) -> None:
    """Roll one filtered joined row into the aggregate state."""
    for _idx, kind, (side, idx), label in agg_specs:
        row = pair[side]
        field = row[idx] if idx < len(row) else ""
        value = _decimal(field, label)
        key = (kind, (side, idx))
        current = state[key]
        if kind in ("sum", "avg"):
            state[key] = current + value
        elif kind == "min":
            if current is None or value < current:
                state[key] = value
        else:  # "max"
            if current is None or value > current:
                state[key] = value


def _finalize_join_values(
    agg_specs: List[_JoinAggSpec],
    state: Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]],
    count: int,
) -> Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]]:
    """Finalize join aggregate state; avg divides its exact sum by count."""
    final: Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]] = {}
    for _idx, kind, position, _label in agg_specs:
        key = (kind, position)
        value = state[key]
        if kind == "avg":
            final[key] = None if count == 0 else value / count
        else:
            final[key] = value
    return final


def _eval_join_having(
    node: HavingExpr,
    count: int,
    values: Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]],
    resolve,
) -> bool:
    """Evaluate a HAVING tree against one computed joined aggregate row.

    As for single-table HAVING, a null aggregate (avg/min/max over an empty
    global input) makes its comparison false and the right side is a
    parser-validated finite decimal literal.
    """
    if isinstance(node, HavingComparison):
        if node.kind == "count_star":
            left: Decimal = Decimal(count)
        else:
            position = resolve(node.table, node.column)
            value = values[(node.kind, position)]
            if value is None:
                return False
            left = value
        return _compare(left, node.op, Decimal(node.value_text))
    if node.op == "AND":
        return all(
            _eval_join_having(operand, count, values, resolve)
            for operand in node.operands
        )
    return any(
        _eval_join_having(operand, count, values, resolve)
        for operand in node.operands
    )


def _join_aggregate_row(
    query: Query,
    resolution: _JoinResolution,
    values: Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]],
    count: int,
    key: Optional[Tuple[Any, ...]] = None,
) -> List[Any]:
    """Build one output row from a join aggregate state.

    A grouped row renders its leading group key values first; a global row
    holds only count(*) and aggregates.
    """
    row: List[Any] = []
    group_count = len(resolution.group_positions)
    for idx, item in enumerate(query.select):
        if item.kind == "count_star":
            row.append(count)
        elif item.kind in _AGG_KINDS:
            value = values[(item.kind, resolution.select_positions[idx])]
            row.append(None if value is None else _number(value))
        elif key is not None:
            row.append(_key_output(key[idx]))
        else:  # pragma: no cover - parser shapes rule plain columns out here
            row.append(None)
    return row


def _run_join(data_dir: str, query: Query) -> Dict[str, Any]:
    """Execute the single supported INNER JOIN query end to end.

    The joined-and-filtered row stream feeds one of three result builders,
    exactly as for single-table queries: a plain projection, one global
    aggregate row (count(*) plus optional sum/avg/min/max), or per-group
    aggregates keyed by the GROUP BY columns.
    """
    left_table = query.table
    right_table = query.join_table
    # Built-in open raises FileNotFoundError, preserving the required type;
    # the left (FROM) table is opened before the right (JOIN) table.
    left_path = Path(data_dir) / f"{left_table}.csv"
    right_path = Path(data_dir) / f"{right_table}.csv"
    left_header, left_rows = _read_table(left_path)
    right_header, right_rows = _read_table(right_path)

    # Every reference is resolved against the headers up front, so all
    # unknown/ambiguous column errors fail before any row is emitted.
    resolution = _resolve_join_query(
        query, left_table, right_table, left_header, right_header
    )
    left_pos, right_pos = resolution.left_pos, resolution.right_pos

    def resolve(table: Optional[str], name: str) -> Tuple[int, int]:
        return _resolve_join_column(
            table, name, left_table, right_table, left_header, right_header
        )

    # Hash the right table by normalized join key; appending in CSV order
    # keeps each bucket in right-table CSV order.
    right_buckets: Dict[Any, List[Sequence[str]]] = {}
    right_idx = right_pos[1]
    for row in right_rows:
        field = row[right_idx] if right_idx < len(row) else ""
        right_buckets.setdefault(_join_key(field), []).append(row)

    # SELECT projection and WHERE positions were resolved up front along
    # with every other reference; resolution failures already surfaced.

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
        mask = _eval_join_where(query.where, pairs, resolution.where_by_id)
        pairs = [pair for pair, matched in zip(pairs, mask) if matched]

    if query.is_grouped:
        result = _run_join_grouped(query, resolution, pairs, resolve)
    elif query.is_aggregate:
        result = _run_join_global(query, resolution, pairs, resolve)
    else:
        result = _run_join_projection(query, resolution, pairs)

    # ORDER BY sorts the fully computed result rows; stability keeps join
    # row order (projection) or first-appearance group order. LIMIT truncates
    # last.
    for select_idx, descending in reversed(resolution.order_specs):
        result["rows"].sort(
            key=lambda row, i=select_idx: _sort_key(row[i]), reverse=descending
        )
    if query.limit is not None:
        _apply_limit(result, query.limit)
    return result


def _run_join_projection(
    query: Query,
    resolution: _JoinResolution,
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
) -> Dict[str, Any]:
    """Project the joined rows in expansion (join) order."""
    positions = resolution.select_positions
    projected_rows = [
        [
            _project_value(pair[side][idx] if idx < len(pair[side]) else "")
            for side, idx in positions
        ]
        for pair in pairs
    ]
    return {
        "columns": [item.text for item in query.select],
        "rows": projected_rows,
        "row_count": len(projected_rows),
    }


def _run_join_global(
    query: Query,
    resolution: _JoinResolution,
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
    resolve,
) -> Dict[str, Any]:
    """Aggregate every filtered joined row into one global row."""
    agg_specs = resolution.agg_specs
    state = _new_join_agg_state(agg_specs)
    count = 0
    for pair in pairs:
        count += 1
        _accumulate_join(state, agg_specs, pair)
    values = _finalize_join_values(agg_specs, state, count)
    result: Dict[str, Any] = {
        "columns": [item.text for item in query.select],
        "rows": [_join_aggregate_row(query, resolution, values, count)],
        "row_count": 1,
    }
    if query.having is not None and not _eval_join_having(
        query.having, count, values, resolve
    ):
        # A global aggregate always computes one row; HAVING failing leaves
        # the same columns but no rows.
        result["rows"] = []
        result["row_count"] = 0
    return result


def _run_join_grouped(
    query: Query,
    resolution: _JoinResolution,
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
    resolve,
) -> Dict[str, Any]:
    """Group the filtered joined rows and aggregate each group."""
    agg_specs = resolution.agg_specs
    group_positions = resolution.group_positions

    # First-appearance order of composite group keys over the filtered stream
    # of joined rows.
    order: List[Tuple[Any, ...]] = []
    group_counts: Dict[Tuple[Any, ...], int] = {}
    group_states: Dict[
        Tuple[Any, ...],
        Dict[Tuple[str, Tuple[int, int]], Optional[Decimal]],
    ] = {}
    for pair in pairs:
        key = tuple(
            _group_key(pair[side][idx] if idx < len(pair[side]) else "")
            for side, idx in group_positions
        )
        if key not in group_counts:
            group_counts[key] = 0
            group_states[key] = _new_join_agg_state(agg_specs)
            order.append(key)
        group_counts[key] += 1
        _accumulate_join(group_states[key], agg_specs, pair)

    # Finalize every group exactly before HAVING or output; each group holds
    # at least one joined row, so its avg/min/max are never null.
    group_values = {
        key: _finalize_join_values(agg_specs, group_states[key], group_counts[key])
        for key in order
    }
    if query.having is not None:
        # HAVING runs after every group has been exactly computed and only
        # drops whole groups; first-appearance order is preserved.
        order = [
            key
            for key in order
            if _eval_join_having(
                query.having, group_counts[key], group_values[key], resolve
            )
        ]
    rows = [
        _join_aggregate_row(
            query, resolution, group_values[key], group_counts[key], key
        )
        for key in order
    ]
    return {
        "columns": [item.text for item in query.select],
        "rows": rows,
        "row_count": len(rows),
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
                {"function": item.kind, "column": item.column, "text": item.text}
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
    """Plan the single INNER JOIN query; only the two CSV headers are read.

    Projection joins and the newer global/grouped aggregate joins share one
    resolver with execution, so GROUP BY/HAVING/aggregate/ORDER BY references
    fail with the same KeyError/ValueError without a single data row read.
    """
    left_table = query.table
    right_table = query.join_table
    # The left (FROM) table is opened before the right (JOIN) table, just as
    # in _run_join; built-in open raises FileNotFoundError.
    left_header = _read_header(Path(data_dir) / f"{left_table}.csv")
    right_header = _read_header(Path(data_dir) / f"{right_table}.csv")

    resolution = _resolve_join_query(
        query, left_table, right_table, left_header, right_header
    )
    left_pos, right_pos = resolution.left_pos, resolution.right_pos

    headers = (left_header, right_header)
    tables = (left_table, right_table)

    def qualified(position: Tuple[int, int]) -> str:
        side, idx = position
        return f"{tables[side]}.{headers[side][idx]}"

    # Every column the join actually materializes: the two ON keys, the
    # WHERE columns, the grouping columns, the aggregate argument columns and
    # (for projection joins) the SELECT columns, left table first in each
    # header's column order, then the right table.
    scan_set = {left_pos, right_pos}
    scan_set.update(resolution.where_ordered)
    scan_set.update(resolution.group_positions)
    for _idx, _kind, position, _label in resolution.agg_specs:
        scan_set.add(position)
    for position in resolution.select_positions:
        if position is not None:
            scan_set.add(position)
    scan_columns = [
        qualified((side, idx))
        for side in (0, 1)
        for idx in sorted(p[1] for p in scan_set if p[0] == side)
    ]

    where_columns = [qualified(position) for position in resolution.where_ordered]
    group_columns = [qualified(position) for position in resolution.group_positions]

    # SELECT keeps the source text of each plain/grouping column; aggregate
    # expressions are reported separately in "aggregates".
    select_columns: List[str] = []
    seen_select: set = set()
    for item in query.select:
        if item.kind == "column" and item.text not in seen_select:
            seen_select.add(item.text)
            select_columns.append(item.text)

    # Join aggregates name their physical argument column qualified, while
    # "text" preserves the expression exactly as written (so sum(amt) and
    # sum(orders.amt) resolving to the same column are both distinguishable
    # in text but identical in function+column).
    agg_position_by_idx = {
        idx: position for idx, _kind, position, _label in resolution.agg_specs
    }
    aggregates: List[Dict[str, Any]] = []
    for idx, item in enumerate(query.select):
        if item.kind == "count_star":
            aggregates.append(
                {"function": "count", "column": None, "text": item.text}
            )
        elif item.kind in _AGG_KINDS:
            aggregates.append(
                {
                    "function": item.kind,
                    "column": qualified(agg_position_by_idx[idx]),
                    "text": item.text,
                }
            )

    return {
        "query_type": "join",
        "tables": [left_table, right_table],
        "scan_columns": scan_columns,
        "where_columns": where_columns,
        "select_columns": select_columns,
        "group_columns": group_columns,
        "aggregates": aggregates,
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
