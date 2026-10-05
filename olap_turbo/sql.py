"""Minimal self-contained SQL lexer and parser.

Supported grammar (the complete surface for this baseline)::

    query      := SELECT select_list FROM identifier
                  [ INNER JOIN identifier ON join_on ]
                  [ WHERE boolean ] [ GROUP BY identifier (, identifier)* ]
                  [ HAVING having_boolean ]
                  [ ORDER BY sort_item (, sort_item)* ] [ LIMIT integer ]
    join_on    := identifier . identifier = identifier . identifier
    sort_item  := (identifier | count_star | agg_expr) [ ASC | DESC ]
    agg_expr   := sum_expr | avg_expr | min_expr | max_expr
    select_list:= grouping_column (, grouping_column)*,
                  count_star, agg_expr (, agg_expr)*   (grouped aggregate query)
                | count_star [, agg_expr ]*            (aggregate query)
                | column (, column)*                   (projection query)
    boolean    := or_expr
    or_expr    := and_expr (OR and_expr)*
    and_expr   := factor (AND factor)*
    factor     := LPAREN or_expr RPAREN | comparison
    comparison := identifier op value
    op         := = | != | > | >= | < | <=
    value      := number-literal | double-quoted-string-literal

HAVING shares the AND/OR/parentheses boolean shape, but its leaves are
aggregate-to-number comparisons::

    having_boolean := having_or
    having_factor  := LPAREN having_or RPAREN | having_comparison
    having_comparison := aggregate op number-literal
    aggregate      := count_star | agg_expr

The left side must be a ``count(*)``, ``sum(column)``, ``avg(column)``,
``min(column)`` or ``max(column)`` expression that already appears in the
SELECT list; the right side is an unquoted finite decimal literal only (no
strings, no columns, no other expressions).

A query may name at most one INNER JOIN, in the fixed position directly
after the FROM table and before WHERE::

    SELECT item, item FROM left INNER JOIN right ON left.a = right.b
    [WHERE ...] [ORDER BY ...] [LIMIT n]

There are no aliases, no LEFT JOIN, no multiple joins, and a join query is a
plain projection only: aggregates/GROUP BY/HAVING are rejected. Column
references in the SELECT list, ON and WHERE may be table-qualified
(``table.column``); an unqualified column is accepted only when it resolves
uniquely across the two tables. ON must be exactly one equality of one column
from each table: literals, two columns of the same table, any operator other
than ``=`` and combined/duplicate conditions are rejected.

AND binds tighter than OR; parentheses may be nested to override the
precedence. Parentheses *combine* boolean conditions only: a parenthesized
group must itself contain an AND/OR combination (``(a = 1)`` is rejected),
and parentheses may never wrap columns, values or comparisons-as-values.
There is no NOT, no constant truth value and no other expression forms.

HAVING may appear at most once and only after GROUP BY (or after WHERE in
an ungrouped aggregate query); it is rejected on plain projection queries.

ORDER BY follows HAVING (or WHERE/GROUP BY when HAVING is absent) and LIMIT
follows ORDER BY; each may appear at most once and never out of order. Each
sort item references a result column already in the SELECT list — a plain or
grouping column, ``count(*)`` or one of ``sum(column)``, ``avg(column)``,
``min(column)``, ``max(column)`` — with an optional ``ASC`` (default) or
``DESC`` direction; duplicate sort items and references to expressions
absent from the SELECT list are rejected. LIMIT takes a single non-negative
decimal integer literal (digits only: no sign, no fraction, no strings,
columns or expressions).

A grouped query names one or more distinct grouping columns (at least two
for the multi-column shape): they must head the SELECT list in the same
order as the GROUP BY list, followed by count(*) and one or more distinct
aggregate expressions (``sum``/``avg``/``min``/``max`` over a column).
Every aggregate expression — the function name plus its column — may appear
at most once in a SELECT list, in grouped and ungrouped queries alike.

Anything else (missing FROM, NOT, functions other than
count(*)/sum()/avg()/min()/max() in SELECT, wildcards, extra clauses,
malformed syntax, unmatched parentheses, OR outside WHERE/HAVING) raises
:class:`ValueError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union


class _TokKind(Enum):
    IDENT = "ident"
    KEYWORD = "keyword"
    NUMBER = "number"
    STRING = "string"
    OP = "op"
    COMMA = "comma"
    DOT = "dot"
    STAR = "star"
    LPAREN = "lparen"
    RPAREN = "rparen"
    EOF = "eof"


_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "GROUP", "BY", "HAVING",
    "ORDER", "LIMIT", "INNER", "JOIN", "ON",
}


@dataclass
class _Token:
    kind: _TokKind
    text: str


def _tokenize(sql: str) -> List[_Token]:
    tokens: List[_Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == ",":
            tokens.append(_Token(_TokKind.COMMA, c))
            i += 1
            continue
        if c == ".":
            tokens.append(_Token(_TokKind.DOT, c))
            i += 1
            continue
        if c == "(":
            tokens.append(_Token(_TokKind.LPAREN, c))
            i += 1
            continue
        if c == ")":
            tokens.append(_Token(_TokKind.RPAREN, c))
            i += 1
            continue
        if c == "*":
            tokens.append(_Token(_TokKind.STAR, c))
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf: List[str] = []
            while j < n and sql[j] != '"':
                buf.append(sql[j])
                j += 1
            if j >= n:
                raise ValueError("unterminated string literal in query")
            tokens.append(_Token(_TokKind.STRING, "".join(buf)))
            i = j + 1
            continue
        if c in "=<>":
            if i + 1 < n and sql[i + 1] == "=":
                tokens.append(_Token(_TokKind.OP, c + "="))
                i += 2
            else:
                tokens.append(_Token(_TokKind.OP, c))
                i += 1
            continue
        if c == "!":
            if i + 1 < n and sql[i + 1] == "=":
                tokens.append(_Token(_TokKind.OP, "!="))
                i += 2
                continue
            raise ValueError(f"unexpected character '!' at position {i}")
        if c.isdigit() or (c == "-" and i + 1 < n and sql[i + 1].isdigit()):
            j = i + 1
            while j < n and (sql[j].isdigit() or sql[j] == "."):
                j += 1
            text = sql[i:j]
            # Reject malformed numbers such as "1.2.3".
            _parse_decimal(text)
            tokens.append(_Token(_TokKind.NUMBER, text))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            text = sql[i:j]
            upper = text.upper()
            kind = _TokKind.KEYWORD if upper in _KEYWORDS else _TokKind.IDENT
            tokens.append(_Token(kind, upper if kind is _TokKind.KEYWORD else text))
            i = j
            continue
        raise ValueError(f"unexpected character {c!r} at position {i}")
    tokens.append(_Token(_TokKind.EOF, ""))
    return tokens


def _parse_decimal(text: str):
    """Validate a decimal numeric literal; return a Decimal-like float-free value."""
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(text)
    except InvalidOperation:
        raise ValueError(f"invalid numeric literal {text!r}")


@dataclass(frozen=True)
class ColumnRef:
    """A column reference, possibly table-qualified (``table.column``).

    ``table`` is the qualifier text as written (``None`` for an unqualified
    reference); ``name`` is the bare column name; ``text`` is the normalized
    reference text used verbatim in the output ``columns`` list.
    """

    name: str
    table: Optional[str] = None
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            object.__setattr__(
                self,
                "text",
                self.name if self.table is None else f"{self.table}.{self.name}",
            )


@dataclass(frozen=True)
class JoinOn:
    """The single ON equality of a join: one column from each table.

    The two sides are stored in FROM order: ``left`` names a column of the
    left (FROM) table and ``right`` a column of the JOIN table, regardless of
    the order written.
    """

    left: ColumnRef
    right: ColumnRef


@dataclass(frozen=True)
class Comparison:
    # Bare column name. For single-table queries this is the full reference;
    # join queries additionally carry the optional table qualifier.
    column: str
    op: str
    # value_text is the raw literal; quoted distinguishes text vs numeric.
    value_text: str
    quoted: bool
    # Table qualifier of the column (join queries only); None if bare.
    table: Optional[str] = None


@dataclass(frozen=True)
class BoolOp:
    """A conjunction or disjunction of boolean conditions.

    ``op`` is ``"AND"`` or ``"OR"``; operands are leaf comparisons
    (:class:`Comparison` for WHERE, :class:`HavingComparison` for HAVING)
    or further :class:`BoolOp` nodes.
    """

    op: str
    operands: Tuple["BooleanNode", ...]


WhereExpr = Union[Comparison, BoolOp]


@dataclass(frozen=True)
class HavingComparison:
    """One HAVING leaf: aggregate op finite-decimal-literal.

    ``kind`` is ``"count_star"`` or one of ``"sum"``, ``"avg"``, ``"min"``,
    ``"max"`` (with ``column`` naming the aggregated column); ``text`` is
    the aggregate's SELECT expression text.
    """

    text: str
    kind: str
    op: str
    value_text: str
    column: Optional[str] = None


HavingExpr = Union[HavingComparison, BoolOp]

BooleanNode = Union[Comparison, HavingComparison, BoolOp]


@dataclass(frozen=True)
class SelectItem:
    # Original expression text as written in the query, e.g. "count(*)",
    # "sum(a)", "avg(a)", "a" or "left.a".
    text: str
    kind: str  # "column" | "count_star" | "sum" | "avg" | "min" | "max"
    column: Optional[str] = None
    # Table qualifier of a plain column reference (join queries only).
    table: Optional[str] = None


@dataclass(frozen=True)
class SortItem:
    """One ORDER BY item: a result-column reference with a direction.

    ``text`` is the normalized expression text (``a``, ``left.a``,
    ``count(*)``, ``sum(a)``, ``avg(a)``); ``kind``/``column``/``table``
    mirror :class:`SelectItem`.
    """

    text: str
    kind: str  # "column" | "count_star" | "sum" | "avg" | "min" | "max"
    column: Optional[str] = None
    descending: bool = False
    # Table qualifier of a plain column reference (join queries only).
    table: Optional[str] = None


@dataclass(frozen=True)
class Query:
    select: Tuple[SelectItem, ...]
    table: str
    where: Optional[WhereExpr] = None
    group_by: Tuple[str, ...] = ()
    having: Optional[HavingExpr] = None
    order_by: Tuple[SortItem, ...] = ()
    limit: Optional[int] = None
    join_table: Optional[str] = None
    join_on: Optional[JoinOn] = None

    @property
    def is_aggregate(self) -> bool:
        return bool(self.group_by) or self.select[0].kind == "count_star"

    @property
    def is_grouped(self) -> bool:
        return bool(self.group_by)

    @property
    def is_join(self) -> bool:
        return self.join_table is not None

    def required_columns(self) -> Tuple[str, ...]:
        """Columns the single-table scan must materialize.

        Join queries use :meth:`required_join_columns` instead.
        """
        cols: List[str] = []
        seen = set()
        for item in self.select:
            if item.kind == "column":
                if item.column not in seen:
                    seen.add(item.column)
                    cols.append(item.column)
            elif item.kind in _AGG_KINDS and item.column not in seen:
                seen.add(item.column)
                cols.append(item.column)
        if self.where is not None:
            self._collect_where_columns(self.where, cols, seen)
        return tuple(cols)

    @staticmethod
    def _collect_where_columns(
        node: WhereExpr, cols: List[str], seen: set
    ) -> None:
        if isinstance(node, Comparison):
            if node.column not in seen:
                seen.add(node.column)
                cols.append(node.column)
            return
        for operand in node.operands:
            Query._collect_where_columns(operand, cols, seen)


_OPS = {"=", "!=", ">", ">=", "<", "<="}

# Column aggregates over finite decimals, besides count(*).
_AGG_KINDS = ("sum", "avg", "min", "max")


class _Parser:
    def __init__(self, sql: str):
        self.tokens = _tokenize(sql)
        self.pos = 0

    def _peek(self) -> _Token:
        return self.tokens[self.pos]

    def _next(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect_keyword(self, word: str) -> None:
        tok = self._next()
        if tok.kind is not _TokKind.KEYWORD or tok.text != word:
            raise ValueError(f"expected {word} but found {tok.text!r}")

    def _accept_keyword(self, word: str) -> bool:
        tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == word:
            self._next()
            return True
        return False

    def parse(self) -> Query:
        self._expect_keyword("SELECT")
        items = self._parse_select_list()
        self._expect_keyword("FROM")
        table_tok = self._next()
        if table_tok.kind is not _TokKind.IDENT:
            raise ValueError("table name must be a simple identifier")
        join_table: Optional[str] = None
        join_on: Optional[JoinOn] = None
        tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "INNER":
            join_table, join_on = self._parse_join(table_tok.text)
            tok = self._peek()
        where: Optional[WhereExpr] = None
        having: Optional[HavingExpr] = None
        group_by: List[str] = []
        order_by: List[SortItem] = []
        limit: Optional[int] = None
        if tok.kind is _TokKind.KEYWORD and tok.text == "WHERE":
            self._next()
            where = self._parse_or(join=True if join_table is not None else False)
            tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "GROUP":
            if join_table is not None:
                raise ValueError("GROUP BY is not supported in a join query")
            self._next()
            self._expect_keyword("BY")
            group_by = [self._expect(_TokKind.IDENT).text]
            while self._peek().kind is _TokKind.COMMA:
                self._next()
                group_by.append(self._expect(_TokKind.IDENT).text)
            self._validate_grouped(items, group_by)
            tok = self._peek()
        elif join_table is not None:
            self._validate_join(items)
        else:
            self._validate_ungrouped(items)
        if tok.kind is _TokKind.KEYWORD and tok.text == "HAVING":
            if join_table is not None:
                raise ValueError("HAVING is not supported in a join query")
            self._next()
            having = self._parse_having_or()
            if not (bool(group_by) or items[0].kind == "count_star"):
                raise ValueError("HAVING is only valid on aggregate queries")
            self._validate_having_aggregates(having, items)
            tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "ORDER":
            self._next()
            self._expect_keyword("BY")
            order_by = [self._parse_sort_item(items, join_table is not None)]
            while self._peek().kind is _TokKind.COMMA:
                self._next()
                order_by.append(self._parse_sort_item(items, join_table is not None))
            seen_items = set()
            for sort_item in order_by:
                key = (sort_item.table, sort_item.kind, sort_item.column)
                if key in seen_items:
                    raise ValueError(
                        f"duplicate ORDER BY item {sort_item.text!r}"
                    )
                seen_items.add(key)
            tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "LIMIT":
            self._next()
            limit = self._parse_limit()
            tok = self._peek()
        if tok.kind is not _TokKind.EOF:
            # Trailing tokens: an unsupported clause, a second ORDER BY or
            # LIMIT, a clause out of order, an unmatched ')' or plain
            # malformed input.
            raise ValueError(f"unsupported or unexpected token {tok.text!r}")
        self._validate_qualifiers(
            items, where, order_by, table_tok.text, join_table
        )
        return Query(
            tuple(items),
            table_tok.text,
            where,
            tuple(group_by),
            having,
            tuple(order_by),
            limit,
            join_table,
            join_on,
        )

    # ----- INNER JOIN -----

    def _parse_join(self, left_table: str) -> Tuple[str, JoinOn]:
        """Parse ``INNER JOIN right ON left.a = right.b`` after FROM.

        Only exactly one INNER JOIN is supported: no aliases, no LEFT/RIGHT
        joins, no second JOIN, and ON must be one equality of one column from
        each table.
        """
        self._expect_keyword("INNER")
        self._expect_keyword("JOIN")
        right_tok = self._next()
        if right_tok.kind is not _TokKind.IDENT:
            raise ValueError("joined table name must be a simple identifier")
        if right_tok.text == left_table:
            raise ValueError(
                f"join tables must have distinct names: {left_table!r} and "
                f"{right_tok.text!r} are the same table"
            )
        self._expect_keyword("ON")
        first_ref = self._parse_column_ref()
        op_tok = self._expect(_TokKind.OP)
        if op_tok.text != "=":
            raise ValueError("ON must be a single column equality using '='")
        second_ref = self._parse_column_ref()
        tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text in (
            "AND", "OR", "INNER", "JOIN", "ON",
        ):
            raise ValueError(
                "ON must be exactly one cross-table column equality, with no "
                "combined conditions or further joins"
            )
        tables = {left_table, right_tok.text}
        # Qualified sides are checked here; bare sides are resolved against
        # the two headers at execution time.
        for ref in (first_ref, second_ref):
            if ref.table is not None and ref.table not in tables:
                raise ValueError(
                    f"ON column {ref.text!r} is not qualified by either "
                    "joined table"
                )
        if (
            first_ref.table is not None
            and second_ref.table is not None
            and first_ref.table == second_ref.table
        ):
            raise ValueError(
                "ON must compare one column from each table, not two columns "
                "of the same table"
            )
        # Bare ON columns are resolved to one side each at execution, once
        # the two headers are known; store the sides in written order.
        return right_tok.text, JoinOn(first_ref, second_ref)

    def _parse_column_ref(self) -> ColumnRef:
        """Parse ``identifier`` or ``identifier . identifier``."""
        first = self._expect(_TokKind.IDENT)
        if self._peek().kind is _TokKind.DOT:
            self._next()
            second = self._expect(_TokKind.IDENT)
            return ColumnRef(second.text, first.text, f"{first.text}.{second.text}")
        return ColumnRef(first.text)

    def _validate_join(self, items: List[SelectItem]) -> None:
        """A join query is a plain projection: no aggregates at all."""
        for item in items:
            if item.kind != "column":
                raise ValueError(
                    "aggregate SELECT expressions are not supported in a "
                    "join query"
                )

    def _validate_qualifiers(
        self,
        items: List[SelectItem],
        where: Optional[WhereExpr],
        order_by: List[SortItem],
        left_table: str,
        join_table: Optional[str],
    ) -> None:
        """Check every ``table.column`` qualifier names a real query table."""

        def check(ref_text: str, table: Optional[str]) -> None:
            if table is None:
                return
            if join_table is None:
                raise ValueError(
                    f"table-qualified column {ref_text!r} is only valid in a "
                    "join query"
                )
            if table not in (left_table, join_table):
                raise ValueError(
                    f"column {ref_text!r} is qualified by unknown table "
                    f"{table!r}"
                )

        for item in items:
            check(item.text, item.table)
        for sort_item in order_by:
            check(sort_item.text, sort_item.table)
        if where is not None:
            self._check_where_qualifiers(where, check)

    def _check_where_qualifiers(self, node: WhereExpr, check) -> None:
        if isinstance(node, Comparison):
            if node.table is not None:
                check(f"{node.table}.{node.column}", node.table)
            return
        for operand in node.operands:
            self._check_where_qualifiers(operand, check)

    def _validate_grouped(
        self, items: List[SelectItem], group_by: List[str]
    ) -> None:
        """Enforce the grouped shape.

        SELECT leads with the GROUP BY columns in the same order (distinct),
        followed by count(*) and one or more distinct aggregate expressions
        (sum/avg/min/max over a non-grouping column).
        """
        if len(group_by) != len(set(group_by)):
            raise ValueError("GROUP BY may not list the same column twice")
        if len(items) < len(group_by) + 2:
            raise ValueError(
                "grouped SELECT must list the grouping columns followed by "
                "count(*) and one or more aggregate expressions"
            )
        for pos, group_col in enumerate(group_by):
            item = items[pos]
            if item.kind != "column" or item.column != group_col:
                raise ValueError(
                    "SELECT grouping columns must lead the SELECT list in "
                    "the same order as GROUP BY"
                )
        if items[len(group_by)].kind != "count_star":
            raise ValueError(
                "grouped SELECT must list count(*) after the grouping columns"
            )
        seen_aggs = set()
        for item in items[len(group_by) + 1 :]:
            if item.kind not in _AGG_KINDS:
                raise ValueError(
                    "grouped SELECT may only contain the grouping columns, "
                    "count(*) and sum/avg/min/max aggregates"
                )
            key = (item.kind, item.column)
            if key in seen_aggs:
                raise ValueError(f"duplicate aggregate {item.text}")
            if item.column in group_by:
                raise ValueError(
                    f"GROUP BY column {item.column} may not be aggregated"
                )
            seen_aggs.add(key)

    def _parse_select_list(self) -> List[SelectItem]:
        items = [self._parse_select_item()]
        while self._peek().kind is _TokKind.COMMA:
            self._next()
            items.append(self._parse_select_item())
        return items

    def _validate_ungrouped(self, items: List[SelectItem]) -> None:
        """Rules for queries without GROUP BY (the baseline shapes)."""
        if items[0].kind in _AGG_KINDS:
            raise ValueError(
                f"{items[0].kind}(column) requires count(*) in the SELECT list"
            )
        aggregate = items[0].kind == "count_star"
        seen_aggs = set()
        for item in items[1:]:
            if aggregate:
                if item.kind not in _AGG_KINDS:
                    raise ValueError(
                        "aggregate SELECT may only contain count(*) and "
                        "sum/avg/min/max aggregates"
                    )
                key = (item.kind, item.column)
                if key in seen_aggs:
                    raise ValueError(f"duplicate aggregate {item.text}")
                seen_aggs.add(key)
            elif item.kind in _AGG_KINDS:
                raise ValueError(
                    f"{item.kind}(column) requires count(*) in the SELECT list"
                )
            elif item.kind != "column":
                raise ValueError(
                    "count(*)/aggregates cannot be mixed with plain columns"
                )

    def _parse_select_item(self) -> SelectItem:
        tok = self._peek()
        # function form: ident '(' ... ')'
        if tok.kind is _TokKind.IDENT and self.tokens[self.pos + 1].kind is _TokKind.LPAREN:
            name_tok = self._next()
            self._next()  # (
            inner = self._next()
            self._expect(_TokKind.RPAREN)
            upper = name_tok.text.upper()
            if upper == "COUNT" and inner.kind is _TokKind.STAR:
                return SelectItem("count(*)", "count_star")
            if upper in ("SUM", "AVG", "MIN", "MAX") and inner.kind is _TokKind.IDENT:
                kind = upper.lower()
                return SelectItem(f"{kind}({inner.text})", kind, inner.text)
            # Any other function (count(col), median, ...) is unsupported.
            raise ValueError(
                f"unsupported SELECT expression {name_tok.text}({inner.text})"
            )
        if tok.kind is _TokKind.STAR:
            raise ValueError("wildcard column '*' is not supported")
        if tok.kind is _TokKind.IDENT:
            ref = self._parse_column_ref()
            return SelectItem(ref.text, "column", ref.name, ref.table)
        raise ValueError(f"invalid SELECT item {tok.text!r}")

    def _expect(self, kind: _TokKind) -> _Token:
        tok = self._next()
        if tok.kind is not kind:
            raise ValueError(f"expected {kind.value} but found {tok.text!r}")
        return tok

    # ----- WHERE boolean grammar: or_expr := and_expr (OR and_expr)* -----

    def _parse_or(self, join: bool = False) -> WhereExpr:
        operands = [self._parse_and(join)]
        while self._accept_keyword("OR"):
            operands.append(self._parse_and(join))
        if len(operands) == 1:
            return operands[0]
        return BoolOp("OR", tuple(operands))

    def _parse_and(self, join: bool = False) -> WhereExpr:
        operands = [self._parse_factor(join)]
        while self._accept_keyword("AND"):
            operands.append(self._parse_factor(join))
        if len(operands) == 1:
            return operands[0]
        return BoolOp("AND", tuple(operands))

    def _parse_factor(self, join: bool = False) -> WhereExpr:
        tok = self._peek()
        if tok.kind is _TokKind.LPAREN:
            self._next()
            node = self._parse_or(join)
            self._expect(_TokKind.RPAREN)
            if isinstance(node, Comparison):
                # Parentheses only *combine* boolean conditions; they may
                # not wrap a single comparison, a column or a value.
                raise ValueError(
                    "parentheses may only group combined AND/OR conditions, "
                    "not a single comparison"
                )
            return node
        if tok.kind is _TokKind.IDENT:
            return self._parse_comparison(join)
        # NOT, dangling AND/OR, ')', EOF, numbers, ...: none can start a
        # boolean condition.
        raise ValueError(f"expected a boolean condition but found {tok.text!r}")

    def _parse_comparison(self, join: bool = False) -> Comparison:
        ref = self._parse_column_ref()
        if ref.table is not None and not join:
            raise ValueError(
                f"table-qualified column {ref.text!r} is only valid in a "
                "join query"
            )
        op_tok = self._expect(_TokKind.OP)
        if op_tok.text not in _OPS:
            raise ValueError(f"unsupported operator {op_tok.text!r}")
        val = self._next()
        if val.kind is _TokKind.NUMBER:
            return Comparison(ref.name, op_tok.text, val.text, False, ref.table)
        if val.kind is _TokKind.STRING:
            return Comparison(ref.name, op_tok.text, val.text, True, ref.table)
        raise ValueError(
            "comparison value must be an unquoted number or a double-quoted string"
        )

    # ----- HAVING boolean grammar: aggregate op number, AND/OR/parens -----

    def _parse_having_or(self) -> HavingExpr:
        operands = [self._parse_having_and()]
        while self._accept_keyword("OR"):
            operands.append(self._parse_having_and())
        if len(operands) == 1:
            return operands[0]
        return BoolOp("OR", tuple(operands))

    def _parse_having_and(self) -> HavingExpr:
        operands = [self._parse_having_factor()]
        while self._accept_keyword("AND"):
            operands.append(self._parse_having_factor())
        if len(operands) == 1:
            return operands[0]
        return BoolOp("AND", tuple(operands))

    def _parse_having_factor(self) -> HavingExpr:
        tok = self._peek()
        if tok.kind is _TokKind.LPAREN:
            self._next()
            node = self._parse_having_or()
            self._expect(_TokKind.RPAREN)
            if isinstance(node, HavingComparison):
                # As in WHERE, parentheses only combine conditions; they
                # may not wrap a single HAVING comparison.
                raise ValueError(
                    "parentheses may only group combined AND/OR conditions, "
                    "not a single comparison"
                )
            return node
        if (
            tok.kind is _TokKind.IDENT
            and self.tokens[self.pos + 1].kind is _TokKind.LPAREN
        ):
            return self._parse_having_comparison()
        # Plain columns/grouping columns, NOT, numbers, dangling AND/OR,
        # ')', EOF, ...: none can start a HAVING aggregate comparison.
        raise ValueError(
            "HAVING condition must compare count(*) or a "
            "sum/avg/min/max aggregate with a number, but found "
            f"{tok.text!r}"
        )

    def _parse_having_comparison(self) -> HavingComparison:
        name_tok = self._next()  # the aggregate identifier
        self._next()  # LPAREN
        inner = self._next()
        self._expect(_TokKind.RPAREN)
        upper = name_tok.text.upper()
        if upper == "COUNT" and inner.kind is _TokKind.STAR:
            text, kind, column = "count(*)", "count_star", None
        elif upper in ("SUM", "AVG", "MIN", "MAX") and inner.kind is _TokKind.IDENT:
            kind = upper.lower()
            text, column = f"{kind}({inner.text})", inner.text
        else:
            raise ValueError(
                "HAVING may only compare count(*) or sum/avg/min/max "
                "aggregates"
            )
        op_tok = self._expect(_TokKind.OP)
        if op_tok.text not in _OPS:
            raise ValueError(f"unsupported operator {op_tok.text!r}")
        val = self._next()
        if val.kind is not _TokKind.NUMBER:
            # Quoted strings, columns, aggregates, keywords, EOF: HAVING
            # compares aggregates against finite decimal literals only.
            raise ValueError(
                "HAVING comparison value must be an unquoted finite decimal number"
            )
        return HavingComparison(text, kind, op_tok.text, val.text, column)

    def _validate_having_aggregates(
        self, node: HavingExpr, items: List[SelectItem]
    ) -> None:
        """Every HAVING aggregate must already appear in the SELECT list."""
        if isinstance(node, BoolOp):
            for operand in node.operands:
                self._validate_having_aggregates(operand, items)
            return
        for item in items:
            if item.kind == node.kind and item.column == node.column:
                return
        raise ValueError(
            f"HAVING aggregate {node.text} is not in the SELECT list"
        )

    # ----- ORDER BY / LIMIT -----

    def _parse_sort_item(self, items: List[SelectItem], join: bool = False) -> SortItem:
        tok = self._peek()
        if (
            tok.kind is _TokKind.IDENT
            and self.tokens[self.pos + 1].kind is _TokKind.LPAREN
        ):
            if join:
                raise ValueError(
                    "ORDER BY aggregates are not supported in a join query"
                )
            name_tok = self._next()
            self._next()  # (
            inner = self._next()
            self._expect(_TokKind.RPAREN)
            upper = name_tok.text.upper()
            if upper == "COUNT" and inner.kind is _TokKind.STAR:
                text, kind, column, table = "count(*)", "count_star", None, None
            elif upper in ("SUM", "AVG", "MIN", "MAX") and inner.kind is _TokKind.IDENT:
                kind = upper.lower()
                text, column, table = f"{kind}({inner.text})", inner.text, None
            else:
                raise ValueError(
                    "ORDER BY may only reference result columns, count(*) "
                    "or sum/avg/min/max aggregates"
                )
        elif tok.kind is _TokKind.IDENT:
            ref = self._parse_column_ref()
            if ref.table is not None and not join:
                raise ValueError(
                    f"table-qualified column {ref.text!r} is only valid in a "
                    "join query"
                )
            text, kind, column, table = ref.text, "column", ref.name, ref.table
        else:
            # Numbers, strings, keywords, '*', parens, EOF: none can start
            # a sort item.
            raise ValueError(f"invalid ORDER BY item {tok.text!r}")
        # ASC/DESC are contextual: they stay plain identifiers so a column
        # named e.g. "asc" remains usable elsewhere in the grammar.
        descending = False
        nxt = self._peek()
        if nxt.kind is _TokKind.IDENT and nxt.text.upper() in ("ASC", "DESC"):
            self._next()
            descending = nxt.text.upper() == "DESC"
        sort_item = SortItem(text, kind, column, descending, table)
        # Join queries match ORDER BY items to result columns after bare
        # references have been resolved against the two headers (execution
        # time); single-table queries validate against the SELECT list here.
        if not join:
            self._validate_sort_item(sort_item, items)
        return sort_item

    def _validate_sort_item(
        self, sort_item: SortItem, items: List[SelectItem]
    ) -> None:
        """Every ORDER BY item must reference a column of the result."""
        for item in items:
            if (
                item.kind == sort_item.kind
                and item.column == sort_item.column
                and item.table == sort_item.table
            ):
                return
        if sort_item.kind == "column":
            raise ValueError(
                f"ORDER BY column {sort_item.text!r} is not in the SELECT list"
            )
        raise ValueError(
            f"ORDER BY aggregate {sort_item.text} is not in the SELECT list"
        )

    def _parse_limit(self) -> int:
        tok = self._next()
        if (
            tok.kind is not _TokKind.NUMBER
            or not tok.text.isascii()
            or not tok.text.isdigit()
        ):
            # Negatives, fractions, strings, columns, expressions, missing
            # values: LIMIT takes digits only.
            raise ValueError(
                "LIMIT must be followed by a non-negative decimal integer"
            )
        return int(tok.text)


def parse_sql(sql: str) -> Query:
    """Parse a query string into a :class:`Query`."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("empty query")
    return _Parser(sql).parse()
