"""Minimal self-contained SQL lexer and parser.

Supported grammar (the complete surface for this baseline)::

    query      := SELECT select_list FROM identifier
                  [ WHERE boolean ] [ GROUP BY identifier (, identifier)* ]
                  [ HAVING having_boolean ]
                  [ ORDER BY sort_item (, sort_item)* ] [ LIMIT integer ]
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
    STAR = "star"
    LPAREN = "lparen"
    RPAREN = "rparen"
    EOF = "eof"


_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "GROUP", "BY", "HAVING",
    "ORDER", "LIMIT",
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
class Comparison:
    column: str
    op: str
    # value_text is the raw literal; quoted distinguishes text vs numeric.
    value_text: str
    quoted: bool


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
    # "sum(a)", "avg(a)".
    text: str
    kind: str  # "column" | "count_star" | "sum" | "avg" | "min" | "max"
    column: Optional[str] = None


@dataclass(frozen=True)
class SortItem:
    """One ORDER BY item: a result-column reference with a direction.

    ``text`` is the normalized expression text (``a``, ``count(*)``,
    ``sum(a)``, ``avg(a)``); ``kind``/``column`` mirror :class:`SelectItem`.
    """

    text: str
    kind: str  # "column" | "count_star" | "sum" | "avg" | "min" | "max"
    column: Optional[str] = None
    descending: bool = False


@dataclass(frozen=True)
class Query:
    select: Tuple[SelectItem, ...]
    table: str
    where: Optional[WhereExpr] = None
    group_by: Tuple[str, ...] = ()
    having: Optional[HavingExpr] = None
    order_by: Tuple[SortItem, ...] = ()
    limit: Optional[int] = None

    @property
    def is_aggregate(self) -> bool:
        return bool(self.group_by) or self.select[0].kind == "count_star"

    @property
    def is_grouped(self) -> bool:
        return bool(self.group_by)

    def required_columns(self) -> Tuple[str, ...]:
        """Columns the scan must materialize (group key, aggregates, predicates)."""
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
        where: Optional[WhereExpr] = None
        having: Optional[HavingExpr] = None
        group_by: List[str] = []
        order_by: List[SortItem] = []
        limit: Optional[int] = None
        tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "WHERE":
            self._next()
            where = self._parse_or()
            tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "GROUP":
            self._next()
            self._expect_keyword("BY")
            group_by = [self._expect(_TokKind.IDENT).text]
            while self._peek().kind is _TokKind.COMMA:
                self._next()
                group_by.append(self._expect(_TokKind.IDENT).text)
            self._validate_grouped(items, group_by)
            tok = self._peek()
        else:
            self._validate_ungrouped(items)
        if tok.kind is _TokKind.KEYWORD and tok.text == "HAVING":
            self._next()
            having = self._parse_having_or()
            if not (bool(group_by) or items[0].kind == "count_star"):
                raise ValueError("HAVING is only valid on aggregate queries")
            self._validate_having_aggregates(having, items)
            tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "ORDER":
            self._next()
            self._expect_keyword("BY")
            order_by = [self._parse_sort_item(items)]
            while self._peek().kind is _TokKind.COMMA:
                self._next()
                order_by.append(self._parse_sort_item(items))
            seen_items = set()
            for sort_item in order_by:
                key = (sort_item.kind, sort_item.column)
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
        return Query(
            tuple(items),
            table_tok.text,
            where,
            tuple(group_by),
            having,
            tuple(order_by),
            limit,
        )

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
            self._next()
            return SelectItem(tok.text, "column", tok.text)
        raise ValueError(f"invalid SELECT item {tok.text!r}")

    def _expect(self, kind: _TokKind) -> _Token:
        tok = self._next()
        if tok.kind is not kind:
            raise ValueError(f"expected {kind.value} but found {tok.text!r}")
        return tok

    # ----- WHERE boolean grammar: or_expr := and_expr (OR and_expr)* -----

    def _parse_or(self) -> WhereExpr:
        operands = [self._parse_and()]
        while self._accept_keyword("OR"):
            operands.append(self._parse_and())
        if len(operands) == 1:
            return operands[0]
        return BoolOp("OR", tuple(operands))

    def _parse_and(self) -> WhereExpr:
        operands = [self._parse_factor()]
        while self._accept_keyword("AND"):
            operands.append(self._parse_factor())
        if len(operands) == 1:
            return operands[0]
        return BoolOp("AND", tuple(operands))

    def _parse_factor(self) -> WhereExpr:
        tok = self._peek()
        if tok.kind is _TokKind.LPAREN:
            self._next()
            node = self._parse_or()
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
            return self._parse_comparison()
        # NOT, dangling AND/OR, ')', EOF, numbers, ...: none can start a
        # boolean condition.
        raise ValueError(f"expected a boolean condition but found {tok.text!r}")

    def _parse_comparison(self) -> Comparison:
        col = self._expect(_TokKind.IDENT)
        op_tok = self._expect(_TokKind.OP)
        if op_tok.text not in _OPS:
            raise ValueError(f"unsupported operator {op_tok.text!r}")
        val = self._next()
        if val.kind is _TokKind.NUMBER:
            return Comparison(col.text, op_tok.text, val.text, False)
        if val.kind is _TokKind.STRING:
            return Comparison(col.text, op_tok.text, val.text, True)
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

    def _parse_sort_item(self, items: List[SelectItem]) -> SortItem:
        tok = self._peek()
        if (
            tok.kind is _TokKind.IDENT
            and self.tokens[self.pos + 1].kind is _TokKind.LPAREN
        ):
            name_tok = self._next()
            self._next()  # (
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
                    "ORDER BY may only reference result columns, count(*) "
                    "or sum/avg/min/max aggregates"
                )
        elif tok.kind is _TokKind.IDENT:
            self._next()
            text, kind, column = tok.text, "column", tok.text
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
        sort_item = SortItem(text, kind, column, descending)
        self._validate_sort_item(sort_item, items)
        return sort_item

    def _validate_sort_item(
        self, sort_item: SortItem, items: List[SelectItem]
    ) -> None:
        """Every ORDER BY item must reference a column of the result."""
        for item in items:
            if item.kind == sort_item.kind and item.column == sort_item.column:
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
