"""Minimal self-contained SQL lexer and parser.

Supported grammar (the complete surface for this baseline)::

    query      := SELECT select_list FROM identifier
                  [ WHERE condition ] [ GROUP BY identifier (, identifier)* ]
    select_list:= grouping_column (, grouping_column)*,
                  count_star, sum_expr (, sum_expr)*   (grouped aggregate query)
                | count_star [, sum_expr ]*            (aggregate query)
                | column (, column)*                   (projection query)
    condition  := disjunction
    disjunction:= conjunction (OR conjunction)*
    conjunction:= factor (AND factor)*
    factor     := comparison | '(' disjunction ')'
    comparison := identifier op value
    op         := = | != | > | >= | < | <=
    value      := number-literal | double-quoted-string-literal

AND binds tighter than OR, and parentheses (which may nest) override
precedence; parentheses only ever group boolean conditions, never
columns, values or arithmetic.

A grouped query names one or more distinct grouping columns (at least two
for the multi-column shape): they must head the SELECT list in the same
order as the GROUP BY list, followed by count(*) and one or more distinct
sum(column) expressions.

Anything else (missing FROM, NOT, functions other than count(*)/sum(),
wildcards, extra clauses, unbalanced or empty parentheses, malformed
syntax) raises :class:`ValueError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union


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


_KEYWORDS = {"SELECT", "FROM", "WHERE", "AND", "OR", "GROUP", "BY"}


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
class And:
    # Conjunction of two or more conditions.
    terms: Tuple["Condition", ...]


@dataclass(frozen=True)
class Or:
    # Disjunction of two or more conditions.
    terms: Tuple["Condition", ...]


Condition = Union[Comparison, And, Or]


def iter_comparisons(node: Condition) -> Iterator[Comparison]:
    """Yield every leaf comparison in a condition tree, left to right."""
    if isinstance(node, Comparison):
        yield node
    else:
        for term in node.terms:
            yield from iter_comparisons(term)


@dataclass(frozen=True)
class SelectItem:
    # Original expression text as written in the query, e.g. "count(*)", "sum(a)".
    text: str
    kind: str  # "column" | "count_star" | "sum"
    column: Optional[str] = None


@dataclass(frozen=True)
class Query:
    select: Tuple[SelectItem, ...]
    table: str
    where: Optional[Condition] = None
    group_by: Tuple[str, ...] = ()

    @property
    def is_aggregate(self) -> bool:
        return bool(self.group_by) or self.select[0].kind == "count_star"

    @property
    def is_grouped(self) -> bool:
        return bool(self.group_by)

    def required_columns(self) -> Tuple[str, ...]:
        """Columns the scan must materialize (group key, sums, predicates)."""
        cols: List[str] = []
        seen = set()
        for item in self.select:
            if item.kind == "column":
                if item.column not in seen:
                    seen.add(item.column)
                    cols.append(item.column)
            elif item.kind == "sum" and item.column not in seen:
                seen.add(item.column)
                cols.append(item.column)
        if self.where is not None:
            for comp in iter_comparisons(self.where):
                if comp.column not in seen:
                    seen.add(comp.column)
                    cols.append(comp.column)
        return tuple(cols)


_OPS = {"=", "!=", ">", ">=", "<", "<="}


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

    def parse(self) -> Query:
        self._expect_keyword("SELECT")
        items = self._parse_select_list()
        self._expect_keyword("FROM")
        table_tok = self._next()
        if table_tok.kind is not _TokKind.IDENT:
            raise ValueError("table name must be a simple identifier")
        where: Optional[Condition] = None
        group_by: List[str] = []
        tok = self._peek()
        if tok.kind is _TokKind.KEYWORD and tok.text == "WHERE":
            self._next()
            where = self._parse_where()
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
        if tok.kind is not _TokKind.EOF:
            # Trailing tokens: unsupported clause (ORDER BY, HAVING, OR ...)
            # or plain malformed input.
            raise ValueError(f"unsupported or unexpected token {tok.text!r}")
        return Query(tuple(items), table_tok.text, where, tuple(group_by))

    def _validate_grouped(
        self, items: List[SelectItem], group_by: List[str]
    ) -> None:
        """Enforce the grouped shape.

        SELECT leads with the GROUP BY columns in the same order (distinct),
        followed by count(*) and one or more distinct sum(column) items.
        """
        if len(group_by) != len(set(group_by)):
            raise ValueError("GROUP BY may not list the same column twice")
        if len(items) < len(group_by) + 2:
            raise ValueError(
                "grouped SELECT must list the grouping columns followed by "
                "count(*) and one or more sum(column)"
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
        seen_sums = set()
        for item in items[len(group_by) + 1 :]:
            if item.kind != "sum":
                raise ValueError(
                    "grouped SELECT may only contain the grouping columns, "
                    "count(*) and sum(column)"
                )
            if item.column in seen_sums:
                raise ValueError(f"duplicate aggregate sum({item.column})")
            if item.column in group_by:
                raise ValueError(
                    f"GROUP BY column {item.column} may not be summed"
                )
            seen_sums.add(item.column)

    def _parse_select_list(self) -> List[SelectItem]:
        items = [self._parse_select_item()]
        while self._peek().kind is _TokKind.COMMA:
            self._next()
            items.append(self._parse_select_item())
        return items

    def _validate_ungrouped(self, items: List[SelectItem]) -> None:
        """Rules for queries without GROUP BY (the baseline shapes)."""
        if items[0].kind == "sum":
            raise ValueError("sum(column) requires count(*) in the SELECT list")
        aggregate = items[0].kind == "count_star"
        for item in items[1:]:
            if aggregate:
                if item.kind != "sum":
                    raise ValueError(
                        "aggregate SELECT may only contain count(*) and sum(column)"
                    )
            elif item.kind == "sum":
                raise ValueError(
                    "sum(column) requires count(*) in the SELECT list"
                )
            elif item.kind != "column":
                raise ValueError(
                    "count(*)/sum() cannot be mixed with plain columns"
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
            if upper == "SUM" and inner.kind is _TokKind.IDENT:
                return SelectItem(f"sum({inner.text})", "sum", inner.text)
            # Any other function (count(col), avg, max, ...) is unsupported.
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

    def _parse_where(self) -> Condition:
        return self._parse_disjunction()

    def _parse_disjunction(self) -> Condition:
        terms = [self._parse_conjunction()]
        while self._peek().kind is _TokKind.KEYWORD and self._peek().text == "OR":
            self._next()
            terms.append(self._parse_conjunction())
        if len(terms) == 1:
            return terms[0]
        return Or(tuple(terms))

    def _parse_conjunction(self) -> Condition:
        factors = [self._parse_factor()]
        while self._peek().kind is _TokKind.KEYWORD and self._peek().text == "AND":
            self._next()
            factors.append(self._parse_factor())
        if len(factors) == 1:
            return factors[0]
        return And(tuple(factors))

    def _parse_factor(self) -> Condition:
        if self._peek().kind is _TokKind.LPAREN:
            self._next()
            node = self._parse_disjunction()
            self._expect(_TokKind.RPAREN)
            return node
        return self._parse_comparison()

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


def parse_sql(sql: str) -> Query:
    """Parse a query string into a :class:`Query`."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("empty query")
    return _Parser(sql).parse()
