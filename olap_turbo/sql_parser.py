"""极简 SQL 解析器。

支持两种语句：

1. ``SELECT col, ... FROM table WHERE col op value AND ...``
2. ``SELECT count(*) [, sum(col) ...] FROM table [WHERE ...]``

解析失败一律抛出 ``ValueError``；列是否存在于表中由执行引擎解析并抛
``KeyError``，本模块只做语法层面的检查。
"""

from dataclasses import dataclass
import re

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_OPS = ("!=", ">=", "<=", "=", ">", "<")


@dataclass(frozen=True)
class Token:
    kind: str  # IDENT | NUMBER | STRING | OP | LPAREN | RPAREN | COMMA | STAR
    value: object
    start: int
    end: int


@dataclass(frozen=True)
class Predicate:
    column: str
    op: str
    value: object  # Decimal（数值）或 str（文本）
    is_text: bool


@dataclass(frozen=True)
class QueryPlan:
    table: str
    is_aggregate: bool
    # 普通列查询时的投影列（按 SELECT 顺序，允许重复）。
    select_columns: tuple
    # 聚合项：("count", 原文) 或 ("sum", 列名, 原文)，按 SELECT 顺序。
    aggregates: tuple
    predicates: tuple


def _tokenize(sql):
    tokens = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha() or c == "_":
            m = _IDENT_RE.match(sql, i)
            tokens.append(Token("IDENT", m.group(0), i, m.end()))
            i = m.end()
            continue
        if c.isdigit() or c == "." or (
            c in "+-" and i + 1 < n and (sql[i + 1].isdigit() or sql[i + 1] == ".")
        ):
            m = _NUMBER_RE.match(sql, i)
            if not m:
                raise ValueError(f"无法解析的 SQL 片段: {sql[i:i + 10]!r}")
            tokens.append(Token("NUMBER", m.group(0), i, m.end()))
            i = m.end()
            continue
        if c == '"':
            j = i + 1
            chars = []
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':  # "" 转义
                        chars.append('"')
                        j += 2
                        continue
                    break
                chars.append(sql[j])
                j += 1
            if j >= n:
                raise ValueError("字符串常量缺少结束双引号")
            tokens.append(Token("STRING", "".join(chars), i, j + 1))
            i = j + 1
            continue
        if c == "(":
            tokens.append(Token("LPAREN", c, i, i + 1))
        elif c == ")":
            tokens.append(Token("RPAREN", c, i, i + 1))
        elif c == ",":
            tokens.append(Token("COMMA", c, i, i + 1))
        elif c == "*":
            tokens.append(Token("STAR", c, i, i + 1))
        elif c in "=!<>":
            two = sql[i:i + 2]
            if two in _OPS:
                tokens.append(Token("OP", two, i, i + 2))
                i += 2
                continue
            one = sql[i]
            if one in _OPS:
                tokens.append(Token("OP", one, i, i + 1))
            else:
                raise ValueError(f"无法解析的运算符: {one!r}")
        else:
            raise ValueError(f"无法解析的 SQL 字符: {c!r}")
        i += 1
    return tokens


class _Parser:
    def __init__(self, sql, tokens):
        self.sql = sql
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0):
        k = self.pos + offset
        return self.tokens[k] if k < len(self.tokens) else None

    def take(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, kind, value=None):
        t = self.peek()
        if t is None or t.kind != kind or (value is not None and t.value != value):
            raise ValueError("SQL 无法解析或包含不支持的语法")
        return self.take()

    def keyword(self, *names):
        t = self.peek()
        return t is not None and t.kind == "IDENT" and t.value.upper() in names

    def parse(self):
        if not self.keyword("SELECT"):
            raise ValueError("SQL 无法解析：缺少 SELECT")
        self.take()
        columns, aggregates = self._parse_select_list()

        if not self.keyword("FROM"):
            raise ValueError("SQL 无法解析：缺少 FROM")
        self.take()
        table_tok = self.peek()
        if table_tok is None or table_tok.kind != "IDENT":
            raise ValueError("SQL 无法解析：表名必须是简单标识符")
        self.take()
        table = table_tok.value

        predicates = ()
        if self.keyword("WHERE"):
            self.take()
            predicates = self._parse_where()
        elif self.peek() is not None:
            raise ValueError("SQL 包含不支持的子句")

        if self.peek() is not None:
            raise ValueError("SQL 包含不支持的子句")

        is_aggregate = bool(aggregates)
        return QueryPlan(table, is_aggregate, tuple(columns), tuple(aggregates), predicates)

    def _parse_select_list(self):
        columns = []
        aggregates = []
        while True:
            start = self.peek()
            if start is None:
                raise ValueError("SQL 无法解析：SELECT 列表为空")
            first = self.take()

            if first.kind == "IDENT" and self.peek() is not None and self.peek().kind == "LPAREN":
                # 函数调用
                name = first.value.upper()
                self.take()  # LPAREN
                if name == "COUNT":
                    arg = self.peek()
                    if arg is None or arg.kind != "STAR":
                        raise ValueError("仅支持 count(*)，不支持其他 count 参数")
                    self.take()
                    self.expect("RPAREN")
                    aggregates.append(("count", self.sql[start.start:self.tokens[self.pos - 1].end]))
                elif name == "SUM":
                    arg = self.peek()
                    if arg is None or arg.kind != "IDENT":
                        raise ValueError("sum() 的参数必须是简单列名")
                    self.take()
                    self.expect("RPAREN")
                    aggregates.append(("sum", arg.value,
                                       self.sql[start.start:self.tokens[self.pos - 1].end]))
                else:
                    raise ValueError(f"不支持的函数: {first.value}")
            elif first.kind == "IDENT":
                columns.append(first.value)
            elif first.kind == "STAR":
                raise ValueError("不支持通配符列 *，请显式列出列名")
            else:
                raise ValueError("SQL 无法解析：SELECT 项必须是列名或 count(*)/sum(列)")

            nxt = self.peek()
            if nxt is None:
                break
            if nxt.kind == "COMMA":
                self.take()
                continue
            break

        if aggregates:
            if columns:
                raise ValueError("聚合查询不能与普通列混用")
            if aggregates[0][0] != "count":
                raise ValueError("聚合查询必须以 count(*) 开头")
            if sum(1 for a in aggregates if a[0] == "count") > 1:
                raise ValueError("聚合查询只能包含一个 count(*)")
        return columns, aggregates

    def _parse_where(self):
        predicates = []
        while True:
            col = self.peek()
            if col is None or col.kind != "IDENT":
                raise ValueError("SQL 无法解析：WHERE 条件格式错误")
            self.take()
            op_tok = self.peek()
            if op_tok is None or op_tok.kind != "OP" or op_tok.value not in _OPS:
                raise ValueError("WHERE 仅支持 =、!=、>、>=、<、<= 比较")
            self.take()
            val_tok = self.peek()
            if val_tok is None or val_tok.kind not in ("NUMBER", "STRING"):
                raise ValueError("比较值必须是不带引号的数字或双引号字符串")
            self.take()

            if val_tok.kind == "STRING":
                predicates.append(Predicate(col.value, op_tok.value, val_tok.value, True))
            else:
                predicates.append(Predicate(col.value, op_tok.value,
                                            _to_decimal(val_tok.value), False))

            nxt = self.peek()
            if nxt is None:
                break
            if nxt.kind == "IDENT" and nxt.value.upper() == "AND":
                self.take()
                continue
            if nxt.kind == "IDENT" and nxt.value.upper() == "OR":
                raise ValueError("不支持 OR，仅支持 AND 组合条件")
            raise ValueError("SQL 无法解析或包含不支持的语法")
        return predicates


def _to_decimal(text):
    from decimal import Decimal, InvalidOperation
    try:
        return Decimal(text)
    except InvalidOperation:
        raise ValueError(f"无法按十进制数读取: {text!r}")


def parse_query(sql):
    """把 SQL 字符串解析为 :class:`QueryPlan`。"""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL 不能为空")
    return _Parser(sql, _tokenize(sql)).parse()
