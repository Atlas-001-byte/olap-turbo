"""Baseline tests for the OLAP Turbo query engine."""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from olap_turbo import execute
from olap_turbo.engine import BATCH_SIZE, render
from olap_turbo.sql import parse_sql

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "sales.csv").write_text(
        "id,region,amount,note\n"
        "1,east,10,a\n"
        "2,west,20.5,b\n"
        "3,east,30,\n"
        "4,east,abc,x\n"
        "5,west,-2.25,y\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------- projection queries ----------

def test_projection_order_and_filter(data_dir):
    result = execute(str(data_dir), 'SELECT amount, id FROM sales WHERE region = "east"')
    assert result["columns"] == ["amount", "id"]
    assert result["rows"] == [[10, 1], [30, 3], ["abc", 4]]
    assert result["row_count"] == 3


def test_projection_preserves_csv_order(data_dir):
    result = execute(str(data_dir), "SELECT id FROM sales WHERE id >= 2")
    assert [r[0] for r in result["rows"]] == [2, 3, 4, 5]


def test_all_operators_numeric(data_dir):
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id > 2")["rows"] == [[3], [4], [5]]
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id >= 2")["rows"] == [[2], [3], [4], [5]]
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id < 3")["rows"] == [[1], [2]]
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id <= 2")["rows"] == [[1], [2]]
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id != 1")["rows"] == [[2], [3], [4], [5]]
    assert execute(str(data_dir), "SELECT id FROM sales WHERE id = 3")["rows"] == [[3]]


def test_string_comparison_is_textual(data_dir):
    result = execute(str(data_dir), 'SELECT id FROM sales WHERE note != "a" AND region = "west"')
    assert result["rows"] == [[2], [5]]


def test_negative_and_decimal_values(data_dir):
    result = execute(str(data_dir), "SELECT amount FROM sales WHERE id = 5")
    assert result["rows"] == [[-2.25]]
    # JSON must carry the exact decimal text, never binary-float artifacts.
    assert render(result) == '{"columns":["amount"],"rows":[[-2.25]],"row_count":1}'


def test_string_that_looks_numeric_stays_string(data_dir):
    # Comparing text with a quoted value never does numeric coercion;
    # "abc" only appears because the predicate touches region, not amount.
    result = execute(str(data_dir), 'SELECT amount FROM sales WHERE note = "x"')
    assert result["rows"] == [["abc"]]


def test_no_where_returns_all(data_dir):
    result = execute(str(data_dir), "SELECT id, region FROM sales")
    assert result["row_count"] == 5
    assert result["rows"][0] == [1, "east"]
    assert result["rows"][2] == [3, "east"]


# ---------- aggregate queries ----------

def test_count_star(data_dir):
    result = execute(str(data_dir), 'SELECT count(*) FROM sales WHERE region = "east"')
    assert result["columns"] == ["count(*)"]
    assert result["rows"] == [[3]]
    assert result["row_count"] == 1


def test_count_and_sums(data_dir):
    result = execute(str(data_dir), "SELECT count(*), sum(amount), sum(id) FROM sales WHERE id <= 3")
    assert result["columns"] == ["count(*)", "sum(amount)", "sum(id)"]
    assert result["rows"] == [[3, 60.5, 6]]
    assert result["row_count"] == 1


def test_sum_exact_decimal(data_dir):
    result = execute(str(data_dir), "SELECT count(*), sum(amount) FROM sales WHERE id != 4")
    assert result["rows"] == [[4, 58.25]]
    assert render(result).endswith('[4,58.25]],"row_count":1}')


def test_sum_integer_stays_integer_json(data_dir):
    result = execute(str(data_dir), "SELECT count(*), sum(id) FROM sales WHERE id <= 2")
    assert render(result) == '{"columns":["count(*)","sum(id)"],"rows":[[2,3]],"row_count":1}'


def test_aggregate_without_where(data_dir):
    result = execute(str(data_dir), "SELECT count(*), sum(id) FROM sales")
    assert result["rows"] == [[5, 15]]


# ---------- grouped aggregate queries ----------

def test_group_basic_shape_and_order(data_dir):
    # Row 4's amount is "abc", so filter it out; "east" first appears
    # before "west" in CSV order.
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount), sum(id) FROM sales "
        "WHERE id != 4 GROUP BY region",
    )
    assert result["columns"] == ["region", "count(*)", "sum(amount)", "sum(id)"]
    assert result["rows"] == [["east", 2, 40, 4], ["west", 2, 18.25, 7]]
    assert result["row_count"] == 2


def test_group_where_filters_before_grouping(data_dir):
    result = execute(
        str(data_dir),
        'SELECT region, count(*), sum(id) FROM sales WHERE id >= 3 GROUP BY region',
    )
    # east: rows 3,4 ; west: row 5
    assert result["rows"] == [["east", 2, 7], ["west", 1, 5]]
    assert result["row_count"] == 2


def test_group_decimal_keys_coalesce(tmp_path):
    # 1, 1.0 and 01 are the same finite decimal; x and 1.10 are distinct.
    (tmp_path / "t.csv").write_text(
        "k,v\n1,1\n1.0,2\n01,4\nx,3\n1.10,5\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT k, count(*), sum(v) FROM t GROUP BY k")
    assert result["columns"] == ["k", "count(*)", "sum(v)"]
    # First appearance order: decimal 1 (from "1"), text "x", decimal 1.10.
    assert result["rows"] == [[1, 3, 7], ["x", 1, 3], [Decimal("1.10"), 1, 5]]
    assert result["row_count"] == 3
    # The coalesced key renders as an integer; the non-integral key keeps
    # its exact decimal text, never a binary-float artifact like 1.1000000.
    assert render(result) == (
        '{"columns":["k","count(*)","sum(v)"],"rows":[[1,3,7],["x",1,3],'
        '[1.10,1,5]],"row_count":3}'
    )


def test_group_text_keys_stay_strings(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,v\neast,1\nwest,2\neast,3\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT g, count(*), sum(v) FROM t GROUP BY g")
    assert result["rows"] == [["east", 2, 4], ["west", 1, 2]]


def test_group_empty_where_match(data_dir):
    result = execute(
        str(data_dir),
        'SELECT region, count(*), sum(id) FROM sales WHERE id > 100 GROUP BY region',
    )
    assert result["columns"] == ["region", "count(*)", "sum(id)"]
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_group_empty_table(tmp_path):
    (tmp_path / "e.csv").write_text("k,v\n", encoding="utf-8")
    result = execute(str(tmp_path), "SELECT k, count(*), sum(v) FROM e GROUP BY k")
    assert result == {"columns": ["k", "count(*)", "sum(v)"], "rows": [], "row_count": 0}


def test_group_sum_integer_and_decimal_json(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,v\na,1\na,2.5\nb,10\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT g, count(*), sum(v) FROM t GROUP BY g")
    assert render(result) == (
        '{"columns":["g","count(*)","sum(v)"],"rows":[["a",2,3.5],["b",1,10]],'
        '"row_count":2}'
    )


def test_group_unreferenced_column_ignored(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,v,noise\na,1,garbage###\nb,2,not-a-number\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT g, count(*), sum(v) FROM t GROUP BY g")
    assert result["rows"] == [["a", 1, 1], ["b", 1, 2]]
    # The WHERE column is read, but an unreferenced garbage column is not.
    result = execute(
        str(tmp_path),
        'SELECT g, count(*), sum(v) FROM t WHERE g = "a" GROUP BY g',
    )
    assert result["rows"] == [["a", 1, 1]]


def test_group_sum_non_number_only_on_matching_rows(data_dir):
    # Row 4 has amount="abc"; excluded by WHERE, so grouping succeeds.
    result = execute(
        str(data_dir),
        'SELECT region, count(*), sum(amount) FROM sales WHERE id != 4 GROUP BY region',
    )
    assert result["rows"] == [["east", 2, 40], ["west", 2, 18.25]]
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT region, count(*), sum(amount) FROM sales GROUP BY region",
        )


def test_group_text_key_never_numeric_coerced(data_dir):
    # Grouping by a text column must never raise, even without WHERE.
    result = execute(
        str(data_dir), "SELECT region, count(*), sum(id) FROM sales GROUP BY region"
    )
    assert result["row_count"] == 2


def test_grouped_unknown_column_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT missing, count(*), sum(id) FROM sales GROUP BY missing",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT region, count(*), sum(missing) FROM sales GROUP BY region",
        )


def test_grouping_across_batches(tmp_path):
    n = BATCH_SIZE * 2 + 7
    lines = ["g,v\n"]
    for i in range(n):
        # Alternate keys so each key's rows span multiple batches.
        lines.append(f"{'ab'[i % 2]},{i}\n")
    (tmp_path / "big.csv").write_text("".join(lines), encoding="utf-8")
    result = execute(str(tmp_path), "SELECT g, count(*), sum(v) FROM big GROUP BY g")
    assert [r[0] for r in result["rows"]] == ["a", "b"]
    counts = {r[0]: r[1] for r in result["rows"]}
    assert counts == {"a": (n + 1) // 2, "b": n // 2}
    assert result["row_count"] == 2
    # Sums over alternating evens/odds must be exact decimals rendered ints.
    sums = {r[0]: r[2] for r in result["rows"]}
    expected_sum = sum(range(0, n, 2))
    assert sums["a"] == expected_sum


# ---------- multi-column grouped aggregate queries ----------

def test_multi_group_basic(tmp_path):
    (tmp_path / "t.csv").write_text(
        "region,prod,amount\n"
        "east,a,1\n"      # (east,a) first
        "west,b,2\n"      # (west,b)
        "east,a,3\n"      # existing
        "east,b,4\n"      # (east,b)
        "west,a,5\n"      # (west,a)
        "east,a,6\n",     # existing
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT region, prod, count(*), sum(amount) FROM t GROUP BY region, prod",
    )
    assert result["columns"] == ["region", "prod", "count(*)", "sum(amount)"]
    assert result["rows"] == [
        ["east", "a", 3, 10],
        ["west", "b", 1, 2],
        ["east", "b", 1, 4],
        ["west", "a", 1, 5],
    ]
    assert result["row_count"] == 4


def test_multi_group_with_where_and_multiple_sums(tmp_path):
    (tmp_path / "t.csv").write_text(
        "r,p,a,b\n"
        "x,u,1,10\n"
        "y,v,2,20\n"
        "x,u,3,30\n"
        "x,v,4,40\n"
        "y,u,5,50\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT r, p, count(*), sum(a), sum(b) FROM t WHERE a >= 2 "
        "GROUP BY r, p",
    )
    assert result["columns"] == ["r", "p", "count(*)", "sum(a)", "sum(b)"]
    # (y,v) first among matches, then (x,u), then (x,v), then (y,u)
    assert result["rows"] == [
        ["y", "v", 1, 2, 20],
        ["x", "u", 1, 3, 30],
        ["x", "v", 1, 4, 40],
        ["y", "u", 1, 5, 50],
    ]
    assert result["row_count"] == 4


def test_multi_group_decimal_keys_coalesce_per_position(tmp_path):
    # Each key position normalizes independently: 1/1.0 coalesce, and a
    # decimal position is never equal to text "1".
    (tmp_path / "t.csv").write_text(
        "k1,k2,v\n"
        "1,1.0,1\n"
        "1.0,01,2\n"      # same composite key as row 1
        "01,1,4\n"        # same composite key
        "1,x,8\n"         # text second position: distinct group
        "x,1,16\n"        # text first position: distinct group
        "1.10,1,32\n",    # non-integral first position: distinct group
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path), "SELECT k1, k2, count(*), sum(v) FROM t GROUP BY k1, k2"
    )
    assert result["rows"] == [
        [1, 1, 3, 7],
        [1, "x", 1, 8],
        ["x", 1, 1, 16],
        [Decimal("1.10"), 1, 1, 32],
    ]
    assert result["row_count"] == 4
    assert render(result) == (
        '{"columns":["k1","k2","count(*)","sum(v)"],"rows":'
        '[[1,1,3,7],[1,"x",1,8],["x",1,1,16],[1.10,1,1,32]],'
        '"row_count":4}'
    )


def test_multi_group_empty_table_and_no_match(tmp_path):
    (tmp_path / "e.csv").write_text("a,b,v\n", encoding="utf-8")
    result = execute(
        str(tmp_path),
        "SELECT a, b, count(*), sum(v) FROM e GROUP BY a, b",
    )
    assert result == {
        "columns": ["a", "b", "count(*)", "sum(v)"],
        "rows": [],
        "row_count": 0,
    }
    (tmp_path / "t.csv").write_text(
        "a,b,v\n1,2,10\n", encoding="utf-8"
    )
    result = execute(
        str(tmp_path),
        "SELECT a, b, count(*), sum(v) FROM t WHERE v > 100 GROUP BY a, b",
    )
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_multi_group_sum_non_number_only_matching_rows(tmp_path):
    (tmp_path / "t.csv").write_text(
        "a,b,v\n"
        "1,x,10\n"
        "2,y,abc\n"
        "1,x,5\n",
        encoding="utf-8",
    )
    # The bad sum cell belongs to a row filtered out by WHERE.
    result = execute(
        str(tmp_path),
        'SELECT a, b, count(*), sum(v) FROM t WHERE b != "y" GROUP BY a, b',
    )
    assert result["rows"] == [[1, "x", 2, 15]]
    with pytest.raises(ValueError):
        execute(
            str(tmp_path),
            "SELECT a, b, count(*), sum(v) FROM t GROUP BY a, b",
        )


def test_multi_group_unreferenced_column_ignored(tmp_path):
    (tmp_path / "t.csv").write_text(
        "a,b,v,noise\n"
        "1,x,1,garbage###\n"
        "2,y,2,not-a-number\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT a, b, count(*), sum(v) FROM t GROUP BY a, b",
    )
    assert result["rows"] == [[1, "x", 1, 1], [2, "y", 1, 2]]


def test_multi_group_unknown_column_raises_keyerror(tmp_path):
    (tmp_path / "t.csv").write_text("a,b,v\n1,2,3\n", encoding="utf-8")
    with pytest.raises(KeyError):
        execute(
            str(tmp_path),
            "SELECT a, missing, count(*), sum(v) FROM t GROUP BY a, missing",
        )
    with pytest.raises(KeyError):
        execute(
            str(tmp_path),
            "SELECT a, b, count(*), sum(missing) FROM t GROUP BY a, b",
        )
    with pytest.raises(KeyError):
        execute(
            str(tmp_path),
            "SELECT a, b, count(*), sum(v) FROM t WHERE missing = 1 GROUP BY a, b",
        )


def test_multi_grouping_across_batches(tmp_path):
    n = BATCH_SIZE * 2 + 7
    lines = ["g,h,v\n"]
    for i in range(n):
        # Two independent alternating keys; rows of each of the four
        # composites span multiple batches.
        lines.append(f"{'ab'[i % 2]},{'CD'[i % 3 % 2]},{i}\n")
    (tmp_path / "big.csv").write_text("".join(lines), encoding="utf-8")
    result = execute(
        str(tmp_path),
        "SELECT g, h, count(*), sum(v) FROM big GROUP BY g, h",
    )
    assert result["row_count"] == 4
    seen = {(r[0], r[1]): (r[2], r[3]) for r in result["rows"]}
    total_count = sum(c for c, _ in seen.values())
    assert total_count == n
    total_sum = sum(s for _, s in seen.values())
    assert total_sum == n * (n - 1) // 2
    # First composite key in CSV order is ("a", "C").
    assert (result["rows"][0][0], result["rows"][0][1]) == ("a", "C")


@pytest.mark.parametrize(
    "sql",
    [
        # SELECT must list every GROUP BY column, in the same order
        "SELECT a, count(*), sum(x) FROM t GROUP BY a, b",
        "SELECT b, a, count(*), sum(x) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), sum(x) FROM t GROUP BY b, a",
        # GROUP BY columns must be distinct, both in GROUP BY and SELECT
        "SELECT a, b, count(*), sum(x) FROM t GROUP BY a, a",
        "SELECT a, a, count(*), sum(x) FROM t GROUP BY a, a",
        # count(*) must follow the grouping columns
        "SELECT a, b, sum(x), count(*) FROM t GROUP BY a, b",
        "SELECT count(*), a, b, sum(x) FROM t GROUP BY a, b",
        # at least one sum, and only non-grouping sum() aggregates
        "SELECT a, b, count(*) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), avg(x) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), x, sum(y) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), sum(a) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), sum(b) FROM t GROUP BY a, b",
        "SELECT a, b, count(*), sum(x), sum(x) FROM t GROUP BY a, b",
        "SELECT a, b, count(*) FROM t GROUP BY a, b, c",
        # three-column GROUP BY follows the same rules
        "SELECT a, b, c, count(*), sum(a) FROM t GROUP BY a, b, c",
        "SELECT a, b, count(*), sum(x) FROM t GROUP BY a, b, c",
    ],
)
def test_bad_multi_grouped_sql_raises_valueerror(tmp_path, sql):
    (tmp_path / "t.csv").write_text(
        "a,b,c,x,y\n1,2,3,4,5\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        execute(str(tmp_path), sql)


def test_three_column_grouping(tmp_path):
    (tmp_path / "t.csv").write_text(
        "a,b,c,v\n"
        "1,x,p,1\n"
        "1,x,q,2\n"
        "1,x,p,4\n"
        "2,x,p,8\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT a, b, c, count(*), sum(v) FROM t GROUP BY a, b, c",
    )
    assert result["columns"] == ["a", "b", "c", "count(*)", "sum(v)"]
    assert result["rows"] == [
        [1, "x", "p", 2, 5],
        [1, "x", "q", 1, 2],
        [2, "x", "p", 1, 8],
    ]
    assert result["row_count"] == 3


@pytest.mark.parametrize(
    "sql",
    [
        # GROUP BY with the projection shape is still invalid.
        "SELECT a FROM t GROUP BY a",
        # grouping column must lead the SELECT list and match GROUP BY
        "SELECT b, count(*), sum(a) FROM t GROUP BY a",
        "SELECT a, count(*), sum(a) FROM t GROUP BY b",
        # count(*) must follow the grouping column
        "SELECT a, sum(b), count(*) FROM t GROUP BY a",
        # at least one sum is required
        "SELECT a, count(*) FROM t GROUP BY a",
        # only sum() aggregates, no duplicates
        "SELECT a, count(*), count(*) FROM t GROUP BY a",
        "SELECT a, count(*), avg(b) FROM t GROUP BY a",
        "SELECT a, count(*), sum(b), sum(b) FROM t GROUP BY a",
        # aggregate shape without a leading grouping column
        "SELECT count(*), sum(a) FROM t GROUP BY a",
        # clauses other than WHERE/GROUP BY
        "SELECT a, count(*), sum(b) FROM t GROUP BY a ORDER BY a",
        # GROUP BY without a column stays rejected; a multi-column GROUP BY
        # whose SELECT list omits a grouping column is rejected too.
        "SELECT a, count(*), sum(b) FROM t GROUP BY",
        "SELECT a, count(*), sum(c) FROM t GROUP BY a, b",
    ],
)
def test_bad_grouped_sql_raises_valueerror(tmp_path, sql):
    (tmp_path / "t.csv").write_text(
        "a,b,x,y\n1,2,1,2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        execute(str(tmp_path), sql)


# ---------- column pruning ----------

def test_unreferenced_garbage_column_is_ignored(tmp_path):
    (tmp_path / "t.csv").write_text(
        "a,b,noise\n"
        "1,x,not-a-number!!!\n"
        "2,y,####\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT a FROM t WHERE b = \"x\"")
    assert result["rows"] == [[1]]
    agg = execute(str(tmp_path), "SELECT count(*), sum(a) FROM t")
    assert agg["rows"] == [[2, 3]]


# ---------- batching ----------

def test_batches_larger_than_batch_size(tmp_path):
    n = BATCH_SIZE * 2 + 7
    (tmp_path / "big.csv").write_text(
        "k,v\n" + "".join(f"{i},{i}\n" for i in range(n)),
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT k FROM big WHERE v >= 100")
    assert result["row_count"] == n - 100
    assert result["rows"][0] == [100]
    assert result["rows"][-1] == [n - 1]
    agg = execute(str(tmp_path), "SELECT count(*), sum(v) FROM big WHERE k < 10")
    assert agg["rows"] == [[10, 45]]


# ---------- boolean WHERE (AND / OR / parentheses) ----------

def test_or_basic_projection(data_dir):
    result = execute(str(data_dir), "SELECT id FROM sales WHERE id = 1 OR id = 3")
    assert result["rows"] == [[1], [3]]
    assert result["row_count"] == 2


def test_and_binds_tighter_than_or(data_dir):
    # a = 1 OR (b = 2 AND c = 3): rows 1 and 2
    result = execute(
        str(data_dir),
        'SELECT id FROM sales WHERE id = 1 OR id = 2 AND region = "west"',
    )
    assert result["rows"] == [[1], [2]]


def test_parentheses_override_precedence(data_dir):
    # (id = 1 OR id = 2) AND region = "east": only row 1
    result = execute(
        str(data_dir),
        'SELECT id FROM sales WHERE (id = 1 OR id = 2) AND region = "east"',
    )
    assert result["rows"] == [[1]]


def test_nested_and_redundant_parentheses(data_dir):
    result = execute(
        str(data_dir),
        'SELECT id FROM sales WHERE ((id = 1 OR id = 2) AND '
        '(region = "east" OR region = "west"))',
    )
    assert result["rows"] == [[1], [2]]
    result = execute(
        str(data_dir),
        'SELECT id FROM sales WHERE (((id = 1 OR id = 3)))',
    )
    assert result["rows"] == [[1], [3]]


def test_or_mixed_types_no_cross_coercion(data_dir):
    # One numeric branch and one text branch; matches rows 4 (id = 4)
    # and 5 (note = "y").
    result = execute(
        str(data_dir),
        'SELECT id FROM sales WHERE id = 4 OR note = "y"',
    )
    assert result["rows"] == [[4], [5]]


def test_or_with_global_aggregate(data_dir):
    result = execute(
        str(data_dir),
        "SELECT count(*), sum(id) FROM sales WHERE id = 4 OR region = \"west\"",
    )
    assert result["rows"] == [[3, 11]]  # rows 2, 4, 5


def test_or_aggregate_excluded_bad_sum_cell(data_dir):
    result = execute(
        str(data_dir),
        "SELECT count(*), sum(amount) FROM sales WHERE id = 2 OR id = 5",
    )
    assert result["rows"] == [[2, 18.25]]


def test_or_with_single_column_grouping(data_dir):
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(id) FROM sales "
        "WHERE id = 1 OR id = 5 GROUP BY region",
    )
    assert result["columns"] == ["region", "count(*)", "sum(id)"]
    assert result["rows"] == [["east", 1, 1], ["west", 1, 5]]
    assert result["row_count"] == 2


def test_or_with_multi_column_grouping(tmp_path):
    (tmp_path / "t.csv").write_text(
        "r,p,v\n"
        "x,a,1\n"
        "y,b,2\n"
        "x,a,3\n"
        "x,b,4\n"
        "y,a,5\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT r, p, count(*), sum(v) FROM t "
        'WHERE (r = "x" AND p = "a") OR v >= 5 GROUP BY r, p',
    )
    # (x,a): rows v=1,3 -> count 2, sum 4 ; (y,a): row v=5
    assert result["rows"] == [
        ["x", "a", 2, 4],
        ["y", "a", 1, 5],
    ]
    assert result["row_count"] == 2


def test_or_empty_result_shape_unchanged(data_dir):
    agg = execute(
        str(data_dir), "SELECT count(*), sum(id) FROM sales WHERE id < 0 OR id > 99"
    )
    assert agg == {"columns": ["count(*)", "sum(id)"], "rows": [[0, 0]], "row_count": 1}
    grouped = execute(
        str(data_dir),
        'SELECT region, count(*), sum(id) FROM sales WHERE id < 0 OR id > 99 '
        "GROUP BY region",
    )
    assert grouped == {
        "columns": ["region", "count(*)", "sum(id)"],
        "rows": [],
        "row_count": 0,
    }


def test_or_numeric_cells_validated_eagerly_even_when_other_branch_true(data_dir):
    # Row 4 matches the id = 4 branch, but its amount is "abc": the
    # numeric OR branch must still be read for every row.
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT id FROM sales WHERE id = 4 OR amount >= 0",
        )
    # Same obligation regardless of branch order or a text branch.
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT id FROM sales WHERE amount >= 0 OR id = 4",
        )
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            'SELECT id FROM sales WHERE note = "z" OR amount >= 0',
        )
    # A numeric comparison under an AND inside parentheses is eager too:
    # row 1 passes the parenthesized group, other rows still get parsed.
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT id FROM sales WHERE id = 4 OR (id >= 0 AND amount >= 0)",
        )


def test_or_unknown_column_in_any_branch_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT id FROM sales WHERE missing = 1 OR id = 1",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT id FROM sales WHERE id = 1 OR (missing = 1 AND id = 2)",
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE (a = 1",            # unmatched '('
        "SELECT a FROM t WHERE a = 1)",            # unmatched ')'
        "SELECT a FROM t WHERE ()",                # empty group
        "SELECT a FROM t WHERE (a)",               # parens around a column
        "SELECT a FROM t WHERE a = (1)",           # parens around a value
        "SELECT a FROM t WHERE (1)",               # parens around a number
        "SELECT a FROM t WHERE (a = 1)",           # parens around one comparison
        "SELECT a FROM t WHERE ((a = 1))",         # nested parens around one
        "SELECT a FROM t WHERE (a = 1) OR b = 2",  # grouped single comparison
        "SELECT a FROM t WHERE (a = 1",            # missing close after valid group
        "SELECT a FROM t WHERE a = 1 OR",          # dangling OR
        "SELECT a FROM t WHERE OR a = 1",          # leading OR
        "SELECT a FROM t WHERE a = 1 AND",         # dangling AND
        "SELECT a FROM t WHERE AND a = 1",         # leading AND
        "SELECT a FROM t WHERE a = 1 AND OR a = 2",  # consecutive connectives
        "SELECT a FROM t WHERE a = 1 a = 2",       # missing connective
        "SELECT a FROM t WHERE NOT a = 1",         # NOT is unsupported
        "SELECT a FROM t WHERE a >= >= 1",         # consecutive operators
        "SELECT a FROM t WHERE a = 1 OR b = 2 OR", # trailing OR
        "SELECT a FROM t WHERE (a = 1 OR b = 2",   # unbalanced nested
        "SELECT a FROM t WHERE a = 1 OR ()",       # empty group on right
        "SELECT a, count(*) FROM t WHERE a = 1 OR b = 2",  # mixed SELECT shape
    ],
)
def test_bad_boolean_where_raises_valueerror(tmp_path, sql):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        execute(str(tmp_path), sql)


def test_boolean_predicate_across_batch_sizes(tmp_path, monkeypatch):
    import olap_turbo.engine as engine_mod

    n = BATCH_SIZE * 2 + 7
    (tmp_path / "big.csv").write_text(
        "k,v\n" + "".join(f"{'ab'[i % 2]},{i}\n" for i in range(n)),
        encoding="utf-8",
    )
    pred = "WHERE (v >= 10 AND v < 20) OR v >= 100"
    projection_sql = f"SELECT k, v FROM big {pred}"
    aggregate_sql = f"SELECT count(*), sum(v) FROM big {pred}"
    grouped_sql = f"SELECT k, count(*), sum(v) FROM big {pred} GROUP BY k"

    expected = {
        "proj": execute(str(tmp_path), projection_sql),
        "agg": execute(str(tmp_path), aggregate_sql),
        "grp": execute(str(tmp_path), grouped_sql),
    }
    expected_matches = [i for i in range(n) if (10 <= i < 20) or i >= 100]
    assert [r[1] for r in expected["proj"]["rows"]] == expected_matches
    assert expected["agg"]["rows"] == [
        [len(expected_matches), sum(expected_matches)]
    ]
    for size in (1, 3, 7, BATCH_SIZE + 1):
        monkeypatch.setattr(engine_mod, "BATCH_SIZE", size)
        assert execute(str(tmp_path), projection_sql) == expected["proj"]
        assert execute(str(tmp_path), aggregate_sql) == expected["agg"]
        assert execute(str(tmp_path), grouped_sql) == expected["grp"]


def test_parse_sql_boolean_tree_shape():
    from olap_turbo.sql import BoolOp, Comparison

    q = parse_sql("SELECT a FROM t WHERE a = 1 OR b = 2 AND c = 3")
    assert isinstance(q.where, BoolOp)
    assert q.where.op == "OR"
    assert len(q.where.operands) == 2
    assert isinstance(q.where.operands[0], Comparison)
    right = q.where.operands[1]
    assert isinstance(right, BoolOp) and right.op == "AND"
    assert [c.column for c in right.operands] == ["b", "c"]
    # Parentheses do not add wrapper nodes around a single grouped expr.
    q2 = parse_sql("SELECT a FROM t WHERE (a = 1 OR b = 2) AND c = 3")
    assert q2.where.op == "AND"
    assert isinstance(q2.where.operands[0], BoolOp)
    assert q2.where.operands[0].op == "OR"
    # No WHERE stays None.
    assert parse_sql("SELECT a FROM t").where is None
    # Predicate columns are all required, in first-appearance order.
    assert q2.required_columns() == ("a", "b", "c")


def test_or_with_unreferenced_garbage_column(tmp_path):
    (tmp_path / "t.csv").write_text(
        "a,b,noise\n"
        "1,x,not-a-number!!!\n"
        "2,y,####\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path), 'SELECT a FROM t WHERE b = "x" OR b = "y"'
    )
    assert result["rows"] == [[1], [2]]


# ---------- HAVING on global aggregates ----------

def test_having_global_count_passes(data_dir):
    result = execute(
        str(data_dir),
        'SELECT count(*), sum(id) FROM sales WHERE region = "east" HAVING count(*) >= 3',
    )
    assert result["columns"] == ["count(*)", "sum(id)"]
    assert result["rows"] == [[3, 8]]
    assert result["row_count"] == 1


def test_having_global_fails_returns_empty(data_dir):
    result = execute(
        str(data_dir),
        "SELECT count(*), sum(id) FROM sales HAVING count(*) > 5",
    )
    assert result["columns"] == ["count(*)", "sum(id)"]
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_having_global_after_where_empty_input_still_one_row(data_dir):
    # No rows match WHERE: count 0, sum 0; HAVING on count(*) <= 0 passes,
    # so the usual single global row is emitted.
    result = execute(
        str(data_dir),
        "SELECT count(*), sum(id) FROM sales WHERE id > 100 HAVING count(*) = 0",
    )
    assert result["rows"] == [[0, 0]]
    assert result["row_count"] == 1
    # A HAVING that fails on the empty aggregate drops the row.
    result = execute(
        str(data_dir),
        "SELECT count(*), sum(id) FROM sales WHERE id > 100 HAVING count(*) > 0",
    )
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_having_global_empty_table(tmp_path):
    (tmp_path / "e.csv").write_text("v\n", encoding="utf-8")
    passing = execute(str(tmp_path), "SELECT count(*), sum(v) FROM e HAVING count(*) = 0")
    assert passing == {"columns": ["count(*)", "sum(v)"], "rows": [[0, 0]], "row_count": 1}
    failing = execute(str(tmp_path), "SELECT count(*), sum(v) FROM e HAVING sum(v) != 0")
    assert failing["rows"] == []
    assert failing["row_count"] == 0


def test_having_sum_exact_decimal_comparison(tmp_path):
    (tmp_path / "t.csv").write_text(
        "v\n0.1\n0.2\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT count(*), sum(v) FROM t HAVING sum(v) = 0.3")
    assert result["rows"] == [[2, Decimal("0.3")]]
    assert render(result) == '{"columns":["count(*)","sum(v)"],"rows":[[2,0.3]],"row_count":1}'


@pytest.mark.parametrize(
    "op,rhs,passes",
    [
        ("=", "3", True),
        ("=", "4", False),
        ("!=", "4", True),
        ("!=", "3", False),
        (">", "2", True),
        (">", "3", False),
        (">=", "3", True),
        (">=", "4", False),
        ("<", "4", True),
        ("<", "3", False),
        ("<=", "3", True),
        ("<=", "2", False),
    ],
)
def test_having_all_operators(data_dir, op, rhs, passes):
    sql = f'SELECT count(*) FROM sales WHERE region = "east" HAVING count(*) {op} {rhs}'
    result = execute(str(data_dir), sql)
    if passes:
        assert result["rows"] == [[3]]
        assert result["row_count"] == 1
    else:
        assert result["rows"] == []
        assert result["row_count"] == 0


def test_having_global_and_or_and_precedence(data_dir):
    # Five rows total: count(*) = 5, sum(id) = 15.
    base = "SELECT count(*), sum(id) FROM sales HAVING "
    assert execute(str(data_dir), base + "count(*) > 1 AND sum(id) = 15")["rows"] == [[5, 15]]
    # AND binds tighter: (false AND ...) OR true -> passes
    assert execute(
        str(data_dir), base + "count(*) > 9 AND sum(id) = 0 OR count(*) = 5"
    )["rows"] == [[5, 15]]
    # Parentheses override: false OR ... grouped, then AND false -> fails
    fails = execute(
        str(data_dir),
        base + "(count(*) > 9 OR count(*) = 5) AND sum(id) = 0",
    )
    assert fails["rows"] == []
    assert fails["row_count"] == 0
    # Nested parentheses combining conditions are allowed.
    assert execute(
        str(data_dir),
        base + "((count(*) = 5 OR count(*) = 0) AND (sum(id) >= 10 AND sum(id) <= 20))",
    )["rows"] == [[5, 15]]


# ---------- HAVING on grouped aggregates ----------

def test_having_grouped_filters_groups(data_dir):
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount), sum(id) FROM sales "
        "WHERE id != 4 GROUP BY region HAVING count(*) >= 2",
    )
    assert result["columns"] == ["region", "count(*)", "sum(amount)", "sum(id)"]
    # Both groups have 2 matching rows; east 40 / west 18.25.
    assert result["rows"] == [["east", 2, 40, 4], ["west", 2, 18.25, 7]]
    assert result["row_count"] == 2
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount) FROM sales "
        "WHERE id != 4 GROUP BY region HAVING sum(amount) > 20",
    )
    assert result["rows"] == [["east", 2, 40]]
    assert result["row_count"] == 1


def test_having_grouped_preserves_first_appearance_order(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,v\n"
        "a,1\n"   # a first
        "b,10\n"  # b second
        "c,1\n"   # c third
        "b,10\n"
        "a,1\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT g, count(*), sum(v) FROM t GROUP BY g HAVING count(*) = 2",
    )
    # b and a survive; a appeared before b in the filtered stream.
    assert result["rows"] == [["a", 2, 2], ["b", 2, 20]]
    assert result["row_count"] == 2


def test_having_grouped_all_groups_filtered(tmp_path):
    (tmp_path / "t.csv").write_text("g,v\na,1\nb,2\n", encoding="utf-8")
    result = execute(
        str(tmp_path), "SELECT g, count(*), sum(v) FROM t GROUP BY g HAVING count(*) > 5"
    )
    assert result["columns"] == ["g", "count(*)", "sum(v)"]
    assert result["rows"] == []
    assert result["row_count"] == 0


def test_having_grouped_empty_input(tmp_path):
    (tmp_path / "e.csv").write_text("g,v\n", encoding="utf-8")
    result = execute(
        str(tmp_path),
        "SELECT g, count(*), sum(v) FROM e GROUP BY g HAVING count(*) > 0",
    )
    assert result == {"columns": ["g", "count(*)", "sum(v)"], "rows": [], "row_count": 0}


def test_having_grouped_multi_column(tmp_path):
    (tmp_path / "t.csv").write_text(
        "r,p,v\n"
        "x,a,1\n"
        "y,b,2\n"
        "x,a,3\n"
        "x,b,4\n"
        "y,a,5\n",
        encoding="utf-8",
    )
    result = execute(
        str(tmp_path),
        "SELECT r, p, count(*), sum(v) FROM t GROUP BY r, p "
        "HAVING count(*) >= 2 OR sum(v) >= 5",
    )
    # (x,a) count 2 passes; (y,b) dropped; (x,b) sum 4 dropped; (y,a) sum 5 passes.
    assert result["rows"] == [["x", "a", 2, 4], ["y", "a", 1, 5]]
    assert result["row_count"] == 2


def test_having_grouped_runs_after_where(data_dir):
    # Rows 3,4 in east (amount of row 4 is "abc", but HAVING reads only
    # the aggregate, and sum(id) avoids it), row 5 in west.
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(id) FROM sales WHERE id >= 3 "
        "GROUP BY region HAVING count(*) = 1",
    )
    assert result["rows"] == [["west", 1, 5]]


def test_having_where_then_having_decimal_boundary(data_dir):
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount) FROM sales WHERE id != 4 "
        "GROUP BY region HAVING sum(amount) >= 18.25",
    )
    assert result["rows"] == [["east", 2, 40], ["west", 2, 18.25]]


# ---------- HAVING column pruning ----------

def test_having_does_not_require_extra_columns(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,v,noise\na,1,garbage###\nb,2,not-a-number\n",
        encoding="utf-8",
    )
    # HAVING only references selected aggregates; the garbage column is
    # never read.
    result = execute(
        str(tmp_path),
        "SELECT g, count(*), sum(v) FROM t GROUP BY g HAVING count(*) = 1",
    )
    assert result["rows"] == [["a", 1, 1], ["b", 1, 2]]
    # A column used only inside a selected sum is materialized as before.
    result = execute(
        str(tmp_path),
        "SELECT g, count(*), sum(v) FROM t GROUP BY g HAVING sum(v) = 2",
    )
    assert result["rows"] == [["b", 1, 2]]


def test_having_across_batch_sizes(tmp_path, monkeypatch):
    import olap_turbo.engine as engine_mod

    n = BATCH_SIZE * 2 + 7
    (tmp_path / "big.csv").write_text(
        "g,v\n" + "".join(f"{'ab'[i % 2]},{i}\n" for i in range(n)),
        encoding="utf-8",
    )
    sql = "SELECT g, count(*), sum(v) FROM big GROUP BY g HAVING count(*) >= 1000"
    expected = execute(str(tmp_path), sql)
    assert expected["row_count"] == 2
    for size in (1, 3, 7, BATCH_SIZE + 1):
        monkeypatch.setattr(engine_mod, "BATCH_SIZE", size)
        assert execute(str(tmp_path), sql) == expected


# ---------- HAVING parse shape ----------

def test_parse_sql_having_tree_shape():
    from olap_turbo.sql import BoolOp, HavingComparison

    q = parse_sql(
        "SELECT count(*), sum(a) FROM t HAVING count(*) > 1 AND sum(a) <= 2 OR count(*) = 0"
    )
    assert isinstance(q.having, BoolOp)
    assert q.having.op == "OR"
    left = q.having.operands[0]
    assert isinstance(left, BoolOp) and left.op == "AND"
    c0, c1 = left.operands
    assert isinstance(c0, HavingComparison)
    assert (c0.kind, c0.op, c0.value_text) == ("count_star", ">", "1")
    assert isinstance(c1, HavingComparison)
    assert (c1.kind, c1.column, c1.op, c1.value_text) == ("sum", "a", "<=", "2")
    assert q.having.operands[1].kind == "count_star"
    assert parse_sql("SELECT count(*) FROM t").having is None


# ---------- HAVING error forms ----------

@pytest.mark.parametrize(
    "sql",
    [
        # HAVING on a plain projection query
        "SELECT a FROM t HAVING count(*) > 1",
        "SELECT a FROM t WHERE a = 1 HAVING count(*) > 1",
        # wrong position: before WHERE / before GROUP BY / between GROUP and BY
        "SELECT count(*) FROM t HAVING count(*) > 1 WHERE a = 1",
        "SELECT a, count(*), sum(b) FROM t HAVING count(*) > 1 GROUP BY a",
        "SELECT a, count(*), sum(b) FROM t GROUP HAVING count(*) > 1 BY a",
        # repeated HAVING
        "SELECT count(*) FROM t HAVING count(*) > 1 HAVING count(*) > 2",
        # left side must be a selected aggregate
        "SELECT count(*) FROM t HAVING sum(a) > 1",            # sum not selected
        "SELECT count(*), sum(a) FROM t HAVING sum(b) > 1",    # other sum
        "SELECT a, count(*), sum(b) FROM t GROUP BY a HAVING a > 1",      # grouping column
        "SELECT a, count(*), sum(b) FROM t GROUP BY a HAVING b > 1",      # plain column
        "SELECT count(*) FROM t HAVING 1 > count(*)",          # reversed sides
        # right side must be an unquoted finite decimal number
        'SELECT count(*) FROM t HAVING count(*) > "1"',        # quoted value
        "SELECT count(*) FROM t HAVING count(*) > a",          # column value
        "SELECT count(*) FROM t HAVING count(*) > count(*)",   # aggregate value
        "SELECT count(*) FROM t HAVING count(*) > -",          # incomplete number
        "SELECT count(*) FROM t HAVING count(*) > 1.2.3",      # malformed number
        # NOT, other functions and other expression forms
        "SELECT count(*) FROM t HAVING NOT count(*) > 1",
        "SELECT count(*) FROM t HAVING avg(x) > 1",
        "SELECT count(*) FROM t HAVING count(col) > 1",
        "SELECT count(*) FROM t HAVING count(*) > 1 + 1",
        "SELECT count(*) FROM t HAVING count(*) ",             # incomplete condition
        "SELECT count(*) FROM t HAVING",                       # no condition
        "SELECT count(*) FROM t HAVING ()",                    # empty group
        "SELECT count(*) FROM t HAVING (count(*) > 1)",        # parens around one
        "SELECT count(*) FROM t HAVING ((count(*) > 1))",      # nested single
        # malformed operators / connectives / parens
        "SELECT count(*) FROM t HAVING count(*) >> 1",         # bad operator
        "SELECT count(*) FROM t HAVING count(*) >",            # missing value
        "SELECT count(*) FROM t HAVING count(*) > 1 AND",      # dangling AND
        "SELECT count(*) FROM t HAVING OR count(*) > 1",       # leading OR
        "SELECT count(*) FROM t HAVING count(*) > 1 OR",       # trailing OR
        "SELECT count(*) FROM t HAVING count(*) > 1 AND OR count(*) > 2",
        "SELECT count(*) FROM t HAVING (count(*) > 1",         # unmatched '('
        "SELECT count(*) FROM t HAVING count(*) > 1)",         # unmatched ')'
        "SELECT count(*) FROM t HAVING count(*) > 1 count(*) > 2",
        # same rules on grouped queries
        "SELECT a, count(*) FROM t GROUP BY a HAVING sum(a) > 1",
        "SELECT a, count(*), sum(b) FROM t GROUP BY a HAVING a = 1",
        'SELECT a, count(*), sum(b) FROM t GROUP BY a HAVING count(*) > "1"',
    ],
)
def test_bad_having_sql_raises_valueerror(tmp_path, sql):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        execute(str(tmp_path), sql)


def test_having_unknown_aggregate_still_valueerror_not_keyerror(data_dir):
    # Missing aggregate references are a syntax error, never a KeyError.
    with pytest.raises(ValueError):
        execute(str(data_dir), "SELECT count(*) FROM sales HAVING sum(nope) > 1")


# ---------- error contract ----------

def test_missing_table_raises_filenotfound(data_dir):
    with pytest.raises(FileNotFoundError):
        execute(str(data_dir), "SELECT a FROM nope")


def test_unknown_column_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        execute(str(data_dir), "SELECT missing FROM sales")
    with pytest.raises(KeyError):
        execute(str(data_dir), "SELECT id FROM sales WHERE missing = 1")
    with pytest.raises(KeyError):
        execute(str(data_dir), "SELECT count(*), sum(missing) FROM sales")


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "SELECT a sales",                      # missing FROM
        "SELECT a FROM sales WHERE",           # incomplete WHERE
        "SELECT a FROM sales WHERE region =",  # missing value
        "SELECT * FROM sales",                 # wildcard
        "SELECT a FROM sales OR a = 1",        # stray OR outside WHERE
        "SELECT avg(a) FROM sales",            # unsupported function
        "SELECT count(a) FROM sales",
        "SELECT a, count(*) FROM sales",       # mixed
        "SELECT sum(a) FROM sales",            # sum without count(*)
        "SELECT a FROM sales GROUP BY a",      # unsupported clause
        "SELECT a FROM sales ORDER BY a",
        "SELECT a FROM sales WHERE a = 'x'",   # single quotes
        "SELECT a FROM sales WHERE a = 1x",    # malformed value
        "SELECT a FROM 'sales'",               # quoted table
        "DELETE FROM sales",
        "select a from sales where a >> 1",    # bad operator
    ],
)
def test_bad_sql_raises_valueerror(data_dir, sql):
    with pytest.raises(ValueError):
        execute(str(data_dir), sql)


def test_numeric_predicate_on_non_number(data_dir):
    with pytest.raises(ValueError):
        execute(str(data_dir), "SELECT id FROM sales WHERE amount >= 0")
    # A row that is never selected must still be validated when the
    # predicate column contains garbage: the predicate itself needs it.
    with pytest.raises(ValueError):
        execute(str(data_dir), "SELECT id FROM sales WHERE amount = 4 AND id = 1")


def test_sum_on_non_number_only_matching_row(data_dir):
    # Row 4 has amount="abc" but does not match, so sum must succeed.
    assert execute(
        str(data_dir), "SELECT count(*), sum(amount) FROM sales WHERE id != 4"
    )["rows"][0][1] == 58.25
    with pytest.raises(ValueError):
        execute(str(data_dir), "SELECT count(*), sum(amount) FROM sales")


def test_simple_identifiers_only():
    q = parse_sql("SELECT a_1 FROM t2")
    assert q.table == "t2"
    assert q.select[0].column == "a_1"


# ---------- output shape & determinism ----------

def test_output_is_single_json_object(data_dir):
    result = execute(str(data_dir), 'SELECT region, note FROM sales WHERE id = 3')
    payload = json.loads(render(result))
    assert payload == {"columns": ["region", "note"], "rows": [["east", ""]], "row_count": 1}


def test_deterministic_output(data_dir):
    sql = 'SELECT id, region FROM sales WHERE region = "east" AND id > 1'
    assert render(execute(str(data_dir), sql)) == render(execute(str(data_dir), sql))


def test_no_derived_files_written(data_dir):
    before = {p.name for p in data_dir.iterdir()}
    execute(str(data_dir), "SELECT count(*), sum(id) FROM sales WHERE id > 2")
    assert {p.name for p in data_dir.iterdir()} == before


# ---------- CLI ----------

def test_module_cli(data_dir):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "olap_turbo",
            "--data-dir",
            str(data_dir),
            "--query",
            "SELECT count(*) FROM sales WHERE id >= 3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload == {"columns": ["count(*)"], "rows": [[3]], "row_count": 1}
