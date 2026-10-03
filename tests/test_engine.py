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
        "SELECT a, count(*), sum(b) FROM t GROUP BY a HAVING count(*) > 1",
        # GROUP BY with multiple columns / missing column
        "SELECT a, count(*), sum(b) FROM t GROUP BY",
        "SELECT a, count(*), sum(b) FROM t GROUP BY a, b",
        # non-AND conditions remain rejected in grouped queries
        "SELECT a, count(*), sum(b) FROM t WHERE x = 1 OR y = 2 GROUP BY a",
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
        "SELECT a FROM sales OR a = 1",        # stray OR
        "SELECT a FROM sales WHERE a = 1 OR b = 2",
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
