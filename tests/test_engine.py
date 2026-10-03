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


# ---------- group by ----------

def test_group_by_basic(data_dir):
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount) FROM sales WHERE id != 4 GROUP BY region",
    )
    assert result["columns"] == ["region", "count(*)", "sum(amount)"]
    # First-appearance order in the filtered CSV rows: east (row 1), west (row 2).
    assert result["rows"] == [["east", 2, 40], ["west", 2, 18.25]]
    assert result["row_count"] == 2


def test_group_by_count_only(data_dir):
    result = execute(str(data_dir), "SELECT region, count(*) FROM sales GROUP BY region")
    assert result["columns"] == ["region", "count(*)"]
    assert result["rows"] == [["east", 3], ["west", 2]]


def test_group_by_first_appearance_order(data_dir):
    result = execute(str(data_dir), "SELECT note, count(*) FROM sales WHERE id >= 2 GROUP BY note")
    assert [r[0] for r in result["rows"]] == ["b", "", "x", "y"]


def test_group_by_numeric_keys_merge(data_dir):
    # 1 and 1.0 are the same decimal value, so they share a group; the key
    # renders as a JSON integer under the projection rules.
    (data_dir / "nums.csv").write_text(
        "k,v\n1,10\n1.0,20\n2,30\n1.00,40\n",
        encoding="utf-8",
    )
    result = execute(str(data_dir), "SELECT k, count(*), sum(v) FROM nums GROUP BY k")
    assert result["rows"] == [[1, 3, 70], [2, 1, 30]]
    assert render(result) == '{"columns":["k","count(*)","sum(v)"],"rows":[[1,3,70],[2,1,30]],"row_count":2}'


def test_group_by_non_integer_key_and_sum(data_dir):
    (data_dir / "frac.csv").write_text(
        "k,v\n1.5,0.1\nx,2\n1.50,0.2\n",
        encoding="utf-8",
    )
    result = execute(str(data_dir), "SELECT k, count(*), sum(v) FROM frac GROUP BY k")
    assert result["rows"] == [[Decimal("1.5"), 2, Decimal("0.3")], ["x", 1, 2]]
    assert render(result) == '{"columns":["k","count(*)","sum(v)"],"rows":[[1.5,2,0.3],["x",1,2]],"row_count":2}'


def test_group_by_empty_result(data_dir):
    result = execute(str(data_dir), "SELECT region, count(*) FROM sales WHERE id > 100 GROUP BY region")
    assert result == {"columns": ["region", "count(*)"], "rows": [], "row_count": 0}


def test_group_by_unreferenced_column_ignored(tmp_path):
    (tmp_path / "t.csv").write_text(
        "g,a,noise\n"
        "x,1,not-a-number!!!\n"
        "y,2,####\n"
        "x,3,???\n",
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT g, count(*), sum(a) FROM t GROUP BY g")
    assert result["rows"] == [["x", 2, 4], ["y", 1, 2]]


def test_group_by_sum_rejects_non_number(data_dir):
    with pytest.raises(ValueError):
        execute(str(data_dir), "SELECT region, count(*), sum(amount) FROM sales GROUP BY region")
    # Row 4 (amount="abc") is filtered out, so the sum succeeds.
    result = execute(
        str(data_dir),
        "SELECT region, count(*), sum(amount) FROM sales WHERE id != 4 GROUP BY region",
    )
    assert result["rows"][0] == ["east", 2, 40]


def test_group_by_spans_batches(tmp_path):
    n = BATCH_SIZE * 2 + 7
    (tmp_path / "big.csv").write_text(
        "g,v\n" + "".join(f"{'even' if i % 2 == 0 else 'odd'},{i}\n" for i in range(n)),
        encoding="utf-8",
    )
    result = execute(str(tmp_path), "SELECT g, count(*), sum(v) FROM big GROUP BY g")
    evens = sum(range(0, n, 2))
    odds = sum(range(1, n, 2))
    assert result["rows"] == [
        ["even", len(range(0, n, 2)), evens],
        ["odd", len(range(1, n, 2)), odds],
    ]


def test_group_by_unknown_column_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        execute(str(data_dir), "SELECT missing, count(*) FROM sales GROUP BY missing")
    with pytest.raises(KeyError):
        execute(str(data_dir), "SELECT region, count(*), sum(missing) FROM sales GROUP BY region")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM sales GROUP BY a",                    # no aggregates
        "SELECT region, sum(amount) FROM sales GROUP BY region",  # missing count(*)
        "SELECT count(*), sum(id) FROM sales GROUP BY region",    # group column not selected
        "SELECT id, count(*) FROM sales GROUP BY region",    # select/group mismatch
        "SELECT region, count(*), note FROM sales GROUP BY region",  # extra plain column
        "SELECT region, count(*), sum(id), sum(id) FROM sales GROUP BY region",  # duplicate aggregate
        "SELECT region, count(*), count(*) FROM sales GROUP BY region",          # duplicate count
        "SELECT region, region, count(*) FROM sales GROUP BY region",            # repeated group column
        "SELECT region, count(*) FROM sales GROUP BY region ORDER BY region",    # trailing clause
        "SELECT region, count(*) FROM sales GROUP BY",       # missing group column
        "SELECT region, count(*) FROM sales GROUP BY 1",     # non-identifier key
    ],
)
def test_bad_group_by_raises_valueerror(data_dir, sql):
    with pytest.raises(ValueError):
        execute(str(data_dir), sql)


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
