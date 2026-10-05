"""Tests for the explain() query-plan entry point and the --explain CLI flag."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from olap_turbo import execute, explain
from olap_turbo.engine import render

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "sales.csv").write_text(
        "id,region,amount,note\n"
        "1,east,10,a\n"
        "2,west,20.5,b\n"
        "3,east,abc,\n"
        "4,west,-2.25,y\n",
        encoding="utf-8",
    )
    (tmp_path / "regions.csv").write_text(
        "rid,city,code\n"
        "1,NY,100\n"
        "3,LA,300\n"
        "3,SF,301\n",
        encoding="utf-8",
    )
    return tmp_path


PLAN_KEYS = [
    "query_type",
    "tables",
    "scan_columns",
    "where_columns",
    "select_columns",
    "group_columns",
    "aggregates",
    "join",
    "order_by",
    "limit",
]


# ---------- projection ----------

def test_projection_plan_shape(data_dir):
    plan = explain(str(data_dir), 'SELECT amount, id FROM sales WHERE region = "east"')
    assert list(plan.keys()) == PLAN_KEYS
    assert plan["query_type"] == "projection"
    assert plan["tables"] == ["sales"]
    # Header order, not SELECT/WHERE order; note is neither selected nor filtered.
    assert plan["scan_columns"] == ["id", "region", "amount"]
    assert plan["where_columns"] == ["region"]
    assert plan["select_columns"] == ["amount", "id"]
    assert plan["group_columns"] == []
    assert plan["aggregates"] == []
    assert plan["join"] is None
    assert plan["order_by"] == []
    assert plan["limit"] is None


def test_scan_columns_follow_csv_header_order(data_dir):
    plan = explain(str(data_dir), "SELECT note FROM sales WHERE amount > 1 AND id = 1")
    assert plan["scan_columns"] == ["id", "amount", "note"]


def test_where_columns_first_appearance_dedup(data_dir):
    plan = explain(
        str(data_dir),
        'SELECT id FROM sales WHERE region = "x" OR (id = 1 AND region = "y")',
    )
    assert plan["where_columns"] == ["region", "id"]


def test_select_columns_first_appearance_dedup(data_dir):
    plan = explain(str(data_dir), "SELECT id, region, id FROM sales")
    assert plan["select_columns"] == ["id", "region"]
    assert plan["scan_columns"] == ["id", "region"]


def test_order_by_and_limit_plan(data_dir):
    plan = explain(str(data_dir), "SELECT id, region FROM sales ORDER BY id DESC, region LIMIT 3")
    assert plan["order_by"] == [
        {"expression": "id", "direction": "DESC"},
        {"expression": "region", "direction": "ASC"},
    ]
    assert plan["limit"] == 3


def test_limit_zero_is_native_integer(data_dir):
    plan = explain(str(data_dir), "SELECT id FROM sales LIMIT 0")
    assert plan["limit"] == 0
    assert isinstance(plan["limit"], int)


def test_no_where_scans_only_selected_columns(data_dir):
    plan = explain(str(data_dir), "SELECT note FROM sales")
    assert plan["scan_columns"] == ["note"]
    assert plan["where_columns"] == []


# ---------- aggregates ----------

def test_global_aggregate_plan(data_dir):
    plan = explain(
        str(data_dir),
        "SELECT count(*), sum(amount), avg(amount), min(id), max(note) FROM sales "
        "WHERE id != 4 HAVING count(*) > 1 AND sum(amount) <= 100",
    )
    assert plan["query_type"] == "aggregate"
    assert plan["tables"] == ["sales"]
    # count(*) reads no column; note belongs only to max(note).
    assert plan["scan_columns"] == ["id", "amount", "note"]
    assert plan["where_columns"] == ["id"]
    assert plan["select_columns"] == []
    assert plan["group_columns"] == []
    assert plan["aggregates"] == [
        {"function": "count", "column": None, "text": "count(*)"},
        {"function": "sum", "column": "amount", "text": "sum(amount)"},
        {"function": "avg", "column": "amount", "text": "avg(amount)"},
        {"function": "min", "column": "id", "text": "min(id)"},
        {"function": "max", "column": "note", "text": "max(note)"},
    ]
    assert plan["join"] is None


def test_grouped_aggregate_plan(data_dir):
    plan = explain(
        str(data_dir),
        'SELECT region, id, count(*), sum(amount) FROM sales '
        'WHERE amount >= 0 OR note = "b" GROUP BY region, id '
        "HAVING count(*) >= 1 ORDER BY sum(amount) DESC, region LIMIT 5",
    )
    assert plan["query_type"] == "grouped_aggregate"
    assert plan["scan_columns"] == ["id", "region", "amount", "note"]
    assert plan["where_columns"] == ["amount", "note"]
    assert plan["select_columns"] == ["region", "id"]
    assert plan["group_columns"] == ["region", "id"]
    assert plan["aggregates"] == [
        {"function": "count", "column": None, "text": "count(*)"},
        {"function": "sum", "column": "amount", "text": "sum(amount)"},
    ]
    assert plan["order_by"] == [
        {"expression": "sum(amount)", "direction": "DESC"},
        {"expression": "region", "direction": "ASC"},
    ]
    assert plan["limit"] == 5


# ---------- joins ----------

def test_join_plan_qualified_columns(data_dir):
    plan = explain(
        str(data_dir),
        "SELECT id, sales.note, regions.city FROM sales "
        "INNER JOIN regions ON sales.id = regions.rid "
        "WHERE id >= 2 AND regions.code = 300 "
        "ORDER BY sales.id DESC, regions.city LIMIT 2",
    )
    assert plan["query_type"] == "join"
    assert plan["tables"] == ["sales", "regions"]
    # Left-table columns first (header order), then right-table columns.
    assert plan["scan_columns"] == [
        "sales.id",
        "sales.note",
        "regions.rid",
        "regions.city",
        "regions.code",
    ]
    assert plan["where_columns"] == ["sales.id", "regions.code"]
    # SELECT keeps the source text of each item.
    assert plan["select_columns"] == ["id", "sales.note", "regions.city"]
    assert plan["group_columns"] == []
    assert plan["aggregates"] == []
    assert plan["join"] == {
        "left_table": "sales",
        "right_table": "regions",
        "left_column": "id",
        "right_column": "rid",
    }
    assert plan["order_by"] == [
        {"expression": "sales.id", "direction": "DESC"},
        {"expression": "regions.city", "direction": "ASC"},
    ]
    assert plan["limit"] == 2


def test_join_plan_bare_on_and_select(data_dir):
    plan = explain(
        str(data_dir),
        "SELECT sales.id, regions.city FROM sales INNER JOIN regions ON id = rid",
    )
    assert plan["scan_columns"] == ["sales.id", "regions.rid", "regions.city"]
    assert plan["where_columns"] == []
    assert plan["select_columns"] == ["sales.id", "regions.city"]
    assert plan["join"] == {
        "left_table": "sales",
        "right_table": "regions",
        "left_column": "id",
        "right_column": "rid",
    }


def test_join_on_written_right_first_still_reports_from_order(data_dir):
    plan = explain(
        str(data_dir),
        "SELECT sales.id FROM sales INNER JOIN regions ON regions.rid = sales.id",
    )
    assert plan["join"]["left_column"] == "id"
    assert plan["join"]["right_column"] == "rid"


# ---------- headers only: no data-row reads ----------

def test_explain_reads_no_data_rows(tmp_path):
    # A valid header followed by malformed/non-numeric garbage rows.
    (tmp_path / "t.csv").write_text(
        "a,b,c\n"
        "not,a,row!!!\n"
        "\n"
        "####\n",
        encoding="utf-8",
    )
    # A numeric predicate and an aggregate over garbage cells still succeed.
    plan = explain(str(tmp_path), "SELECT a, c FROM t WHERE b >= 0 OR a = 1")
    assert plan["scan_columns"] == ["a", "b", "c"]
    assert plan["where_columns"] == ["b", "a"]
    plan = explain(str(tmp_path), "SELECT count(*), sum(b), min(c) FROM t")
    assert plan["aggregates"][0] == {"function": "count", "column": None, "text": "count(*)"}
    assert plan["scan_columns"] == ["b", "c"]


def test_explain_non_numeric_cells_do_not_fail(data_dir):
    plan = explain(str(data_dir), "SELECT id FROM sales WHERE amount >= 0")
    assert plan["scan_columns"] == ["id", "amount"]
    plan = explain(str(data_dir), "SELECT count(*), sum(amount) FROM sales")
    assert plan["aggregates"][1]["column"] == "amount"


def test_explain_does_not_modify_files(data_dir):
    before = {p.name: p.read_text() for p in data_dir.iterdir()}
    explain(str(data_dir), "SELECT id, region FROM sales WHERE amount > 1 ORDER BY id LIMIT 2")
    explain(
        str(data_dir),
        "SELECT sales.id FROM sales INNER JOIN regions ON sales.id = regions.rid",
    )
    after = {p.name: p.read_text() for p in data_dir.iterdir()}
    assert before == after


# ---------- error contract ----------

def test_explain_missing_table_raises_filenotfound(data_dir):
    with pytest.raises(FileNotFoundError):
        explain(str(data_dir), "SELECT a FROM nope")
    with pytest.raises(FileNotFoundError):
        explain(
            str(data_dir),
            "SELECT sales.id FROM sales INNER JOIN nope ON sales.id = nope.k",
        )


def test_explain_missing_column_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        explain(str(data_dir), "SELECT missing FROM sales")
    with pytest.raises(KeyError):
        explain(str(data_dir), "SELECT id FROM sales WHERE missing = 1")
    with pytest.raises(KeyError):
        explain(str(data_dir), "SELECT count(*), sum(missing) FROM sales")
    with pytest.raises(KeyError):
        explain(
            str(data_dir),
            "SELECT sales.id FROM sales INNER JOIN regions ON sales.nope = regions.rid",
        )
    with pytest.raises(KeyError):
        explain(
            str(data_dir),
            "SELECT id FROM sales INNER JOIN regions ON sales.id = regions.nope",
        )


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "SELECT * FROM sales",
        "SELECT a FROM sales WHERE",
        "DELETE FROM sales",
        "SELECT a, count(*) FROM sales",
        "SELECT id FROM sales GROUP BY id",  # invalid grouped shape
    ],
)
def test_explain_bad_sql_raises_valueerror(data_dir, sql):
    with pytest.raises(ValueError):
        explain(str(data_dir), sql)


def test_explain_ambiguous_join_column_raises_valueerror(tmp_path):
    (tmp_path / "l.csv").write_text("id,name\n1,a\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text("id,name\n1,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        explain(str(tmp_path), "SELECT name FROM l INNER JOIN r ON l.id = r.id")
    with pytest.raises(ValueError):
        explain(str(tmp_path), "SELECT l.name FROM l INNER JOIN r ON id = id")


def test_explain_join_order_by_unselected_raises_valueerror(data_dir):
    with pytest.raises(ValueError):
        explain(
            str(data_dir),
            "SELECT sales.id FROM sales INNER JOIN regions ON sales.id = regions.rid "
            "ORDER BY regions.city",
        )


def test_explain_join_duplicate_order_by_raises_valueerror(data_dir):
    with pytest.raises(ValueError):
        explain(
            str(data_dir),
            "SELECT sales.id, regions.city FROM sales INNER JOIN regions "
            "ON sales.id = regions.rid ORDER BY sales.id, id",
        )


# ---------- JSON rendering ----------

def test_plan_renders_as_one_json_object_without_rows(data_dir):
    payload = json.loads(
        render(explain(str(data_dir), "SELECT count(*) FROM sales LIMIT 0"))
    )
    assert set(payload.keys()) == set(PLAN_KEYS)
    assert "rows" not in payload
    assert "row_count" not in payload
    assert payload["aggregates"] == [
        {"function": "count", "column": None, "text": "count(*)"}
    ]
    assert payload["limit"] == 0
    assert payload["join"] is None


# ---------- execute / CLI stay unchanged ----------

def test_execute_unchanged(data_dir):
    result = execute(str(data_dir), 'SELECT id FROM sales WHERE region = "east"')
    assert result == {"columns": ["id"], "rows": [[1], [3]], "row_count": 2}


def test_cli_explain_outputs_only_plan(data_dir):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "olap_turbo",
            "--data-dir",
            str(data_dir),
            "--query",
            "SELECT region, count(*), sum(id) FROM sales GROUP BY region",
            "--explain",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload.keys()) == set(PLAN_KEYS)
    assert payload["query_type"] == "grouped_aggregate"
    assert payload["scan_columns"] == ["id", "region"]
    assert "rows" not in payload
    assert "row_count" not in payload
    assert proc.stderr == ""


def test_cli_without_explain_unchanged(data_dir):
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
    assert json.loads(proc.stdout) == {
        "columns": ["count(*)"],
        "rows": [[2]],
        "row_count": 1,
    }


def test_cli_explain_still_requires_query(data_dir):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "olap_turbo",
            "--data-dir",
            str(data_dir),
            "--explain",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--query" in proc.stderr
