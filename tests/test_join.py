"""Tests for single INNER JOIN queries."""

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
    (tmp_path / "users.csv").write_text(
        "id,name,city\n"
        "1,ann,east\n"
        "2,bob,west\n"
        "3,cid,east\n",
        encoding="utf-8",
    )
    (tmp_path / "orders.csv").write_text(
        "oid,uid,amount\n"
        "10,1,5\n"
        "11,1,7.5\n"
        "12,2,abc\n"
        "13,9,1\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------- basic join ----------

def test_join_qualified_projection(data_dir):
    result = execute(
        str(data_dir),
        "SELECT users.id, orders.amount FROM users "
        "INNER JOIN orders ON users.id = orders.uid",
    )
    assert result["columns"] == ["users.id", "orders.amount"]
    # uid 9 matches nothing; user 3 matches nothing.
    assert result["rows"] == [[1, 5], [1, Decimal("7.5")], [2, "abc"]]
    assert result["row_count"] == 3
    assert render(result) == (
        '{"columns":["users.id","orders.amount"],"rows":'
        '[[1,5],[1,7.5],[2,"abc"]],"row_count":3}'
    )


def test_join_unqualified_unique_columns(data_dir):
    result = execute(
        str(data_dir),
        "SELECT name, amount FROM users INNER JOIN orders ON id = uid",
    )
    assert result["columns"] == ["name", "amount"]
    assert result["rows"] == [["ann", 5], ["ann", Decimal("7.5")], ["bob", "abc"]]
    assert result["row_count"] == 3


def test_join_mixed_qualified_and_unqualified(data_dir):
    result = execute(
        str(data_dir),
        'SELECT users.name, amount FROM users '
        'INNER JOIN orders ON users.id = uid WHERE city = "east"',
    )
    assert result["columns"] == ["users.name", "amount"]
    assert result["rows"] == [["ann", 5], ["ann", Decimal("7.5")]]


def test_join_expansion_order(data_dir):
    # Every left row expands to all matching right rows in right CSV
    # order; left rows expand in left CSV order.
    result = execute(
        str(data_dir),
        "SELECT id, oid FROM users INNER JOIN orders ON id = uid",
    )
    assert result["rows"] == [[1, 10], [1, 11], [2, 12]]


def test_join_on_refs_either_order(data_dir):
    result = execute(
        str(data_dir),
        "SELECT name FROM users INNER JOIN orders ON orders.uid = users.id",
    )
    assert result["rows"] == [["ann"], ["ann"], ["bob"]]


def test_join_empty_result_keeps_columns(data_dir):
    result = execute(
        str(data_dir),
        "SELECT id, oid FROM users INNER JOIN orders ON id = uid WHERE id > 100",
    )
    assert result == {"columns": ["id", "oid"], "rows": [], "row_count": 0}


def test_join_empty_tables(tmp_path):
    (tmp_path / "a.csv").write_text("k,v\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("k,w\n1,2\n", encoding="utf-8")
    result = execute(
        str(tmp_path), "SELECT a.k, b.w FROM a INNER JOIN b ON a.k = b.k"
    )
    assert result == {"columns": ["a.k", "b.w"], "rows": [], "row_count": 0}
    result = execute(
        str(tmp_path), "SELECT b.k, b.w FROM b INNER JOIN a ON b.k = a.k"
    )
    assert result == {"columns": ["b.k", "b.w"], "rows": [], "row_count": 0}


def test_join_keywords_case_insensitive(data_dir):
    result = execute(
        str(data_dir),
        "select name from users inner join orders on id = uid",
    )
    assert result["rows"] == [["ann"], ["ann"], ["bob"]]


# ---------- join key matching ----------

def test_join_numeric_keys_match_by_exact_value(tmp_path):
    # 1, 1.0 and 01 are the same finite decimal and join together.
    (tmp_path / "l.csv").write_text("k,v\n1,a\n2,b\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text(
        "k,w\n1.0,x\n01,y\n1.10,z\n", encoding="utf-8"
    )
    result = execute(
        str(tmp_path), "SELECT v, w FROM l INNER JOIN r ON l.k = r.k"
    )
    assert result["rows"] == [["a", "x"], ["a", "y"]]


def test_join_text_keys_exact_no_cross_coercion(tmp_path):
    (tmp_path / "l.csv").write_text("k,v\n1,a\nx,b\n,c\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text(
        "k,w\n1a,p\nx,q\n1.0,r\n,s\n", encoding="utf-8"
    )
    result = execute(
        str(tmp_path), "SELECT v, w FROM l INNER JOIN r ON l.k = r.k"
    )
    # "1" (decimal) never matches text "1a"; "1" matches "1.0" (decimal);
    # text "x" matches "x"; empty matches empty.
    assert result["rows"] == [["a", "r"], ["b", "q"], ["c", "s"]]


def test_join_key_columns_not_projected(tmp_path):
    (tmp_path / "l.csv").write_text("k,v\n1,a\n2,b\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text("k,w\n1,x\n1,y\n", encoding="utf-8")
    result = execute(
        str(tmp_path), "SELECT w FROM l INNER JOIN r ON l.k = r.k"
    )
    assert result["columns"] == ["w"]
    assert result["rows"] == [["x"], ["y"]]


# ---------- WHERE / ORDER BY / LIMIT on joins ----------

def test_join_where_runs_after_join(data_dir):
    # WHERE reads joined rows: the "abc" amount is never numerically
    # compared once its row is filtered by the text predicate.
    result = execute(
        str(data_dir),
        'SELECT name, amount FROM users INNER JOIN orders ON id = uid '
        'WHERE city = "west"',
    )
    assert result["rows"] == [["bob", "abc"]]
    result = execute(
        str(data_dir),
        "SELECT name, oid FROM users INNER JOIN orders ON id = uid "
        "WHERE oid >= 11",
    )
    assert result["rows"] == [["ann", 11], ["bob", 12]]


def test_join_where_boolean_tree(data_dir):
    result = execute(
        str(data_dir),
        'SELECT name, oid FROM users INNER JOIN orders ON id = uid '
        'WHERE city = "east" OR oid = 12',
    )
    assert result["rows"] == [["ann", 10], ["ann", 11], ["bob", 12]]


def test_join_where_numeric_cell_validated_eagerly(data_dir):
    # Row (bob, 12) has amount "abc": a numeric predicate on amount reads
    # every joined row, even those another OR branch already settles.
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT name FROM users INNER JOIN orders ON id = uid "
            "WHERE oid = 12 OR amount >= 0",
        )


def test_join_order_by_qualified(data_dir):
    result = execute(
        str(data_dir),
        "SELECT users.name, orders.oid FROM users "
        "INNER JOIN orders ON users.id = orders.uid ORDER BY orders.oid DESC",
    )
    assert result["rows"] == [["bob", 12], ["ann", 11], ["ann", 10]]


def test_join_order_by_unqualified_and_ties_stable(data_dir):
    # Ties on amount keep the join expansion order (left CSV, then right
    # CSV order).
    result = execute(
        str(data_dir),
        "SELECT name, oid, amount FROM users INNER JOIN orders ON id = uid "
        "ORDER BY amount",
    )
    assert [r[1] for r in result["rows"]] == [10, 11, 12]
    result = execute(
        str(data_dir),
        "SELECT name, oid FROM users INNER JOIN orders ON id = uid "
        "ORDER BY name, oid DESC",
    )
    assert result["rows"] == [["ann", 11], ["ann", 10], ["bob", 12]]


def test_join_order_by_decimal_ties_keep_join_order(tmp_path):
    (tmp_path / "l.csv").write_text("k,v\n1,a\n1.0,b\n", encoding="utf-8")
    (tmp_path / "r.csv").write_text("k,w\n1,x\n", encoding="utf-8")
    result = execute(
        str(tmp_path),
        "SELECT l.k, v, w FROM l INNER JOIN r ON l.k = r.k ORDER BY l.k",
    )
    # 1 and 1.0 tie; the join order (left CSV order) is preserved.
    assert result["rows"] == [[1, "a", "x"], [Decimal("1.0"), "b", "x"]]


def test_join_limit(data_dir):
    result = execute(
        str(data_dir),
        "SELECT name, oid FROM users INNER JOIN orders ON id = uid LIMIT 2",
    )
    assert result == {
        "columns": ["name", "oid"],
        "rows": [["ann", 10], ["ann", 11]],
        "row_count": 2,
    }


def test_join_limit_zero(data_dir):
    result = execute(
        str(data_dir),
        "SELECT users.id, orders.amount FROM users "
        "INNER JOIN orders ON users.id = orders.uid LIMIT 0",
    )
    assert result == {
        "columns": ["users.id", "orders.amount"],
        "rows": [],
        "row_count": 0,
    }


def test_join_order_by_then_limit(data_dir):
    result = execute(
        str(data_dir),
        "SELECT name, oid FROM users INNER JOIN orders ON id = uid "
        "ORDER BY oid DESC LIMIT 1",
    )
    assert result["rows"] == [["bob", 12]]
    assert result["row_count"] == 1


# ---------- column pruning / batching ----------

def test_join_unreferenced_garbage_columns_ignored(tmp_path):
    (tmp_path / "l.csv").write_text(
        "k,v,noise\n1,a,###\n2,b,not-a-number\n", encoding="utf-8"
    )
    (tmp_path / "r.csv").write_text(
        "k,w,junk\n1,x,@@@\n1,y,???\n", encoding="utf-8"
    )
    result = execute(
        str(tmp_path), "SELECT v, w FROM l INNER JOIN r ON l.k = r.k"
    )
    assert result["rows"] == [["a", "x"], ["a", "y"]]


def test_join_across_batch_sizes(tmp_path, monkeypatch):
    import olap_turbo.engine as engine_mod

    n = BATCH_SIZE * 2 + 7
    (tmp_path / "l.csv").write_text(
        "k,v\n" + "".join(f"{i % 10},{i}\n" for i in range(n)),
        encoding="utf-8",
    )
    (tmp_path / "r.csv").write_text(
        "k,w\n" + "".join(f"{i},{i * 2}\n" for i in range(10)),
        encoding="utf-8",
    )
    sql = "SELECT v, w FROM l INNER JOIN r ON l.k = r.k ORDER BY v LIMIT 5"
    expected = execute(str(tmp_path), sql)
    assert expected["row_count"] == 5
    full = execute(str(tmp_path), "SELECT v, w FROM l INNER JOIN r ON l.k = r.k")
    assert full["row_count"] == n
    for size in (1, 3, 7, BATCH_SIZE + 1):
        monkeypatch.setattr(engine_mod, "BATCH_SIZE", size)
        assert execute(str(tmp_path), sql) == expected
        assert (
            execute(str(tmp_path), "SELECT v, w FROM l INNER JOIN r ON l.k = r.k")
            == full
        )


# ---------- parse shape ----------

def test_parse_sql_join_shape():
    q = parse_sql(
        "SELECT a.x, y FROM a INNER JOIN b ON a.x = b.x WHERE y > 1 "
        "ORDER BY a.x LIMIT 3"
    )
    assert q.is_join
    assert q.table == "a"
    assert q.right_table == "b"
    left_ref, right_ref = q.join_on
    assert (left_ref.qualifier, left_ref.column) == ("a", "x")
    assert (right_ref.qualifier, right_ref.column) == ("b", "x")
    assert [item.text for item in q.select] == ["a.x", "y"]
    assert q.limit == 3
    assert not parse_sql("SELECT a FROM t").is_join


# ---------- error contract ----------

def test_join_missing_table_raises_filenotfound(data_dir):
    with pytest.raises(FileNotFoundError):
        execute(str(data_dir), "SELECT id FROM nope INNER JOIN orders ON id = uid")
    with pytest.raises(FileNotFoundError):
        execute(str(data_dir), "SELECT id FROM users INNER JOIN nope ON id = uid")


def test_join_unknown_column_raises_keyerror(data_dir):
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT missing FROM users INNER JOIN orders ON id = uid",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT users.missing FROM users INNER JOIN orders ON id = uid",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT nope.id FROM users INNER JOIN orders ON id = uid",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT id FROM users INNER JOIN orders ON id = missing",
        )
    with pytest.raises(KeyError):
        execute(
            str(data_dir),
            "SELECT id FROM users INNER JOIN orders ON id = uid "
            "WHERE missing = 1",
        )


def test_join_ambiguous_unqualified_column_raises_valueerror(tmp_path):
    (tmp_path / "a.csv").write_text("k,v\n1,a\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("k,w\n1,x\n", encoding="utf-8")
    # k exists in both tables: unqualified references are ambiguous.
    with pytest.raises(ValueError):
        execute(str(tmp_path), "SELECT k FROM a INNER JOIN b ON a.k = b.k")
    with pytest.raises(ValueError):
        execute(
            str(tmp_path),
            "SELECT v FROM a INNER JOIN b ON k = k",
        )
    with pytest.raises(ValueError):
        execute(
            str(tmp_path),
            "SELECT v FROM a INNER JOIN b ON a.k = b.k WHERE k = 1",
        )
    with pytest.raises(ValueError):
        execute(
            str(tmp_path),
            "SELECT a.k FROM a INNER JOIN b ON a.k = b.k ORDER BY k",
        )
    # Qualified references to the same names are fine.
    result = execute(
        str(tmp_path),
        "SELECT a.k, b.k FROM a INNER JOIN b ON a.k = b.k ORDER BY a.k",
    )
    assert result["rows"] == [[1, 1]]


def test_join_same_table_raises_valueerror(data_dir):
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT id FROM users INNER JOIN users ON id = id",
        )


@pytest.mark.parametrize(
    "sql",
    [
        # ON must be exactly one cross-table column equality
        "SELECT id FROM users INNER JOIN orders ON id = 1",
        "SELECT id FROM users INNER JOIN orders ON 1 = uid",
        'SELECT id FROM users INNER JOIN orders ON id = "x"',
        "SELECT id FROM users INNER JOIN orders ON users.id = users.id",
        "SELECT id FROM users INNER JOIN orders ON orders.uid = orders.oid",
        "SELECT id FROM users INNER JOIN orders ON id > uid",
        "SELECT id FROM users INNER JOIN orders ON id != uid",
        "SELECT id FROM users INNER JOIN orders ON id = uid AND id = 1",
        "SELECT id FROM users INNER JOIN orders ON id = uid OR id = 1",
        "SELECT id FROM users INNER JOIN orders ON (id = uid)",
        "SELECT id FROM users INNER JOIN orders ON id = uid = 1",
        "SELECT id FROM users INNER JOIN orders ON id",
        "SELECT id FROM users INNER JOIN orders",
        "SELECT id FROM users INNER JOIN orders WHERE id = 1",
        # aliases are not supported
        "SELECT u.id FROM users u INNER JOIN orders ON u.id = uid",
        "SELECT u.id FROM users AS u INNER JOIN orders ON u.id = uid",
        "SELECT id FROM users INNER JOIN orders o ON id = o.uid",
        # only INNER JOIN, exactly once
        "SELECT id FROM users LEFT JOIN orders ON id = uid",
        "SELECT id FROM users LEFT OUTER JOIN orders ON id = uid",
        "SELECT id FROM users RIGHT JOIN orders ON id = uid",
        "SELECT id FROM users JOIN orders ON id = uid",
        "SELECT id FROM users CROSS JOIN orders",
        "SELECT id FROM users INNER JOIN orders ON id = uid "
        "INNER JOIN users u2 ON id = u2.id",
        "SELECT id FROM users INNER JOIN orders ON id = uid INNER JOIN t ON id = t.id",
        # no aggregates / GROUP BY / HAVING on join queries
        "SELECT count(*) FROM users INNER JOIN orders ON id = uid",
        "SELECT count(*), sum(amount) FROM users INNER JOIN orders ON id = uid",
        "SELECT sum(amount) FROM users INNER JOIN orders ON id = uid",
        "SELECT id FROM users INNER JOIN orders ON id = uid GROUP BY id",
        "SELECT city, count(*), sum(amount) FROM users "
        "INNER JOIN orders ON id = uid GROUP BY city",
        "SELECT id FROM users INNER JOIN orders ON id = uid HAVING count(*) > 1",
        # wildcard stays rejected
        "SELECT * FROM users INNER JOIN orders ON id = uid",
    ],
)
def test_bad_join_sql_raises_valueerror(data_dir, sql):
    with pytest.raises(ValueError):
        execute(str(data_dir), sql)


def test_join_numeric_where_on_text_cell_raises(data_dir):
    with pytest.raises(ValueError):
        execute(
            str(data_dir),
            "SELECT name FROM users INNER JOIN orders ON id = uid "
            "WHERE amount >= 0",
        )


def test_join_no_derived_files_written(data_dir):
    before = {p.name for p in data_dir.iterdir()}
    execute(str(data_dir), "SELECT name, oid FROM users INNER JOIN orders ON id = uid")
    assert {p.name for p in data_dir.iterdir()} == before


def test_join_cli(data_dir):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "olap_turbo",
            "--data-dir",
            str(data_dir),
            "--query",
            "SELECT users.name, orders.oid FROM users "
            "INNER JOIN orders ON users.id = orders.uid ORDER BY orders.oid",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload == {
        "columns": ["users.name", "orders.oid"],
        "rows": [["ann", 10], ["ann", 11], ["bob", 12]],
        "row_count": 3,
    }
