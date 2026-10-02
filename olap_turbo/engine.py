"""向量化执行引擎。

读取 CSV（首行表头），按至多 ``BATCH_SIZE`` 行的批次只物化 SELECT 与
WHERE 涉及的列，在批次上做谓词过滤，再做投影或聚合。
"""

import csv
import os
from decimal import Decimal, InvalidOperation

from .sql_parser import parse_query

BATCH_SIZE = 1024


def _to_number(text):
    """把字段文本读成十进制数；不可转换时抛 ValueError。"""
    try:
        d = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"字段无法转换为十进制数: {text!r}")
    if not d.is_finite():
        raise ValueError(f"字段无法转换为十进制数: {text!r}")
    return d


def _number_to_json(d):
    """Decimal 转 JSON 原生数值：整数转 int，其余转 float。"""
    return int(d) if d == d.to_integral_value() else float(d)


def _field_to_output(text):
    """输出字段类型推断：能按十进制数读取则为数值，否则为字符串。"""
    try:
        return _number_to_json(_to_number(text))
    except ValueError:
        return text


def _compare(literal, op, field_text):
    if isinstance(literal, str):
        target = field_text  # 带引号：文本精确比较
    else:
        target = _to_number(field_text)  # 不带引号：按十进制数比较
    if op == "=":
        return target == literal
    if op == "!=":
        return target != literal
    if op == ">":
        return target > literal
    if op == ">=":
        return target >= literal
    if op == "<":
        return target < literal
    return target <= literal  # <=


def run_query(data_dir, sql):
    """在 ``data_dir`` 下的 CSV 表上执行 SQL，返回可直接 JSON 序列化的结果。"""
    plan = parse_query(sql)
    path = os.path.join(data_dir, plan.table + ".csv")
    # 表文件不存在时由 open 抛出 FileNotFoundError。
    f = open(path, "r", encoding="utf-8", newline="")
    try:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"表文件为空，缺少表头: {plan.table}")

        name_to_idx = {name: i for i, name in enumerate(header)}

        needed = [p.column for p in plan.predicates]
        if plan.is_aggregate:
            needed += [a[1] for a in plan.aggregates if a[0] == "sum"]
        else:
            needed += plan.select_columns

        # 引用不存在的列 -> KeyError（去重并保持首次出现顺序）。
        seen = set()
        needed_idx = []
        for col in needed:
            if col not in name_to_idx:
                raise KeyError(col)
            if col not in seen:
                seen.add(col)
                needed_idx.append((col, name_to_idx[col]))

        if plan.is_aggregate:
            return _run_aggregate(reader, plan, needed_idx)
        return _run_projection(reader, plan, needed_idx)
    finally:
        f.close()


def _read_batch(reader, needed_idx):
    """读取至多 BATCH_SIZE 行，返回 {列名: [字段文本, ...]} 与批次行数。"""
    vectors = {name: [] for name, _ in needed_idx}
    count = 0
    for row in reader:
        for name, idx in needed_idx:
            vectors[name].append(row[idx] if idx < len(row) else "")
        count += 1
        if count >= BATCH_SIZE:
            break
    return vectors, count


def _apply_predicates(predicates, vectors, size):
    """谓词下推：按 WHERE 顺序逐级收敛存活行掩码（短路式向量化过滤）。"""
    mask = [True] * size
    for p in predicates:
        col = vectors[p.column]
        for i in range(size):
            if mask[i]:
                mask[i] = _compare(p.value, p.op, col[i])
    return mask


def _run_projection(reader, plan, needed_idx):
    out_rows = []
    while True:
        vectors, size = _read_batch(reader, needed_idx)
        if size == 0:
            break
        mask = _apply_predicates(plan.predicates, vectors, size)
        projected = [vectors[name] for name in plan.select_columns]
        for i in range(size):
            if mask[i]:
                out_rows.append([_field_to_output(col[i]) for col in projected])

    return {
        "columns": list(plan.select_columns),
        "rows": out_rows,
        "row_count": len(out_rows),
    }


def _run_aggregate(reader, plan, needed_idx):
    count = 0
    sums = [Decimal(0)] * sum(1 for a in plan.aggregates if a[0] == "sum")
    while True:
        vectors, size = _read_batch(reader, needed_idx)
        if size == 0:
            break
        mask = _apply_predicates(plan.predicates, vectors, size)
        sum_cols = [vectors[a[1]] for a in plan.aggregates if a[0] == "sum"]
        for i in range(size):
            if not mask[i]:
                continue
            count += 1
            for k, col in enumerate(sum_cols):
                sums[k] += _to_number(col[i])  # 仅对命中行求和

    row = []
    si = 0
    for agg in plan.aggregates:
        if agg[0] == "count":
            row.append(count)
        else:
            row.append(_number_to_json(sums[si]))
            si += 1

    return {
        "columns": [agg[-1] for agg in plan.aggregates],
        "rows": [row],
        "row_count": 1,
    }
