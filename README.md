# OLAP Turbo

列式 OLAP 查询加速引擎：列裁剪、谓词下推与向量化执行。

## 范围

本仓库从零开始实现上述方向的可用工具，不依赖外部同类实现。

## 状态

首个可查询基线：CSV 表上的 SELECT/WHERE，支持列裁剪、谓词先于投影的批处理向量化执行。

## 用法

```
python -m olap_turbo --data-dir <目录> --query <SQL>
```

- 表为 `<目录>/<表名>.csv`，表名为去掉 `.csv` 的文件名；首行为列名，后续每行为记录。
- 普通查询：`SELECT 列 [, 列 ...] FROM 表 [WHERE 条件]`，输出 `columns`、`rows`、`row_count`。
- 聚合查询：`SELECT count(*) [, sum(列) ...] FROM 表 [WHERE 条件]`，`columns` 为原始表达式文本，`rows` 为单行，`row_count` 恒为 1。
- 分组查询：`SELECT 分组列, count(*) [, sum(列) ...] FROM 表 [WHERE 条件] GROUP BY 分组列`；先按 WHERE 过滤再分组，扫描只读取分组列、聚合列与条件列。`rows` 每行依次为分组键、`count(*)` 与各 `sum(列)`，`row_count` 为分组数；分组按过滤后 CSV 行序中键首次出现的先后输出，空结果 `rows` 为 `[]`、`row_count` 为 0。分组键能读成有限十进制数时按数值精确比较（`1` 与 `1.0` 同组）并按投影口径输出 JSON 数值，否则按原文输出 JSON 字符串。
- 条件为 `列 op 值` 以 `AND` 组合；`op` 为 `=`、`!=`、`>`、`>=`、`<`、`<=`；值为不带引号的数字或双引号字符串。
- 数字按十进制精确比较与求和，输出为 JSON 数值；双引号值按文本精确比较，输出为 JSON 字符串。
- 结果以 CSV 原始行序、按 SELECT 列序输出为单个 JSON 对象。

不支持的写法（OR、其他函数、通配符列、其他子句、缺少 FROM、分组查询中混入普通列或重复聚合、SELECT 分组列与 GROUP BY 不一致等）抛出 `ValueError`；表文件不存在抛出 `FileNotFoundError`；引用不存在的列抛出 `KeyError`；字段无法按十进制数读取抛出 `ValueError`。

## 约定

- 公开行为以 README 与源码为准。
- 后续需求在此基线上增量实现。
