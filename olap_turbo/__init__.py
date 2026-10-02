"""OLAP Turbo: 列裁剪、谓词下推与向量化执行的基线实现。"""

from .engine import run_query

__all__ = ["run_query"]
__version__ = "0.1.0"
