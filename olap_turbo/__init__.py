"""OLAP Turbo: columnar OLAP query acceleration baseline.

Public entry points: :func:`olap_turbo.engine.execute` and
:func:`olap_turbo.engine.explain`, or the module CLI
(``python -m olap_turbo --data-dir <dir> --query <sql> [--explain]``).
"""

from .engine import execute, explain

__all__ = ["execute", "explain"]
