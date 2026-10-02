"""OLAP Turbo: columnar OLAP query acceleration baseline.

Public entry point: :func:`olap_turbo.engine.execute`, or the module CLI
(``python -m olap_turbo --data-dir <dir> --query <sql>``).
"""

from .engine import execute

__all__ = ["execute"]
