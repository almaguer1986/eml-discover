"""eml-discover — identify famous mathematical formulas from
arbitrary SymPy expressions.

    >>> from eml_discover import identify
    >>> import sympy as sp
    >>> u = sp.Symbol("u")
    >>> matches = identify(1 / (1 + sp.exp(-u)))
    >>> for m in matches:
    ...     print(m.formula.name, "—", m.confidence)
    sigmoid (canonical) — exact

The registry currently covers ~17 named formulas across statistics,
ML activations, trig/hyperbolic identities, physics, and finance.
Add more via :data:`eml_discover.registry.FORMULAS`.
"""
from __future__ import annotations

from .identify import Match, identify
from .registry import FORMULAS, Formula, by_name, list_all, list_by_domain

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "identify",
    "Match",
    "Formula",
    "FORMULAS",
    "by_name",
    "list_all",
    "list_by_domain",
]
