"""Identify a user-supplied SymPy expression against the formula
registry.

Two-stage match:

1. **Fingerprint pre-filter** (cheap, structural). Compute the
   :func:`eml_cost.fingerprint` of the user's expression and the same
   for every registry template. Templates whose fingerprint matches
   on the cost axes (the leading ``p…-d…-w…-c…`` block, ignoring the
   tail hash) become candidates.
2. **Symbolic equivalence** (expensive, true). For each candidate, try
   variable renames over the candidate's free symbols and check
   ``sp.simplify(user_expr - renamed_template) == 0``. Matches are
   ranked by how much rename freedom was needed (zero-rename matches
   come first).

Returns a list of :class:`Match` ordered by confidence. Empty list
means "we don't recognize this", not "this is invalid".
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable

import sympy as sp

from eml_cost import fingerprint as _fp

from .registry import FORMULAS, Formula


_AXES_RE = re.compile(r"^(p\d+-d\d+-w\d+-c-?\d+)-h[0-9a-f]+$")


@dataclass(frozen=True)
class Match:
    """A formula that matches a user-supplied expression.

    Attributes
    ----------
    formula:
        The :class:`Formula` registry entry that matched.
    confidence:
        ``"exact"`` (full symbolic equivalence after symbol rename),
        ``"axes"`` (fingerprint cost-axes matched but symbolic
        equivalence not established), or ``"identical"`` (exact
        Python-level equality, no rename needed).
    rename:
        Mapping from canonical-template symbol -> user-expr symbol
        used when ``confidence == "exact"``. Empty for ``"axes"``
        and ``"identical"``.
    """

    formula: Formula
    confidence: str
    rename: dict[sp.Symbol, sp.Symbol]


def _axes(fp_value: str) -> str:
    """Strip the trailing tail hash to compare cost axes only."""
    m = _AXES_RE.match(fp_value)
    return m.group(1) if m else fp_value


def _rename_attempts(
    template_syms: list[sp.Symbol], user_syms: list[sp.Symbol]
) -> Iterable[dict[sp.Symbol, sp.Symbol]]:
    """Yield candidate rename maps from template symbols to user
    symbols. Falls back to the empty map when arities mismatch."""
    if not template_syms:
        yield {}
        return
    if len(template_syms) > len(user_syms):
        # User expr has fewer free symbols than the template — match
        # only if some template symbols become constants (skip).
        return
    for chosen in itertools.permutations(user_syms, len(template_syms)):
        yield dict(zip(template_syms, chosen))


def _try_match(template: sp.Basic, user: sp.Basic, rename: dict) -> bool:
    """True when template.subs(rename) is symbolically equal to user."""
    try:
        renamed = template.subs(rename) if rename else template
        diff = sp.simplify(renamed - user)
        return diff == 0
    except (TypeError, ValueError, RecursionError):
        return False


def identify(expr: sp.Basic | str, *, max_results: int = 5) -> list[Match]:
    """Return registry formulas that match ``expr``, newest-first by
    confidence then by registry order.

    ``expr`` may be a SymPy expression or a string sympify-able into
    one. Returns an empty list when no match is found.
    """
    if isinstance(expr, str):
        expr = sp.sympify(expr)

    user_fp = _axes(_fp(expr))
    user_syms = sorted(
        (s for s in expr.free_symbols if isinstance(s, sp.Symbol)),
        key=lambda s: s.name,
    )

    matches: list[Match] = []
    seen_names: set[str] = set()

    for formula in FORMULAS:
        if formula.name in seen_names:
            continue
        try:
            template = formula.expression_factory()
        except Exception:
            continue

        # Identical-expression shortcut.
        if expr == template:
            matches.append(Match(formula=formula, confidence="identical", rename={}))
            seen_names.add(formula.name)
            continue

        template_fp = _axes(_fp(template))
        if template_fp != user_fp:
            continue   # Cost axes differ — not the same Pfaffian class.

        template_syms = sorted(
            (s for s in template.free_symbols if isinstance(s, sp.Symbol)),
            key=lambda s: s.name,
        )

        # Try renames; first one that satisfies symbolic equivalence wins.
        matched = False
        for rename in _rename_attempts(template_syms, user_syms):
            if _try_match(template, expr, rename):
                matches.append(Match(formula=formula, confidence="exact", rename=rename))
                seen_names.add(formula.name)
                matched = True
                break

        if not matched:
            # Fingerprint axes match but no rename produced symbolic
            # equivalence — surface as a weaker "axes" match.
            matches.append(Match(formula=formula, confidence="axes", rename={}))
            seen_names.add(formula.name)

    # Order: identical > exact > axes; preserve registry order within tier.
    rank = {"identical": 0, "exact": 1, "axes": 2}
    matches.sort(key=lambda m: rank.get(m.confidence, 9))
    return matches[:max_results]
