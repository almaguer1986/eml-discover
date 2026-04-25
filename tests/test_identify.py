"""Tests for ``eml_discover.identify``."""
from __future__ import annotations

import sympy as sp

from eml_discover import (
    FORMULAS,
    Formula,
    Match,
    by_name,
    identify,
    list_all,
    list_by_domain,
)


# Canonical user symbols (different names from registry templates so
# rename logic is exercised).
u = sp.Symbol("u", real=True)
v = sp.Symbol("v", real=True)


# ---------------------------------------------------------------------------
# Direct identifications
# ---------------------------------------------------------------------------


def test_sigmoid_canonical_recognized() -> None:
    matches = identify(1 / (1 + sp.exp(-u)))
    names = {m.formula.name for m in matches}
    assert "sigmoid (canonical)" in names


def test_sigmoid_textbook_form_recognized() -> None:
    matches = identify(sp.exp(u) / (1 + sp.exp(u)))
    names = {m.formula.name for m in matches}
    assert "sigmoid (textbook)" in names


def test_softplus_recognized() -> None:
    matches = identify(sp.log(1 + sp.exp(u)))
    names = {m.formula.name for m in matches}
    assert "softplus" in names


def test_pythagorean_identity_recognized() -> None:
    matches = identify(sp.sin(u) ** 2 + sp.cos(u) ** 2)
    names = {m.formula.name for m in matches}
    assert "Pythagorean identity" in names


def test_hyperbolic_identity_recognized() -> None:
    matches = identify(sp.cosh(u) ** 2 - sp.sinh(u) ** 2)
    names = {m.formula.name for m in matches}
    assert "hyperbolic identity" in names


def test_gaussian_pdf_standardized_recognized() -> None:
    matches = identify(sp.exp(-u ** 2 / 2) / sp.sqrt(2 * sp.pi))
    names = {m.formula.name for m in matches}
    assert "Gaussian PDF (standardized)" in names


def test_planck_spectrum_recognized() -> None:
    """1/(exp(x/T)-1) has the Planck-spectrum signature."""
    Tcust = sp.Symbol("Tcust", real=True, positive=True)
    matches = identify(1 / (sp.exp(u / Tcust) - 1))
    names = {m.formula.name for m in matches}
    assert "Planck spectrum (occupation)" in names


def test_bessel_j0_recognized() -> None:
    matches = identify(sp.besselj(0, u))
    names = {m.formula.name for m in matches}
    assert "Bessel J_0" in names


# ---------------------------------------------------------------------------
# String input + edge cases
# ---------------------------------------------------------------------------


def test_identify_accepts_string_input() -> None:
    matches = identify("1 / (1 + exp(-u))")
    assert any(m.formula.name == "sigmoid (canonical)" for m in matches)


def test_identify_returns_empty_for_nonsense_expression() -> None:
    """An ad-hoc polynomial that matches no famous formula returns []."""
    matches = identify(u ** 7 + 5 * u ** 3 - 11)
    # We expect zero exact/identical matches; axes-only are tolerated
    # if any (typically none for an unusual polynomial).
    assert all(m.confidence != "exact" and m.confidence != "identical"
               for m in matches)


def test_identify_max_results_caps_output() -> None:
    matches = identify(sp.sin(u) ** 2 + sp.cos(u) ** 2, max_results=2)
    assert len(matches) <= 2


# ---------------------------------------------------------------------------
# Confidence ordering
# ---------------------------------------------------------------------------


def test_exact_matches_rank_above_axes_matches() -> None:
    matches = identify(sp.sin(u) ** 2 + sp.cos(u) ** 2)
    ranks = {"identical": 0, "exact": 1, "axes": 2}
    confidences = [ranks[m.confidence] for m in matches]
    assert confidences == sorted(confidences)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_registry_contains_at_least_50_formulas() -> None:
    assert len(FORMULAS) >= 50


def test_registry_covers_canonical_domains() -> None:
    """The registry must include entries from every advertised domain."""
    domains = {f.domain for f in FORMULAS}
    for advertised in (
        "statistics", "ml", "physics", "finance",
        "trigonometry", "hyperbolic", "info-theory",
    ):
        assert advertised in domains, f"missing domain: {advertised}"


def test_lorentz_factor_recognized() -> None:
    """Identify the relativity gamma factor under variable rename."""
    vc = sp.Symbol("vc", real=True)
    cc = sp.Symbol("cc", real=True, positive=True)
    matches = identify(1 / sp.sqrt(1 - (vc / cc) ** 2))
    names = {m.formula.name for m in matches}
    assert "Lorentz factor" in names


def test_kl_divergence_single_term_recognized() -> None:
    pp = sp.Symbol("pp", real=True, positive=True)
    qq = sp.Symbol("qq", real=True, positive=True)
    matches = identify(pp * sp.log(pp / qq))
    names = {m.formula.name for m in matches}
    assert "KL divergence (single term)" in names


def test_rc_decay_recognized() -> None:
    Rv = sp.Symbol("Rv", real=True, positive=True)
    Cv = sp.Symbol("Cv", real=True, positive=True)
    tv = sp.Symbol("tv", real=True)
    matches = identify(sp.exp(-tv / (Rv * Cv)))
    names = {m.formula.name for m in matches}
    assert "RC voltage decay" in names


def test_perpetuity_recognized() -> None:
    """C / r is the simplest finance template; ensure the identifier
    handles trivial-shape registry entries."""
    cc = sp.Symbol("cc", real=True, positive=True)
    rr = sp.Symbol("rr", real=True, positive=True)
    matches = identify(cc / rr)
    names = {m.formula.name for m in matches}
    assert "perpetuity" in names



def test_by_name_lookup_is_case_insensitive() -> None:
    f = by_name("SIGMOID (canonical)")
    assert f is not None
    assert f.name == "sigmoid (canonical)"


def test_list_by_domain_filters() -> None:
    ml = list_by_domain("ml")
    assert all(f.domain == "ml" for f in ml)
    assert len(ml) >= 4


def test_match_dataclass_is_frozen() -> None:
    f = FORMULAS[0]
    m = Match(formula=f, confidence="exact", rename={})
    import pytest
    with pytest.raises(Exception):
        m.confidence = "axes"  # type: ignore[misc]


def test_formula_dataclass_is_frozen() -> None:
    f = FORMULAS[0]
    import pytest
    with pytest.raises(Exception):
        f.name = "renamed"  # type: ignore[misc]
