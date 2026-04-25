"""Curated registry of famous mathematical formulas.

Each entry pairs a SymPy template with a short name, a domain of
application, and a citation. The template uses the canonical symbol
names declared at module top so ``identify`` can rename the user's
free symbols and check structural equivalence cheaply.

To add a formula, append an entry to :data:`FORMULAS`. Keep the
template in canonical SymPy form (apply ``sp.simplify`` mentally
before inserting); ``identify`` will normalize the user's input
the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import sympy as sp


# Canonical symbols used in templates. These are the names ``identify``
# looks for and renames against the user's expression.
x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
t = sp.Symbol("t", real=True)
mu = sp.Symbol("mu", real=True)
sigma = sp.Symbol("sigma", real=True, positive=True)
T = sp.Symbol("T", real=True, positive=True)
S = sp.Symbol("S", real=True, positive=True)
K = sp.Symbol("K", real=True, positive=True)
r = sp.Symbol("r", real=True)
vol = sp.Symbol("vol", real=True, positive=True)
n = sp.Symbol("n", real=True)
E = sp.Symbol("E", real=True)
kT = sp.Symbol("kT", real=True, positive=True)


@dataclass(frozen=True)
class Formula:
    """A named formula in the registry.

    Attributes
    ----------
    name:
        Human-readable formula name (e.g., "Gaussian PDF").
    expression_factory:
        Callable returning the canonical SymPy template. We use a
        factory rather than a bare expression so that registry import
        is cheap and templates can reference module-level symbols.
    domain:
        Field of application: "statistics", "physics", "ml", etc.
    description:
        One-line description of what the formula represents.
    citation:
        Short citation (Wikipedia URL or canonical reference).
    """

    name: str
    expression_factory: Callable[[], sp.Basic]
    domain: str
    description: str
    citation: str = ""


FORMULAS: list[Formula] = [
    # --- Statistics ---
    Formula(
        name="Gaussian PDF (standardized)",
        expression_factory=lambda:
            sp.exp(-x**2 / 2) / sp.sqrt(2 * sp.pi),
        domain="statistics",
        description="Standard normal probability density.",
        citation="https://en.wikipedia.org/wiki/Normal_distribution",
    ),
    Formula(
        name="Gaussian PDF (general)",
        expression_factory=lambda:
            sp.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi)),
        domain="statistics",
        description="General normal pdf with mean mu, std sigma.",
        citation="https://en.wikipedia.org/wiki/Normal_distribution",
    ),

    # --- ML activations ---
    Formula(
        name="sigmoid (canonical)",
        expression_factory=lambda: 1 / (1 + sp.exp(-x)),
        domain="ml",
        description="Logistic sigmoid 1/(1+e^-x).",
        citation="https://en.wikipedia.org/wiki/Sigmoid_function",
    ),
    Formula(
        name="sigmoid (textbook)",
        expression_factory=lambda: sp.exp(x) / (1 + sp.exp(x)),
        domain="ml",
        description="Sigmoid in the textbook exp(x)/(1+exp(x)) form; "
                    "mathematically equal to the canonical sigmoid.",
        citation="https://en.wikipedia.org/wiki/Sigmoid_function",
    ),
    Formula(
        name="softplus",
        expression_factory=lambda: sp.log(1 + sp.exp(x)),
        domain="ml",
        description="Smooth ReLU approximation: log(1+e^x).",
        citation="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus",
    ),
    Formula(
        name="swish/SiLU",
        expression_factory=lambda: x / (1 + sp.exp(-x)),
        domain="ml",
        description="Self-gated activation x*sigmoid(x).",
        citation="Ramachandran et al., 2017",
    ),
    Formula(
        name="tanh (definition)",
        expression_factory=lambda:
            (sp.exp(x) - sp.exp(-x)) / (sp.exp(x) + sp.exp(-x)),
        domain="ml",
        description="Hyperbolic tangent in raw exp form.",
        citation="https://en.wikipedia.org/wiki/Hyperbolic_functions",
    ),
    Formula(
        name="GELU (erf form)",
        expression_factory=lambda:
            x * (1 + sp.erf(x / sp.sqrt(2))) / 2,
        domain="ml",
        description="Gaussian Error Linear Unit, exact erf form.",
        citation="Hendrycks & Gimpel, 2016",
    ),

    # --- Identities ---
    Formula(
        name="Pythagorean identity",
        expression_factory=lambda: sp.sin(x) ** 2 + sp.cos(x) ** 2,
        domain="trigonometry",
        description="sin^2 + cos^2 = 1.",
        citation="https://en.wikipedia.org/wiki/Pythagorean_trigonometric_identity",
    ),
    Formula(
        name="hyperbolic identity",
        expression_factory=lambda: sp.cosh(x) ** 2 - sp.sinh(x) ** 2,
        domain="hyperbolic",
        description="cosh^2 - sinh^2 = 1.",
        citation="https://en.wikipedia.org/wiki/Hyperbolic_functions",
    ),

    # --- Physics ---
    Formula(
        name="Boltzmann factor",
        expression_factory=lambda: sp.exp(-E / kT),
        domain="physics",
        description="Statistical-mechanics probability weight.",
        citation="https://en.wikipedia.org/wiki/Boltzmann_distribution",
    ),
    Formula(
        name="Planck spectrum (occupation)",
        expression_factory=lambda: 1 / (sp.exp(x / T) - 1),
        domain="physics",
        description="Bose-Einstein occupation; Planck radiation kernel.",
        citation="https://en.wikipedia.org/wiki/Planck%27s_law",
    ),
    Formula(
        name="Schrodinger plane wave",
        expression_factory=lambda: sp.exp(sp.I * x * t),
        domain="physics",
        description="Free-particle plane-wave eigenstate (units absorbed).",
        citation="https://en.wikipedia.org/wiki/Plane_wave",
    ),

    # --- Finance ---
    Formula(
        name="Black-Scholes d1",
        expression_factory=lambda:
            (sp.log(S / K) + (r + vol ** 2 / 2) * T) / (vol * sp.sqrt(T)),
        domain="finance",
        description="d1 from the Black-Scholes call-option formula.",
        citation="https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model",
    ),
    Formula(
        name="Black-Scholes d2",
        expression_factory=lambda:
            (sp.log(S / K) + (r - vol ** 2 / 2) * T) / (vol * sp.sqrt(T)),
        domain="finance",
        description="d2 from the Black-Scholes call-option formula.",
        citation="https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model",
    ),

    # --- Special transcendentals ---
    Formula(
        name="Bessel J_0",
        expression_factory=lambda: sp.besselj(0, x),
        domain="physics",
        description="Bessel function of the first kind, order 0.",
        citation="https://en.wikipedia.org/wiki/Bessel_function",
    ),
    Formula(
        name="Airy Ai",
        expression_factory=lambda: sp.airyai(x),
        domain="physics",
        description="Airy function of the first kind.",
        citation="https://en.wikipedia.org/wiki/Airy_function",
    ),
    Formula(
        name="Lambert W",
        expression_factory=lambda: sp.LambertW(x),
        domain="combinatorics",
        description="Inverse of x*e^x.",
        citation="https://en.wikipedia.org/wiki/Lambert_W_function",
    ),
]


# Convenience accessors
def by_name(name: str) -> Formula | None:
    """Return the registry entry whose name matches (case-insensitive),
    or None."""
    needle = name.strip().lower()
    for f in FORMULAS:
        if f.name.lower() == needle:
            return f
    return None


def list_all() -> list[Formula]:
    """All registered formulas."""
    return list(FORMULAS)


def list_by_domain(domain: str) -> list[Formula]:
    return [f for f in FORMULAS if f.domain == domain]
