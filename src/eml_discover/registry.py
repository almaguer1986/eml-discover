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
# Physics constants + variables
m = sp.Symbol("m", real=True, positive=True)
v = sp.Symbol("v", real=True)
c = sp.Symbol("c", real=True, positive=True)
G = sp.Symbol("G", real=True, positive=True)
M = sp.Symbol("M", real=True, positive=True)
q1 = sp.Symbol("q1", real=True)
q2 = sp.Symbol("q2", real=True)
h = sp.Symbol("h", real=True, positive=True)
k_const = sp.Symbol("k", real=True, positive=True)   # rate / spring / Boltzmann
sigma_sb = sp.Symbol("sigma_sb", real=True, positive=True)  # Stefan-Boltzmann
# Probabilities
p = sp.Symbol("p", real=True, positive=True)
q = sp.Symbol("q", real=True, positive=True)
# Finance
P = sp.Symbol("P", real=True, positive=True)
F = sp.Symbol("F", real=True, positive=True)
C = sp.Symbol("C", real=True, positive=True)
R = sp.Symbol("R", real=True, positive=True)   # rate (alias of r where ambiguous)
# Generic positive scale
lam = sp.Symbol("lam", real=True, positive=True)   # rate / wavelength
alpha = sp.Symbol("alpha", real=True)
beta = sp.Symbol("beta", real=True)


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

    # --- Statistics: more distributions ---
    Formula(
        name="Cauchy PDF (standardized)",
        expression_factory=lambda: 1 / (sp.pi * (1 + x**2)),
        domain="statistics",
        description="Standard Cauchy / Lorentz density.",
        citation="https://en.wikipedia.org/wiki/Cauchy_distribution",
    ),
    Formula(
        name="exponential PDF",
        expression_factory=lambda: lam * sp.exp(-lam * x),
        domain="statistics",
        description="Exponential density with rate lam, x >= 0.",
        citation="https://en.wikipedia.org/wiki/Exponential_distribution",
    ),
    Formula(
        name="lognormal PDF",
        expression_factory=lambda:
            sp.exp(-(sp.log(x) - mu)**2 / (2 * sigma**2))
            / (x * sigma * sp.sqrt(2 * sp.pi)),
        domain="statistics",
        description="Log-normal density: log of variable is normal.",
        citation="https://en.wikipedia.org/wiki/Log-normal_distribution",
    ),
    Formula(
        name="chi-square PDF (k=2)",
        expression_factory=lambda: sp.exp(-x / 2) / 2,
        domain="statistics",
        description="Chi-square with 2 dof; equivalent to Exp(1/2).",
        citation="https://en.wikipedia.org/wiki/Chi-squared_distribution",
    ),

    # --- ML: more activations / derivatives ---
    Formula(
        name="Mish",
        expression_factory=lambda: x * sp.tanh(sp.log(1 + sp.exp(x))),
        domain="ml",
        description="Mish activation: x * tanh(softplus(x)).",
        citation="Misra, 2019",
    ),
    Formula(
        name="ELU (alpha=1)",
        expression_factory=lambda:
            sp.Piecewise((x, x >= 0), (sp.exp(x) - 1, True)),
        domain="ml",
        description="Exponential Linear Unit with alpha=1.",
        citation="Clevert et al., 2015",
    ),
    Formula(
        name="sigmoid derivative",
        expression_factory=lambda:
            sp.exp(-x) / (1 + sp.exp(-x))**2,
        domain="ml",
        description="d/dx sigmoid(x); equals sigmoid(x)*(1-sigmoid(x)).",
        citation="https://en.wikipedia.org/wiki/Sigmoid_function",
    ),

    # --- Trig identities ---
    Formula(
        name="double-angle sin",
        expression_factory=lambda: 2 * sp.sin(x) * sp.cos(x),
        domain="trigonometry",
        description="2 sin(x) cos(x) = sin(2x).",
        citation="https://en.wikipedia.org/wiki/List_of_trigonometric_identities",
    ),
    Formula(
        name="double-angle cos",
        expression_factory=lambda: sp.cos(x)**2 - sp.sin(x)**2,
        domain="trigonometry",
        description="cos^2(x) - sin^2(x) = cos(2x).",
        citation="https://en.wikipedia.org/wiki/List_of_trigonometric_identities",
    ),
    Formula(
        name="tangent half-angle",
        expression_factory=lambda: (1 - sp.cos(x)) / sp.sin(x),
        domain="trigonometry",
        description="(1 - cos x) / sin x = tan(x/2).",
        citation="https://en.wikipedia.org/wiki/Tangent_half-angle_formula",
    ),

    # --- Hyperbolic identities ---
    Formula(
        name="double-angle sinh",
        expression_factory=lambda: 2 * sp.sinh(x) * sp.cosh(x),
        domain="hyperbolic",
        description="2 sinh(x) cosh(x) = sinh(2x).",
        citation="https://en.wikipedia.org/wiki/Hyperbolic_functions",
    ),
    Formula(
        name="tanh half-angle",
        expression_factory=lambda: sp.sinh(x) / (1 + sp.cosh(x)),
        domain="hyperbolic",
        description="sinh(x) / (1 + cosh(x)) = tanh(x/2).",
        citation="https://en.wikipedia.org/wiki/Hyperbolic_functions",
    ),

    # --- Physics: classical ---
    Formula(
        name="Stefan-Boltzmann law",
        expression_factory=lambda: sigma_sb * T**4,
        domain="physics",
        description="Total emitted power per unit area: sigma_sb * T^4.",
        citation="https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law",
    ),
    Formula(
        name="kinetic energy",
        expression_factory=lambda: m * v**2 / 2,
        domain="physics",
        description="Newtonian KE = (1/2) m v^2.",
        citation="https://en.wikipedia.org/wiki/Kinetic_energy",
    ),
    Formula(
        name="Coulomb's law",
        expression_factory=lambda: k_const * q1 * q2 / r**2,
        domain="physics",
        description="Electrostatic force between two point charges.",
        citation="https://en.wikipedia.org/wiki/Coulomb%27s_law",
    ),
    Formula(
        name="Newton's gravitation",
        expression_factory=lambda: G * M * m / r**2,
        domain="physics",
        description="Gravitational force between two masses.",
        citation="https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation",
    ),
    Formula(
        name="Lorentz factor",
        expression_factory=lambda: 1 / sp.sqrt(1 - (v / c)**2),
        domain="physics",
        description="Special-relativity time-dilation factor gamma.",
        citation="https://en.wikipedia.org/wiki/Lorentz_factor",
    ),
    Formula(
        name="de Broglie wavelength",
        expression_factory=lambda: h / (m * v),
        domain="physics",
        description="Wavelength of a particle with momentum m v.",
        citation="https://en.wikipedia.org/wiki/Matter_wave",
    ),
    Formula(
        name="Schwarzschild radius",
        expression_factory=lambda: 2 * G * M / c**2,
        domain="physics",
        description="Event-horizon radius of a non-rotating black hole.",
        citation="https://en.wikipedia.org/wiki/Schwarzschild_radius",
    ),
    Formula(
        name="heat kernel (1D)",
        expression_factory=lambda:
            sp.exp(-x**2 / (4 * k_const * t)) / sp.sqrt(4 * sp.pi * k_const * t),
        domain="physics",
        description="Free-space Green's function for the diffusion equation.",
        citation="https://en.wikipedia.org/wiki/Heat_kernel",
    ),
    Formula(
        name="Planck radiance kernel",
        expression_factory=lambda: x**3 / (sp.exp(x) - 1),
        domain="physics",
        description="Black-body radiance in dimensionless form.",
        citation="https://en.wikipedia.org/wiki/Planck%27s_law",
    ),
    Formula(
        name="Maxwell-Boltzmann speed PDF (kernel)",
        expression_factory=lambda: x**2 * sp.exp(-x**2 / T),
        domain="physics",
        description="Speed distribution kernel; full PDF includes a 4*pi/(sqrt(pi))^3 prefactor.",
        citation="https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution",
    ),

    # --- Finance ---
    Formula(
        name="continuous compound interest",
        expression_factory=lambda: P * sp.exp(r * t),
        domain="finance",
        description="Continuous compounding from principal P at rate r over time t.",
        citation="https://en.wikipedia.org/wiki/Compound_interest",
    ),
    Formula(
        name="present value (discrete)",
        expression_factory=lambda: F / (1 + r)**t,
        domain="finance",
        description="Discrete-period present value of a future amount F.",
        citation="https://en.wikipedia.org/wiki/Present_value",
    ),
    Formula(
        name="perpetuity",
        expression_factory=lambda: C / r,
        domain="finance",
        description="Present value of an infinite cash-flow stream C at rate r.",
        citation="https://en.wikipedia.org/wiki/Perpetuity",
    ),
    Formula(
        name="annuity factor",
        expression_factory=lambda: (1 - (1 + r)**(-n)) / r,
        domain="finance",
        description="Present-value factor for an n-period annuity at rate r.",
        citation="https://en.wikipedia.org/wiki/Annuity",
    ),

    # --- Information theory ---
    Formula(
        name="Shannon entropy (single term)",
        expression_factory=lambda: -p * sp.log(p),
        domain="info-theory",
        description="Single-term contribution to Shannon entropy.",
        citation="https://en.wikipedia.org/wiki/Entropy_(information_theory)",
    ),
    Formula(
        name="KL divergence (single term)",
        expression_factory=lambda: p * sp.log(p / q),
        domain="info-theory",
        description="Single-term contribution to KL divergence D(p||q).",
        citation="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence",
    ),
    Formula(
        name="cross-entropy (single term)",
        expression_factory=lambda: -p * sp.log(q),
        domain="info-theory",
        description="Single-term contribution to H(p, q).",
        citation="https://en.wikipedia.org/wiki/Cross-entropy",
    ),

    # --- Calculus / special functions ---
    Formula(
        name="tanh derivative",
        expression_factory=lambda: 1 - sp.tanh(x)**2,
        domain="calculus",
        description="d/dx tanh(x); equals sech^2(x).",
        citation="https://en.wikipedia.org/wiki/Hyperbolic_functions",
    ),
    Formula(
        name="erf",
        expression_factory=lambda: sp.erf(x),
        domain="special",
        description="Gauss error function (Pfaffian-but-not-EML).",
        citation="https://en.wikipedia.org/wiki/Error_function",
    ),
    Formula(
        name="sinc (unnormalized)",
        expression_factory=lambda: sp.sin(x) / x,
        domain="signal-processing",
        description="Cardinal sine function.",
        citation="https://en.wikipedia.org/wiki/Sinc_function",
    ),

    # --- More physics + variants ---
    Formula(
        name="power-law decay",
        expression_factory=lambda: x**(-alpha),
        domain="physics",
        description="x^(-alpha); fundamental scaling law.",
        citation="https://en.wikipedia.org/wiki/Power_law",
    ),
    Formula(
        name="logistic growth",
        expression_factory=lambda: K / (1 + sp.exp(-r * t)),
        domain="biology",
        description="Logistic / saturating growth toward carrying capacity K.",
        citation="https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation",
    ),
    Formula(
        name="RC voltage decay",
        expression_factory=lambda: sp.exp(-t / (R * C)),
        domain="electrical-engineering",
        description="Capacitor discharge voltage shape; tau = RC.",
        citation="https://en.wikipedia.org/wiki/RC_circuit",
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
