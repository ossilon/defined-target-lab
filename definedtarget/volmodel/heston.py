"""The Heston (1993) stochastic-volatility model: parameters, characteristic function and moments.

Plain words
- The index's instantaneous variance v moves at random. It is pulled toward a long-run level theta at speed kappa,
  shaken by a "vol of vol" eta in proportion to sqrt(v), and its shocks are correlated (rho) with the index's:
      dS/S = sqrt(v) dW1  (in forward terms),   dv = kappa (theta - v) dt + eta sqrt(v) dW2,   dW1 dW2 = rho dt
- Everything here is under the pricing (risk-neutral) measure and in forward terms: X_T = ln(S_T / F_T), where F_T
  is the forward for expiry T. The forward is a martingale; rates and dividends enter only through F and the
  discount factor, which the pricers take as inputs.
- The characteristic function phi(u) = E[exp(i u X_T)] is known in closed form. The option pricers (pricers.py)
  and the variance identities (varswap.py) are built on it.

The form of phi. Two algebraically equal ways of writing it exist. The original one (Heston 1993) uses e^{+dT}; with
the principal complex logarithm it jumps whenever the logarithm's argument crosses the negative real axis, which
happens at long maturities (Gatheral 2006, p. 20; Albrecher et al. 2007, "the little Heston trap"). The form used
here uses e^{-dT} (Gatheral 2006, eq. 2.12; Fang and Oosterlee 2008, section 3.3), for which Lord and Kahl (2008,
working paper, as cited by Fang and Oosterlee) proved that the principal logarithm gives the right function. tests/test_vol_heston.py shows both
behaviors for Andersen's (2008) Case I at T = 10.

Notation in the sources: Gatheral (2006) v, vbar, lambda, eta, rho; Andersen (2008) V(0), theta, kappa, epsilon,
rho; Fang and Oosterlee (2008) u0, ubar, lambda, eta, rho. Here: v0, theta, kappa, eta, rho. Time is in years.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Heston:
    """Heston parameters: v0 (variance now), theta (long-run variance), kappa (mean-reversion speed per year),
    eta (vol of variance), rho (correlation of index and variance shocks). Variances are in units of vol^2 per year
    (a 20% vol is 0.04).

    The Feller condition 2 kappa theta >= eta^2 keeps v away from zero; it is reported (`feller`), not required,
    because fits to index options usually break it (Gatheral 2006, ch. 2; his Table 3.2 fit gives 0.62).
    """
    v0: float
    theta: float
    kappa: float
    eta: float
    rho: float

    def __post_init__(self):
        for name in ("v0", "theta", "kappa", "eta", "rho"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Heston {name} must be a finite number, got {value!r}")
        if self.v0 < 0:
            raise ValueError(f"Heston v0 (variance now) must be 0 or more, got {self.v0}")
        for name in ("theta", "kappa"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Heston {name} must be positive, got {getattr(self, name)}")
        if self.eta <= 0:
            raise ValueError(f"Heston eta (vol of variance) must be positive, got {self.eta}; for eta = 0 the variance "
                             "path is deterministic: price with Black-76 at sqrt(expected_variance(T) / T)")
        if not -1 < self.rho < 1:
            raise ValueError(f"Heston rho must lie strictly between -1 and 1, got {self.rho}")

    @property
    def feller(self):
        """2 kappa theta / eta^2: at 1 or more the variance can never reach zero (Andersen 2008, Proposition 2)."""
        return 2 * self.kappa * self.theta / self.eta ** 2

    def mean_variance(self, t):
        """E[v_t] = theta + (v0 - theta) e^{-kappa t} (Andersen 2008, Corollary 1)."""
        return self.theta + (self.v0 - self.theta) * math.exp(-self.kappa * t)

    def expected_variance(self, T):
        """E[W_T], the expected total variance from 0 to T: theta T + (v0 - theta)(1 - e^{-kappa T}) / kappa
        (Gatheral 2006, ch. 11). It does not depend on eta or rho."""
        return self.theta * T + (self.v0 - self.theta) * -math.expm1(-self.kappa * T) / self.kappa

    def variance_swap_rate(self, T):
        """The fair variance-swap rate for maturity T (expected variance per year): E[W_T] / T."""
        return self.expected_variance(T) / T

    def cumulants(self, T):
        """(c1, c2): mean and variance of X_T = ln(S_T / F_T).

        c1 = -E[W_T] / 2 (Fang and Oosterlee 2008, Table 11, with zero drift; Andersen 2008, Table A.1).
        c2 = Var(X_T) from Andersen (2008, Table A.1). Fang and Oosterlee's Table 11 prints the same quantity with the
        term theta (6 e^{-kappa T} - 7) where theta (4 e^{-kappa T} - 5) is right: tests/test_vol_heston.py shows the
        printed form disagrees with the numerical derivatives of the characteristic function and can turn negative
        (their own remark), while the corrected form equals Andersen's to rounding (found 2026-10-21). Their paper
        uses c2 only to size the integration range, so its prices are unaffected."""
        k, th, e, rho, v0 = self.kappa, self.theta, self.eta, self.rho, self.v0
        x, x2 = math.exp(-k * T), math.exp(-2 * k * T)
        c1 = -self.expected_variance(T) / 2
        o1 = (x2 * e * e + 4 * x * ((1 + k * T) * e * e - 2 * rho * k * e * (2 + k * T) + 2 * k * k)
              + (2 * k * T - 5) * e * e - 8 * rho * k * e * (k * T - 2) + 8 * k * k * (k * T - 1))
        o2 = (-x2 * e * e + 2 * x * (-k * T * e * e + 2 * rho * e * k * (1 + k * T) - 2 * k * k)
              + e * e - 4 * k * rho * e + 4 * k * k)
        c2 = th / (8 * k ** 3) * o1 + v0 / (4 * k ** 3) * o2
        if k < 0.01 or k * T < 1e-3:
            # the closed form subtracts terms of order kappa^2 to get a result of order kappa^3 T: it loses digits as
            # kappa (or kappa T) shrinks (code review, 2026-10-23: relative error 2e-8 at kappa = 0.001, a negative
            # variance at kappa = 1e-6). There the variance comes from log phi's second derivative instead.
            c2 = _variance_numeric(self, T)
        return c1, c2


@dataclass(frozen=True)
class DoubleHeston:
    """The two-factor Heston model (Christoffersen, Heston and Jacobs 2009): two independent variance factors, each a
    Heston factor with its own speed, long-run level, vol of variance and correlation with the index:
        dS/S = sqrt(v1) dW1 + sqrt(v2) dW2,   dv_i = kappa_i (theta_i - v_i) dt + eta_i sqrt(v_i) dZ_i,
        dW_i dZ_i = rho_i dt, and every other pair of shocks independent.
    The variance now is v1 + v2. The log price X_T = ln(S_T / F_T) is the sum of two independent parts, and each
    part is a one-factor Heston log price with that factor's parameters. So log phi = log phi_1 + log phi_2, and the
    mean and variance of X_T add (derived here from the independence; tests/test_vol_twofactor.py checks it, also
    through the exact reduction: with equal kappa, eta and rho the model is one-factor Heston with v0 = v1 + v2 and
    theta = theta1 + theta2, because the sum of the two square-root processes is then one square-root process)."""
    f1: Heston
    f2: Heston

    def __post_init__(self):
        for name in ("f1", "f2"):
            if not isinstance(getattr(self, name), Heston):
                raise ValueError(f"DoubleHeston {name} must be a Heston factor, got {getattr(self, name)!r}")

    @property
    def v0(self):
        return self.f1.v0 + self.f2.v0

    @property
    def theta(self):
        """The long-run total variance theta1 + theta2."""
        return self.f1.theta + self.f2.theta

    def mean_variance(self, t):
        return self.f1.mean_variance(t) + self.f2.mean_variance(t)

    def expected_variance(self, T):
        return self.f1.expected_variance(T) + self.f2.expected_variance(T)

    def variance_swap_rate(self, T):
        return self.expected_variance(T) / T

    def cumulants(self, T):
        (a1, b1), (a2, b2) = self.f1.cumulants(T), self.f2.cumulants(T)
        return a1 + a2, b1 + b2


def _variance_numeric(model, T, scale=1e-3):
    """Var(X_T) = -d^2/du^2 log phi at 0: central differences, Richardson-extrapolated over h and 2h, with the step
    h = scale / sqrt(E[W_T]) in units of the distribution's width (tests/test_vol_heston.py: within 1e-8)."""
    h = scale / math.sqrt(model.expected_variance(T))

    def second(s):
        lp, lm = log_cf(model, s, T), log_cf(model, -s, T)
        return -((lp + lm) / (s * s)).real

    return (4.0 * second(h) - second(2.0 * h)) / 3.0


def cf_parts(model, u, T):
    """(C, D) with log E[exp(i u X_T)] = C + D v0. Neither part depends on v0, so one evaluation prices any number of
    starting variances (pricetable.py uses this). One-factor models only."""
    if not isinstance(model, Heston):
        raise TypeError("cf_parts takes a one-factor Heston model (a DoubleHeston has one D per factor)")
    if not T > 0:
        raise ValueError(f"maturity T must be positive (years), got {T}")
    u = np.asarray(u, dtype=complex)
    k, th, eta, rho = model.kappa, model.theta, model.eta, model.rho
    beta = k - 1j * rho * eta * u
    d = np.sqrt(beta * beta + eta * eta * (u * u + 1j * u))      # principal square root: real part >= 0
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    D = (beta - d) / (eta * eta) * (1 - e) / (1 - g * e)
    C = k * th / (eta * eta) * ((beta - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return C, D


def log_cf(model, u, T):
    """log E[exp(i u X_T)] for real or complex u (array or scalar), in the continuous form described above. For a
    DoubleHeston, the sum over its two factors."""
    if isinstance(model, DoubleHeston):
        return log_cf(model.f1, u, T) + log_cf(model.f2, u, T)
    C, D = cf_parts(model, u, T)
    return C + D * model.v0


def cf(model, u, T):
    """The characteristic function E[exp(i u X_T)] of X_T = ln(S_T / F_T) under the pricing measure."""
    return np.exp(log_cf(model, u, T))
