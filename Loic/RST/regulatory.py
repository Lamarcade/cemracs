"""Basel IRB capital charge and the combined erosion function ``Psi``.

Pure, stateless, vectorised functions. This layer knows nothing about portfolios or
scenarios: it maps default probabilities to capital quantities, nothing else.

The model (specification section 2.2), for corporate exposures at the F-IRB default
maturity ``M = 2.5``::

    w(p)  = (1 - exp(-50 p)) / (1 - exp(-50))
    R(p)  = 0.12 w(p) + 0.24 (1 - w(p))          asset correlation, decreasing
    b(p)  = (0.11852 - 0.05478 ln p)^2
    MA(p) = 1 / (1 - 1.5 b(p))                   maturity adjustment
    K(p)  = ell [Phi((Phi^-1(p) + sqrt(R) Phi^-1(0.999)) / sqrt(1 - R)) - p] MA(p)

The ``- p`` inside the bracket removes the expected loss, so ``K`` charges capital
for *unexpected* loss only. That is what guarantees no double counting with the
provision channel: EL goes to the numerator via IFRS 9 provisions, UL goes to the
denominator via RWA, and the two partition the same 99.9% conditional loss.

The erosion function combines both channels into one increasing function per bucket::

    Psi(p) = ell * p  +  12.5 * R_star * K(p)

Monotonicity (specification section 4). ``K`` is **not** monotone: it peaks at
``p = 0.2962`` and decreases afterwards. ``Psi`` nevertheless is, on the whole
admissible domain ``[3e-4, 0.7202]``, because the provision slope ``ell`` absorbs the
negative slope of the capital charge -- the sufficient condition is
``K'(p) > -ell / (12.5 R_star) = -0.3048``. This is what makes the bucket-by-bucket
inversion of section 8 legitimate.

Conditioning warning: ``Psi'`` varies by a factor ~190 between ``p = 3 bp``
(``Psi' = 27.5``) and ``p = 20%`` (``Psi' = 0.63``). Harmless at Level 1, which only
evaluates and inverts; critical for the linearisation of Levels 2 and 3.

Assumptions carried by this module (specification section 5.1): 3 (fixed LGD),
4 (one PD serves both the point-in-time provision and the through-the-cycle IRB
charge), 6 (IRB portfolio, output floor and leverage ratio non-binding),
7 (infinitely granular ASRF portfolio, no concentration add-on).

Units: ``p`` is a fraction, ``r_star`` is a fraction, ``K`` and ``Psi`` are per unit
of exposure and therefore dimensionless.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from config import MA_POLE, PSI_MONOTONE_MAX, RstConfig

#: 99.9% confidence level of the ASRF model, ``Phi^-1(0.999)``.
Q999 = norm.ppf(0.999)

_B_INTERCEPT = 0.11852
_B_SLOPE = -0.05478
_CORR_LO = 0.12
_CORR_HI = 0.24
_CORR_DECAY = 50.0


def _as_array(p: ArrayLike) -> NDArray[np.float64]:
    """Coerce to a float array without copying when already one."""
    return np.asarray(p, dtype=float)


def check_domain(p: ArrayLike, bounds: tuple[float, float]) -> None:
    """Raise if any PD falls outside the admissible domain.

    Parameters
    ----------
    p : array_like
        Default probabilities, as fractions.
    bounds : tuple of float
        ``(p_min, p_max)``, typically ``RstConfig.pd_bounds``.

    Raises
    ------
    ValueError
        If any value lies outside ``bounds``, or if ``p_min`` sits below the pole of
        the maturity adjustment.

    Notes
    -----
    Callers are expected to clip *before* reaching this layer, and to report the
    clipping explicitly rather than silently -- see :func:`scenarios.clip_pd`.
    """
    lo, hi = bounds
    if lo < MA_POLE:
        raise ValueError(
            f"lower bound {lo:.3e} is below the maturity-adjustment pole "
            f"{MA_POLE:.3e}: K would change sign."
        )
    arr = _as_array(p)
    outside = (arr < lo) | (arr > hi)
    if np.any(outside):
        n = int(np.count_nonzero(outside))
        raise ValueError(
            f"{n} PD value(s) outside the admissible domain [{lo:.3e}, {hi:.3f}]; "
            f"observed range [{np.nanmin(arr):.3e}, {np.nanmax(arr):.3e}]. "
            "Clip with scenarios.clip_pd first."
        )


# -- asset correlation ---------------------------------------------------------


def asset_correlation(p: ArrayLike) -> NDArray[np.float64]:
    """Basel asset correlation ``R(p)``, decreasing from 0.24 towards 0.12."""
    arr = _as_array(p)
    w = (1.0 - np.exp(-_CORR_DECAY * arr)) / (1.0 - np.exp(-_CORR_DECAY))
    return _CORR_LO * w + _CORR_HI * (1.0 - w)


def d_asset_correlation(p: ArrayLike) -> NDArray[np.float64]:
    """Derivative ``R'(p)``, negative everywhere."""
    arr = _as_array(p)
    dw = _CORR_DECAY * np.exp(-_CORR_DECAY * arr) / (1.0 - np.exp(-_CORR_DECAY))
    return (_CORR_LO - _CORR_HI) * dw


# -- maturity adjustment -------------------------------------------------------


def _b(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Maturity smoothing term ``b(p)``."""
    return (_B_INTERCEPT + _B_SLOPE * np.log(p)) ** 2


def _db(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative ``b'(p)``."""
    return 2.0 * (_B_INTERCEPT + _B_SLOPE * np.log(p)) * (_B_SLOPE / p)


def maturity_adjustment(p: ArrayLike) -> NDArray[np.float64]:
    """Maturity adjustment ``MA(p)`` at the F-IRB default maturity ``M = 2.5``.

    Singular at ``p = MA_POLE = 2.927e-06`` and negative below it. The regulatory PD
    floor of 3 basis points sits 102x above that pole, so the admissible domain never
    comes near it.
    """
    arr = _as_array(p)
    return 1.0 / (1.0 - 1.5 * _b(arr))


def d_maturity_adjustment(p: ArrayLike) -> NDArray[np.float64]:
    """Derivative ``MA'(p)``."""
    arr = _as_array(p)
    return 1.5 * _db(arr) / (1.0 - 1.5 * _b(arr)) ** 2


# -- conditional loss rate -----------------------------------------------------


def conditional_loss_rate(p: ArrayLike) -> NDArray[np.float64]:
    """99.9% conditional default rate of the ASRF model, LGD excluded."""
    arr = _as_array(p)
    r = asset_correlation(arr)
    return norm.cdf((norm.ppf(arr) + np.sqrt(r) * Q999) / np.sqrt(1.0 - r))


def d_conditional_loss_rate(p: ArrayLike) -> NDArray[np.float64]:
    """Derivative of :func:`conditional_loss_rate`, including the ``R'(p)`` term."""
    arr = _as_array(p)
    r = asset_correlation(arr)
    rp = d_asset_correlation(arr)
    x = norm.ppf(arr)
    s = np.sqrt(1.0 - r)
    z = (x + np.sqrt(r) * Q999) / s
    dx = 1.0 / norm.pdf(x)
    numerator = dx + 0.5 * rp / np.sqrt(r) * Q999
    dz = numerator / s + (x + np.sqrt(r) * Q999) * (0.5 * rp / s**3)
    return norm.pdf(z) * dz


# -- capital charge ------------------------------------------------------------


def capital_charge(p: ArrayLike, ell: float) -> NDArray[np.float64]:
    """IRB capital charge ``K(p)`` per unit of exposure, unexpected loss only.

    Parameters
    ----------
    p : array_like
        Default probabilities, as fractions, inside the admissible domain.
    ell : float
        Loss given default.

    Returns
    -------
    ndarray
        Capital charge per unit of exposure. Peaks at ``K(0.2962) = 0.1769`` and
        decreases afterwards -- do not assume monotonicity.
    """
    arr = _as_array(p)
    return ell * (conditional_loss_rate(arr) - arr) * maturity_adjustment(arr)


def d_capital_charge(p: ArrayLike, ell: float) -> NDArray[np.float64]:
    """Analytic derivative ``K'(p)``.

    Exact, not a finite difference: the product rule is applied to both factors, and
    both the ``R'(p)`` and ``MA'(p)`` terms are carried. Verified against central
    differences to ~1e-10 on ``[3e-4, 0.5]`` (run this module as a script).

    Negative beyond ``p = K_ARGMAX = 0.2962``. That is expected and does not break
    the invertibility of :func:`psi`.
    """
    arr = _as_array(p)
    ma = maturity_adjustment(arr)
    d_ma = d_maturity_adjustment(arr)
    return (
        ell * (d_conditional_loss_rate(arr) - 1.0) * ma
        + ell * (conditional_loss_rate(arr) - arr) * d_ma
    )


# -- erosion function ----------------------------------------------------------


def psi(p: ArrayLike, ell: float, r_star: float, kappa_tax: float = 0.0) -> NDArray[np.float64]:
    """Per-bucket erosion function, both channels combined.

    ``Psi(p) = (1 - kappa_tax) * ell * p + 12.5 * r_star * K(p)``, where the first
    term is the IFRS 9 Stage 1 provision hitting the numerator and the second is the
    RWA charge hitting the denominator, already scaled by the breach level.

    The tax shield scales the *provision* channel only. The ``ell`` inside ``K`` is
    the LGD of the capital charge and is untouched by tax -- keeping the two uses of
    ``ell`` distinct is what makes the affine breach form agree with the ratio
    computed directly (:func:`breach.cet1_ratio`) once ``kappa_tax > 0``.

    Strictly increasing on ``[3e-4, 0.7202]``, hence invertible -- see :func:`psi_inv`.
    """
    arr = _as_array(p)
    return (1.0 - kappa_tax) * ell * arr + 12.5 * r_star * capital_charge(arr, ell)


def d_psi(p: ArrayLike, ell: float, r_star: float, kappa_tax: float = 0.0) -> NDArray[np.float64]:
    """Analytic derivative ``Psi'(p) = (1 - kappa_tax) * ell + 12.5 * r_star * K'(p)``.

    Not used by Level 1, which evaluates ``Psi`` by direct substitution. Exposed here
    because Level 2 linearises the erosion around the projection, and the sensitivity
    vector ``(d_n)_g = E[g,n] * Psi'(pbar[s,g,n])`` is exactly this quantity times
    exposure. Level 3 reuses the same vector in the Mahalanobis programme.
    """
    arr = _as_array(p)
    return (1.0 - kappa_tax) * ell + 12.5 * r_star * d_capital_charge(arr, ell)


def psi_inv(
    target: ArrayLike,
    ell: float,
    r_star: float,
    bounds: tuple[float, float],
    kappa_tax: float = 0.0,
) -> NDArray[np.float64]:
    """Invert ``Psi`` on ``bounds``, elementwise.

    Parameters
    ----------
    target : array_like
        Target values of ``Psi``. May contain ``NaN`` or infinities.
    ell, r_star : float
        Erosion function parameters.
    bounds : tuple of float
        ``(p_min, p_max)`` bracketing interval. ``Psi`` must be increasing on it,
        which :class:`config.SensitivityParams` already enforces.

    Returns
    -------
    ndarray
        Preimages, with ``NaN`` wherever the target lies outside
        ``[Psi(p_min), Psi(p_max)]``.

    Notes
    -----
    Returns ``NaN`` instead of raising when the target is out of reach. That is a
    deliberate contract: an unreachable target is an economic result (the bucket
    cannot break the ratio on its own even at maximum PD), not a numerical failure,
    and :func:`breach.critical_pd` is the layer that labels it as such.
    """
    lo, hi = bounds
    psi_lo = float(psi(lo, ell, r_star, kappa_tax))
    psi_hi = float(psi(hi, ell, r_star, kappa_tax))

    arr = _as_array(target)
    out = np.full(arr.shape, np.nan, dtype=float)
    reachable = np.isfinite(arr) & (arr >= psi_lo) & (arr <= psi_hi)

    # brentq is scalar-only; loop over the reachable entries rather than vectorising
    # a bisection by hand, since accuracy matters more than speed here. Cost is one
    # root-find per cell, so inverting a full (scenario, bucket, date) cube at
    # (sector x region) granularity is seconds, not milliseconds.
    for index in zip(*np.nonzero(reachable)):
        value = float(arr[index])
        out[index] = brentq(
            lambda p, v=value: float(psi(p, ell, r_star, kappa_tax)) - v,
            lo,
            hi,
            xtol=1e-14,
        )
    return out


# -- config-bound convenience --------------------------------------------------


def psi_from_config(p: ArrayLike, cfg: RstConfig) -> NDArray[np.float64]:
    """:func:`psi` with all parameters taken from ``cfg``."""
    return psi(p, cfg.ell, cfg.r_star, cfg.sensitivity.kappa_tax)


def d_psi_from_config(p: ArrayLike, cfg: RstConfig) -> NDArray[np.float64]:
    """:func:`d_psi` with all parameters taken from ``cfg``."""
    return d_psi(p, cfg.ell, cfg.r_star, cfg.sensitivity.kappa_tax)


def psi_inv_from_config(target: ArrayLike, cfg: RstConfig) -> NDArray[np.float64]:
    """:func:`psi_inv` with parameters and bounds taken from ``cfg``."""
    return psi_inv(
        target, cfg.ell, cfg.r_star, cfg.pd_bounds, cfg.sensitivity.kappa_tax
    )


if __name__ == "__main__":
    # Self-check, not a test suite (tests are out of scope for this iteration).
    # Reproduces the reference table of the specification appendix B and validates
    # the analytic derivatives against central differences.
    ELL, RSTAR = 0.40, 0.105

    grid = np.array(
        [0.0003, 0.001, 0.003, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    )
    expected_k = np.array(
        [0.01027, 0.02109, 0.03867, 0.06565, 0.08167, 0.10656,
         0.13731, 0.15753, 0.16941, 0.17694, 0.17054, 0.15493]
    )
    expected_psi = np.array(
        [0.01360, 0.02808, 0.05195, 0.09016, 0.11520, 0.15986,
         0.22021, 0.26676, 0.30235, 0.35223, 0.38383, 0.40334]
    )

    k = capital_charge(grid, ELL)
    ps = psi(grid, ELL, RSTAR)
    dk = d_capital_charge(grid, ELL)
    dps = d_psi(grid, ELL, RSTAR)

    print(f"{'p':>8} {'K':>10} {'Psi':>10} {'dK':>10} {'dPsi':>10}")
    for row in zip(grid, k, ps, dk, dps):
        print("{:8.4f} {:10.5f} {:10.5f} {:10.3f} {:10.3f}".format(*row))

    print(f"\nmax |K - appendix B|   = {np.abs(k - expected_k).max():.2e}")
    print(f"max |Psi - appendix B| = {np.abs(ps - expected_psi).max():.2e}")

    # analytic vs central differences, on a *relative* step: K' spans four orders of
    # magnitude over the domain, so a fixed absolute step is dominated by truncation
    # error at the low-PD end and says nothing about the derivative there.
    fine = np.geomspace(3e-4, 0.5, 200)
    h = 1e-6 * fine
    dk_fd = (capital_charge(fine + h, ELL) - capital_charge(fine - h, ELL)) / (2 * h)
    dps_fd = (psi(fine + h, ELL, RSTAR) - psi(fine - h, ELL, RSTAR)) / (2 * h)
    dk_err = np.abs(d_capital_charge(fine, ELL) - dk_fd) / np.abs(dk_fd)
    dps_err = np.abs(d_psi(fine, ELL, RSTAR) - dps_fd) / np.abs(dps_fd)
    print(f"max relative |dK - central diff|   = {dk_err.max():.2e}")
    print(f"max relative |dPsi - central diff| = {dps_err.max():.2e}")

    # monotonicity of Psi over the admissible domain, and non-monotonicity of K
    domain = np.geomspace(3e-4, 0.30, 2000)
    slope = d_psi(domain, ELL, RSTAR)
    print(f"\nmin Psi' on [3e-4, 0.30]  = {slope.min():.3f} at p = {domain[slope.argmin()]:.4f}")
    print(f"K' at p = 0.30            = {float(d_capital_charge(0.30, ELL)):+.4f}  (K is decreasing there)")
    print(f"sufficient condition K' > {-ELL / (12.5 * RSTAR):.4f}")

    # inversion round-trip
    probe = np.array([1e-3, 0.01, 0.05, 0.2, 0.29])
    back = psi_inv(psi(probe, ELL, RSTAR), ELL, RSTAR, (3e-4, 0.30))
    print(f"max |psi_inv(psi(p)) - p| = {np.abs(back - probe).max():.2e}")
    unreachable = psi_inv(np.array([psi(0.30, ELL, RSTAR) * 2]), ELL, RSTAR, (3e-4, 0.30))
    print(f"psi_inv beyond Psi(p_max) -> {unreachable[0]} (NaN expected, no exception)")
