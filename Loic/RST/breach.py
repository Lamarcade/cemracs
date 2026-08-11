"""The breach inequality: cushion, erosion, distance to breach, critical PD.

This is the analytic core. The derivation below is exact -- no approximation, no
linearisation. Level 2 will linearise ``Psi`` around the projection; Level 1 does not.

Since ``RWA[n] > 0``::

    Ratio[n] < R_star   <=>   CET1[n] - R_star * RWA[n] < 0

which turns a nonlinear constraint on a ratio into an **affine** one. Expanding,
grouping bucket by bucket, and subtracting the same expression evaluated at the
reference narrative gives::

    Psi_g(p) = ell_g * p            provision channel (numerator)
             + 12.5 * R_star * K_g(p)   RWA channel (denominator)

    H[n]       = (CET1_RE[n] - R_star * RWA_oth) - sum_g E[g,n] * Psi_g(p0[g,n])
    Erosion[n] = sum_g E[g,n] * (Psi_g(p[g,n]) - Psi_g(p0[g,n]))

    breach  <=>  Erosion[n] > H[n]

Three properties worth keeping in mind:

1. ``RWA_oth`` and ``CET1_RE[n]`` cancel exactly between the two sides. They enter
   only through the level of ``H[n]``, never through the erosion function.
2. The choice of ``p0`` is a pure normalisation: changing it shifts ``H[n]`` and
   ``Erosion[n]`` by the same amount, so the breach set is invariant. That makes it
   the cheapest available consistency check -- see ``run_rst.py``.
3. The provision channel telescopes over past dates: only the PD *at the current
   date* enters, consistent with the IFRS 9 Stage 1 snapshot.

The inequality is **strict** (``>``, not ``>=``), to stay consistent with
``Ratio[n] < R_star``.

Assumption 8 (specification section 5.1) is the admissibility condition of the whole
exercise: ``H[n] > 0``. A bank already below ``R_star`` on its own reference narrative
makes the reverse stress test vacuous, so :func:`cushion` checks and reports it.

Vectorisation: the tidy input is pivoted to ``(n_scenario, n_bucket, n_date)`` arrays
upstream, and the erosion is a single broadcast -- there is no loop over scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import regulatory
from config import RstConfig
from portfolio import Portfolio, check_bucket_alignment
from scenarios import ScenarioSet


def _check_aligned(portfolio: Portfolio, scenarios: ScenarioSet) -> None:
    """Buckets and dates must agree between the balance sheet and the projections."""
    check_bucket_alignment(portfolio, scenarios.buckets)
    if not np.array_equal(portfolio.dates, scenarios.dates):
        raise ValueError(
            f"date grids differ: portfolio {portfolio.dates.tolist()} vs scenarios "
            f"{scenarios.dates.tolist()}"
        )


# -- ratio and cushion ---------------------------------------------------------


def cet1_ratio(
    portfolio: Portfolio, scenarios: ScenarioSet, cfg: RstConfig, scenario: str | None = None
) -> NDArray[np.float64]:
    """CET1 ratio under one scenario, evaluated directly rather than through the inequality.

    ``Ratio[n] = (CET1_RE[n] - sum_g ell_g E[g,n] p[g,n]) /
    (RWA_oth + 12.5 sum_g K_g(p[g,n]) E[g,n])``.

    Parameters
    ----------
    portfolio : Portfolio
    scenarios : ScenarioSet
    cfg : RstConfig
    scenario : str, optional
        Defaults to the reference narrative.

    Returns
    -------
    ndarray, shape (n_date,)
        The ratio as a fraction.

    Notes
    -----
    Not used by the breach analysis, which works on the equivalent affine form. Kept
    because ``R_0`` is needed to build the relative ``R_star`` convention
    (:meth:`config.RstConfig.r_star_conventions`), and because comparing this against
    the affine form is the most direct check that the algebra was transcribed right.
    """
    _check_aligned(portfolio, scenarios)
    index = (
        scenarios.baseline_index if scenario is None else scenarios.scenario_index(scenario)
    )
    pd_matrix = scenarios.pd_cube[index]

    provision = (cfg.ell * portfolio.exposure * pd_matrix).sum(axis=0)
    provision *= 1.0 - cfg.sensitivity.kappa_tax  # assumption 5: pre-tax by default
    cet1 = portfolio.cet1_re(cfg) - provision

    charge = regulatory.capital_charge(pd_matrix, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * (
        charge * portfolio.exposure
    ).sum(axis=0)
    return cet1 / rwa


def cushion(
    portfolio: Portfolio, scenarios: ScenarioSet, cfg: RstConfig, check: bool = True
) -> NDArray[np.float64]:
    """Reference capital cushion ``H[n]``, in euros.

    ``H[n] = (CET1_RE[n] - R_star * RWA_oth) - sum_g E[g,n] * Psi_g(p0[g,n])``.

    Parameters
    ----------
    portfolio : Portfolio
    scenarios : ScenarioSet
        Supplies ``p0[g,n]`` through its reference narrative.
    cfg : RstConfig
    check : bool, optional
        Enforce assumption 8, ``H[n] > 0``. Default True.

    Returns
    -------
    ndarray, shape (n_date,)
        Capital surplus above the breach level on the reference narrative.

    Raises
    ------
    ValueError
        If ``check`` and the cushion is non-positive at some date: the bank is
        already below ``R_star`` on its own reference narrative and the reverse
        stress test has nothing to say.
    """
    _check_aligned(portfolio, scenarios)

    baseline_erosion = (
        portfolio.exposure * regulatory.psi_from_config(scenarios.baseline_pd, cfg)
    ).sum(axis=0)
    h = portfolio.cet1_re(cfg) - cfg.r_star * portfolio.rwa_oth - baseline_erosion

    # H[n] > 0 is *equivalent* to Ratio_0[n] > R_star, since
    #   H[n] = CET1_0[n] - R_star * RWA_0[n]  and  Ratio_0[n] = CET1_0[n] / RWA_0[n].
    # That is what makes calibrate_cet1_for_ratio a sufficient way to guarantee
    # admissibility: pin the baseline ratio above R_star and the cushion follows.
    if check and np.any(h <= 0.0):
        offending = scenarios.dates[h <= 0.0].tolist()
        raise ValueError(
            f"H[n] <= 0 at dates {offending} under {cfg.label} and reference narrative "
            f"{scenarios.baseline_scenario!r}: the bank is already at or below the "
            "breach level on its own baseline, so the reverse stress test is vacuous "
            "(assumption 8). Values: "
            f"{np.array2string(h, precision=3, suppress_small=False)}"
        )
    return h


def calibrate_cet1_for_ratio(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    target_ratio: float,
    date_index: int | None = None,
) -> Portfolio:
    """Rescale ``cet1_0`` so the reference-narrative ratio hits ``target_ratio``.

    Necessary in practice because the NGFS projections carry PD *levels*, not just
    climate increments: the CLIMACRED baseline sits around 5-7% on a European
    corporate book, an order of magnitude above a typical through-the-cycle corporate
    PD. A balance sheet calibrated by eye against textbook PDs is then instantly
    inadmissible (``H[n] < 0``) and the whole exercise is vacuous. Pinning the
    starting ratio makes the admissibility condition hold by construction, and moves
    the calibration choice into the open.

    Parameters
    ----------
    target_ratio : float
        Desired baseline CET1 ratio, as a fraction. Must exceed ``cfg.r_star``,
        otherwise the bank starts in breach.
    date_index : int, optional
        Date at which to pin the ratio. ``None`` (default) pins the *tightest* date,
        so the ratio is at or above the target over the whole horizon -- and hence
        ``H[n] > 0`` everywhere, since ``H[n] > 0`` is equivalent to
        ``Ratio_0[n] > R_star``.

    Returns
    -------
    Portfolio
        Copy with an adjusted ``cet1_0``. Exposures and ``rwa_oth`` are untouched.

    Notes
    -----
    This is the one place where the choice of reference narrative stops being a pure
    normalisation. The breach algebra itself is strictly invariant in ``p0``
    (specification section 3, property 2), but ``cet1_0`` is pinned *against* ``p0``
    here, so a different ``p0`` yields a different balance sheet and hence a different
    ``H[n]``. The effect is a **uniform level shift** of every scenario's distance to
    breach -- the ranking and the gaps between scenarios are untouched -- but the
    absolute distances are not comparable across runs that used different ``p0``.
    Calibrate once, then vary ``p0``, rather than the other way round.
    """
    if target_ratio <= cfg.r_star:
        raise ValueError(
            f"target_ratio={target_ratio} is not above r_star={cfg.r_star}: the bank "
            "would start in breach and the reverse stress test would be vacuous."
        )
    _check_aligned(portfolio, scenarios)

    pd_matrix = scenarios.baseline_pd
    provision = (1.0 - cfg.sensitivity.kappa_tax) * (
        cfg.ell * portfolio.exposure * pd_matrix
    ).sum(axis=0)
    charge = regulatory.capital_charge(pd_matrix, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * (
        charge * portfolio.exposure
    ).sum(axis=0)

    # CET1_RE is flat at cet1_0 under assumption 10, so the required level is explicit
    required = target_ratio * rwa + provision
    cet1_0 = float(required.max() if date_index is None else required[date_index])

    return Portfolio(
        buckets=portfolio.buckets,
        dates=portfolio.dates,
        exposure=portfolio.exposure,
        cet1_0=cet1_0,
        rwa_oth=portfolio.rwa_oth,
    )


# -- erosion and breach --------------------------------------------------------


def erosion(
    portfolio: Portfolio, scenarios: ScenarioSet, cfg: RstConfig
) -> NDArray[np.float64]:
    """Climate capital erosion ``Erosion[s,n]``, in euros.

    ``sum_g E[g,n] * (Psi_g(p[s,g,n]) - Psi_g(p0[g,n]))``, for every scenario at once.

    Returns
    -------
    ndarray, shape (n_scenario, n_date)
        Zero by construction on the row of the reference narrative.
    """
    _check_aligned(portfolio, scenarios)

    # one broadcast over the whole cube: (n_scen, n_bucket, n_date)
    psi_cube = regulatory.psi_from_config(scenarios.pd_cube, cfg)
    psi_baseline = regulatory.psi_from_config(scenarios.baseline_pd, cfg)
    increment = psi_cube - psi_baseline[None, :, :]
    return np.einsum("gn,sgn->sn", portfolio.exposure, increment)


def distance_to_breach(
    portfolio: Portfolio, scenarios: ScenarioSet, cfg: RstConfig, check_cushion: bool = True
) -> NDArray[np.float64]:
    """``H[n] - Erosion[s,n]``, in units of capital.

    Returns
    -------
    ndarray, shape (n_scenario, n_date)
        **Breach is certain where this is negative.** Positive values are the amount
        of capital erosion the bank could still absorb at that date under that
        scenario before crossing ``R_star``.
    """
    h = cushion(portfolio, scenarios, cfg, check=check_cushion)
    return h[None, :] - erosion(portfolio, scenarios, cfg)


def breach_matrix(
    portfolio: Portfolio, scenarios: ScenarioSet, cfg: RstConfig, check_cushion: bool = True
) -> NDArray[np.bool_]:
    """Boolean ``(n_scenario, n_date)`` mask of ``Erosion > H`` (strict)."""
    return distance_to_breach(portfolio, scenarios, cfg, check_cushion) < 0.0


def breach_set(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    date_index: int | None = None,
    check_cushion: bool = True,
) -> list[str]:
    """Scenarios that breach ``R_star``.

    Parameters
    ----------
    date_index : int, optional
        Position on the date grid. ``None`` means "breaches at any date on the
        horizon", the first-passage reading.

    Returns
    -------
    list of str
        ``{s : Erosion[s,n] > H[n]}``, in the scenario order of the cube.
    """
    mask = breach_matrix(portfolio, scenarios, cfg, check_cushion)
    hits = mask.any(axis=1) if date_index is None else mask[:, date_index]
    return [s for s, breached in zip(scenarios.scenarios, hits) if breached]


# -- critical PD ---------------------------------------------------------------


@dataclass(frozen=True)
class CriticalPdResult:
    """Per-bucket critical PDs and the three limiting cases they can fall into.

    Attributes
    ----------
    critical_pd : ndarray, shape (n_scenario, n_bucket, n_date)
        ``p_crit[g,n] = Psi_g^-1(Psi_g(p0[g,n]) + H_tilde[g,n] / E[g,n])``, the PD
        bucket ``g`` would have to reach, on its own, to break the ratio given what
        every other bucket is already doing. ``-inf`` where already breached, ``NaN``
        where unreachable or where exposure is zero.
    residual_cushion : ndarray, same shape
        ``H_tilde[g,n]``, the cushion left after the other buckets have eroded theirs.
    already_breached : ndarray of bool, same shape
        ``H_tilde < 0``: the other buckets suffice on their own, bucket ``g`` needs to
        do nothing.
    unreachable : ndarray of bool, same shape
        The target exceeds ``Psi_g(p_max)``. **Not an error** but an economic result:
        the bucket is too small to break the ratio alone, even at maximum PD.
    zero_exposure : ndarray of bool, same shape
        ``E[g,n] ~ 0``, short-circuited before any inversion is attempted.
    reach_margin : ndarray, same shape
        ``E[g,n] * (Psi_g(p_max) - Psi_g(p0[g,n])) - H_tilde[g,n]``, in euros: the
        capital by which pushing the bucket to the top of the admissible domain
        overshoots (positive) or falls short of (negative) the residual cushion.

        The one ranking key that stays finite and meaningful in all three limiting
        cases, which ``critical_pd`` itself does not -- it is ``-inf`` when already
        breached and ``NaN`` when unreachable, so sorting on it degenerates exactly
        when the portfolio is granular enough for every bucket to be unreachable.
        Larger means closer to breaking the ratio alone.
    """

    critical_pd: NDArray[np.float64]
    residual_cushion: NDArray[np.float64]
    already_breached: NDArray[np.bool_]
    unreachable: NDArray[np.bool_]
    zero_exposure: NDArray[np.bool_]
    reach_margin: NDArray[np.float64]

    @property
    def solved(self) -> NDArray[np.bool_]:
        """Cells where an actual inversion was performed."""
        return ~(self.already_breached | self.unreachable | self.zero_exposure)


def critical_pd(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    exposure_atol: float = 1.0,
    check_cushion: bool = True,
) -> CriticalPdResult:
    """Per-bucket critical PD, the inverse question of the stress test.

    "By how much would the PD of one single bucket have to rise to break the ratio,
    given what all the others are doing?" The other buckets are summed into a residual
    cushion::

        H_tilde[g,n] = H[n] - sum_{g' != g} E[g',n] (Psi_g'(p[g',n]) - Psi_g'(p0[g',n]))
        p_crit[g,n]  = Psi_g^-1( Psi_g(p0[g,n]) + H_tilde[g,n] / E[g,n] )

    Parameters
    ----------
    exposure_atol : float, optional
        Exposures below this (in euros) are treated as zero and short-circuited.
        Default 1 euro.

    Returns
    -------
    CriticalPdResult
        Values plus the masks of the three limiting cases. None of them raises:
        ``brentq`` is never allowed to surface an exception, because "unreachable" is
        a finding, not a failure.
    """
    _check_aligned(portfolio, scenarios)
    lo, hi = cfg.pd_bounds

    h = cushion(portfolio, scenarios, cfg, check=check_cushion)  # (n_date,)
    psi_cube = regulatory.psi_from_config(scenarios.pd_cube, cfg)  # (s, g, n)
    psi_baseline = regulatory.psi_from_config(scenarios.baseline_pd, cfg)  # (g, n)

    contribution = portfolio.exposure[None, :, :] * (psi_cube - psi_baseline[None, :, :])
    total = contribution.sum(axis=1, keepdims=True)  # (s, 1, n)
    # residual cushion: total erosion minus the bucket's own contribution
    residual = h[None, None, :] - (total - contribution)

    exposure = np.broadcast_to(portfolio.exposure[None, :, :], contribution.shape)
    zero_exposure = exposure < exposure_atol
    already_breached = (residual < 0.0) & ~zero_exposure

    with np.errstate(divide="ignore", invalid="ignore"):
        target = psi_baseline[None, :, :] + residual / exposure
    target = np.where(zero_exposure | already_breached, np.nan, target)

    psi_max = float(regulatory.psi_from_config(hi, cfg))
    unreachable = np.isfinite(target) & (target > psi_max)

    solvable = np.where(unreachable, np.nan, target)
    values = regulatory.psi_inv_from_config(solvable, cfg)

    # A target below Psi(p_min) means the bucket is already past its critical PD at
    # the floor; psi_inv returns NaN there, so pin it to the floor instead.
    psi_min = float(regulatory.psi_from_config(lo, cfg))
    below_floor = np.isfinite(target) & (target < psi_min)
    values = np.where(below_floor, lo, values)

    values = np.where(already_breached, -np.inf, values)
    values = np.where(zero_exposure, np.nan, values)

    reach_margin = exposure * (psi_max - psi_baseline[None, :, :]) - residual
    reach_margin = np.where(zero_exposure, -np.inf, reach_margin)

    return CriticalPdResult(
        critical_pd=values,
        residual_cushion=residual,
        already_breached=already_breached,
        unreachable=unreachable,
        zero_exposure=zero_exposure,
        reach_margin=reach_margin,
    )


# -- iso-breach frontier -------------------------------------------------------


def iso_breach_frontier(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    date_index: int,
    n_points: int = 200,
    check_cushion: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iso-breach curve in the ``(p_H, p_L)`` plane, for the two-bucket case.

    The breach boundary is the level set
    ``E_H Psi_H(p_H) + E_L Psi_L(p_L) = const``, a decreasing curve of a separable
    increasing function -- so it is traced analytically, one scalar inversion per
    point, with no root-finding in two dimensions.

    Parameters
    ----------
    date_index : int
        Position on the date grid.
    n_points : int, optional
        Number of points along the ``p_H`` axis.

    Returns
    -------
    p_high, p_low : ndarray
        Points on the frontier. ``p_low`` is ``NaN`` wherever the required
        low-carbon PD falls outside the admissible domain -- that is, where the
        high-carbon bucket alone already determines the outcome.

    Raises
    ------
    ValueError
        If the portfolio does not have exactly two buckets.
    """
    _check_aligned(portfolio, scenarios)
    if portfolio.n_buckets != 2:
        raise ValueError(
            f"the iso-breach frontier is defined for two buckets, got "
            f"{portfolio.n_buckets}: {list(portfolio.buckets)}"
        )

    lo, hi = cfg.pd_bounds
    h = cushion(portfolio, scenarios, cfg, check=check_cushion)[date_index]
    e_high, e_low = portfolio.exposure[:, date_index]
    psi0_high, psi0_low = regulatory.psi_from_config(
        scenarios.baseline_pd[:, date_index], cfg
    )

    # E_H (Psi_H(p_H) - Psi_H(p0_H)) + E_L (Psi_L(p_L) - Psi_L(p0_L)) = H
    constant = h + e_high * psi0_high + e_low * psi0_low

    p_high = np.geomspace(lo, hi, n_points)
    target_low = (constant - e_high * regulatory.psi_from_config(p_high, cfg)) / e_low
    p_low = regulatory.psi_inv_from_config(target_low, cfg)
    return p_high, p_low
