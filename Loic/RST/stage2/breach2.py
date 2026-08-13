"""Breach analysis with a staged provision channel.

Mirrors the parent :mod:`breach` module, with one change: the provision term of ``Psi``
carries the per-cell multiplier from :mod:`stage2.staging`.

    Psi_staged(p) = (1 - kappa_tax) * ell * lambda * p  +  12.5 * R_star * K(p)

**Only the numerator moves.** Basel computes the IRB charge on the 12-month PD whatever
IFRS 9 stage a loan sits in, so ``K(p)`` is untouched and the RWA channel is the Stage 1
one. That is what keeps the change this small.

**The cushion depends on the SICR reference.** Under ``same_date`` the reference
narrative compares to itself, never migrates, and ``H[n]`` is exactly the Stage 1
cushion. Under ``origination`` -- IFRS 9 as written -- the baseline is judged against
its own origination and migrates like anything else, so the cushion, the calibration
and the relative ``R_star`` all have to be rebuilt on the staged baseline. That is what
:func:`cushion_staged`, :func:`calibrate_staged` and :func:`baseline_ratio_staged` are
for. Everything else is still reused unchanged: :mod:`portfolio`, :mod:`scenarios`,
:mod:`config` and ``regulatory.capital_charge``.

The breach inequality itself is unaffected in form. Its derivation never assumed the
provision was a *function* of the current PD, only that it was some amount per unit of
exposure, so ``(12)`` stays an exact rewrite of ``(11)``.
:func:`check_forms_agree_staged` is what verifies that claim once ``lambda`` is in play.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import breach
import regulatory
from config import RstConfig
from portfolio import Portfolio
from scenarios import ScenarioSet


def psi_staged(
    pd_values: NDArray[np.float64], cfg: RstConfig, multiplier: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Erosion function with the provision channel scaled by ``multiplier``.

    Passing ``multiplier = 1`` reproduces :func:`regulatory.psi_from_config` exactly,
    which is what makes a Stage 1 regression check possible.
    """
    provision = (1.0 - cfg.sensitivity.kappa_tax) * cfg.ell * multiplier * pd_values
    return provision + 12.5 * cfg.r_star * regulatory.capital_charge(pd_values, cfg.ell)


def baseline_multiplier(
    scenarios: ScenarioSet, multiplier: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The reference narrative's own row of the multiplier cube, shape ``(n_bucket, n_date)``.

    Identically 1 under the ``same_date`` rule. Under ``origination`` the baseline
    migrates like any other narrative, and this is what the cushion has to be built on.
    """
    return multiplier[scenarios.baseline_index]


def cushion_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
    check: bool = False,
) -> NDArray[np.float64]:
    """Reference capital cushion with the baseline's own staging applied.

    ``H[n] = (CET1_RE[n] - R* * RWA_oth) - sum_g E[g,n] * Psi_staged(p0[g,n])``.

    Reduces to :func:`breach.cushion` exactly when the baseline carries a multiplier of
    1, which is the ``same_date`` case. Under ``origination`` the baseline migrates and
    the cushion genuinely shrinks -- a bank whose reference path already triples its PD
    is provisioning on a lifetime basis before any climate increment is applied.

    Raises
    ------
    ValueError
        If ``check`` and the cushion is non-positive somewhere. Left off by default:
        under ``origination`` an inadmissible cushion is a *result* -- it says the
        reference narrative alone puts the bank under the threshold -- and the driver
        reports it rather than aborting.
    """
    staged = psi_staged(scenarios.baseline_pd, cfg, baseline_multiplier(scenarios, multiplier))
    h = (
        portfolio.cet1_re(cfg)
        - cfg.r_star * portfolio.rwa_oth
        - (portfolio.exposure * staged).sum(axis=0)
    )
    if check and np.any(h <= 0.0):
        offending = scenarios.dates[h <= 0.0].tolist()
        raise ValueError(
            f"staged H[n] <= 0 at dates {offending} under {cfg.label}: with the baseline "
            "itself in Stage 2, the reference narrative is already at or below the "
            "breach level."
        )
    return h


def calibrate_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
    target_ratio: float,
) -> Portfolio:
    """Pin the **staged** baseline CET1 ratio to ``target_ratio``.

    The counterpart of :func:`breach.calibrate_cet1_for_ratio` once the baseline is
    itself in Stage 2. Using the Stage 1 calibration instead is a different and equally
    legitimate question -- "the bank was capitalised for Stage 1, then staging hit" --
    and the driver reports both, because they answer different things and disagree
    sharply: on EU27 the Stage 1 calibration leaves the exercise inadmissible under the
    relative convention, while recalibrating restores it.
    """
    if target_ratio <= cfg.r_star:
        raise ValueError(
            f"target_ratio={target_ratio} is not above r_star={cfg.r_star}"
        )
    lam0 = baseline_multiplier(scenarios, multiplier)
    provision = (1.0 - cfg.sensitivity.kappa_tax) * (
        cfg.ell * portfolio.exposure * lam0 * scenarios.baseline_pd
    ).sum(axis=0)
    charge = regulatory.capital_charge(scenarios.baseline_pd, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * (
        charge * portfolio.exposure
    ).sum(axis=0)

    return Portfolio(
        buckets=portfolio.buckets,
        dates=portfolio.dates,
        exposure=portfolio.exposure,
        cet1_0=float((target_ratio * rwa + provision).max()),
        rwa_oth=portfolio.rwa_oth,
    )


def baseline_ratio_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> NDArray[np.float64]:
    """CET1 ratio on the reference narrative, with its own staging applied.

    What the relative ``R_star`` convention must be derived from once the baseline can
    migrate: a threshold built from the *unstaged* baseline would be compared against a
    staged path, and the reference narrative would appear to breach for no reason other
    than the mismatch.
    """
    lam0 = baseline_multiplier(scenarios, multiplier)
    provision = (1.0 - cfg.sensitivity.kappa_tax) * (
        cfg.ell * portfolio.exposure * lam0 * scenarios.baseline_pd
    ).sum(axis=0)
    charge = regulatory.capital_charge(scenarios.baseline_pd, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * (
        charge * portfolio.exposure
    ).sum(axis=0)
    return (portfolio.cet1_re(cfg) - provision) / rwa


def erosion_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Climate capital erosion in euros, shape ``(n_scenario, n_date)``.

    ``sum_g E[g,n] * (Psi_staged(p[s,g,n]) - Psi_staged(p0[g,n]))``. **Both** terms
    carry their own multiplier: under ``origination`` the baseline is staged too, and
    measuring a staged scenario against an unstaged reference would count the baseline's
    own migration as climate erosion.
    """
    staged = psi_staged(scenarios.pd_cube, cfg, multiplier)
    baseline = psi_staged(
        scenarios.baseline_pd, cfg, baseline_multiplier(scenarios, multiplier)
    )
    return np.einsum("gn,sgn->sn", portfolio.exposure, staged - baseline[None, :, :])


def distance_to_breach_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
    check_cushion: bool = False,
) -> NDArray[np.float64]:
    """``H[n] - Erosion[s,n]``. Breach is certain where this is negative."""
    cushion = cushion_staged(portfolio, scenarios, cfg, multiplier, check=check_cushion)
    return cushion[None, :] - erosion_staged(portfolio, scenarios, cfg, multiplier)


def breach_set_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> list[str]:
    """Scenarios breaching at any date on the horizon, strict inequality."""
    distance = distance_to_breach_staged(portfolio, scenarios, cfg, multiplier)
    return [
        name for name, hit in zip(scenarios.scenarios, (distance < 0.0).any(axis=1)) if hit
    ]


def cet1_ratio_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> NDArray[np.float64]:
    """CET1 ratio evaluated directly, shape ``(n_scenario, n_date)``.

    Equation (11) with the staged impairment: the provision is the lifetime ECL where a
    bucket has migrated, the 12-month ECL elsewhere. The denominator is unchanged.
    """
    provision = (1.0 - cfg.sensitivity.kappa_tax) * np.einsum(
        "gn,sgn->sn", cfg.ell * portfolio.exposure, multiplier * scenarios.pd_cube
    )
    charge = regulatory.capital_charge(scenarios.pd_cube, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * np.einsum(
        "gn,sgn->sn", portfolio.exposure, charge
    )
    return (portfolio.cet1_re(cfg)[None, :] - provision) / rwa


def check_forms_agree_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> dict[str, float | int]:
    """Verify that the affine form still reproduces the ratio form once staged.

    The central check of this package, transposing :func:`breach.check_forms_agree`.
    Adding ``lambda`` means touching the provision in two places -- inside ``Psi``, and
    inside the directly evaluated ratio -- and nothing else would catch an inconsistency
    between them. A cell can be wrong in both by the same amount and still look right on
    every other diagnostic.

    Checks ``CET1[n] - R* RWA[n] == H[n] - Erosion[n]`` numerically, and the flag
    ``Ratio[n] < R*  <=>  Erosion[n] > H[n]``.

    Returns
    -------
    dict
        ``max_rel_gap``, ``n_cells``, ``n_breach_ratio``, ``n_breach_affine``,
        ``n_staged_cells``, ``tightest_cell``.

    Raises
    ------
    ValueError
        If the flags disagree anywhere, quoting the offending cells.
    """
    ratio = cet1_ratio_staged(portfolio, scenarios, cfg, multiplier)

    provision = (1.0 - cfg.sensitivity.kappa_tax) * np.einsum(
        "gn,sgn->sn", cfg.ell * portfolio.exposure, multiplier * scenarios.pd_cube
    )
    charge = regulatory.capital_charge(scenarios.pd_cube, cfg.ell)
    rwa = portfolio.rwa_oth + cfg.regulatory.rwa_factor * np.einsum(
        "gn,sgn->sn", portfolio.exposure, charge
    )
    signed = (portfolio.cet1_re(cfg)[None, :] - provision) - cfg.r_star * rwa
    # (11) and (12) can only agree if the cushion carries the baseline's own staging,
    # which is the whole point of routing distance_to_breach_staged through
    # cushion_staged rather than breach.cushion.

    affine = distance_to_breach_staged(portfolio, scenarios, cfg, multiplier)
    gap = np.abs(signed - affine)
    scale = max(float(np.abs(affine).max()), 1.0)

    flag_ratio = ratio < cfg.r_star
    flag_affine = affine < 0.0
    disagree = np.nonzero(flag_ratio != flag_affine)
    if disagree[0].size:
        cells = [
            f"{scenarios.scenarios[s]}@{int(scenarios.dates[n])} "
            f"(ratio {ratio[s, n]:.6f} vs R*={cfg.r_star:.6f}, affine {affine[s, n]:+.3e})"
            for s, n in zip(*disagree)
        ]
        raise ValueError(
            f"staged affine form disagrees with the staged ratio form on {len(cells)} "
            f"cell(s): {cells}. The multiplier was not applied consistently to both."
        )

    s, n = np.unravel_index(int(np.abs(affine).argmin()), affine.shape)
    return {
        "max_rel_gap": float(gap.max() / scale),
        "n_cells": int(affine.size),
        "n_breach_ratio": int(flag_ratio.sum()),
        "n_breach_affine": int(flag_affine.sum()),
        "n_staged_cells": int((multiplier > 1.0).sum()),
        "tightest_cell": float(np.abs(affine[s, n])),
    }
