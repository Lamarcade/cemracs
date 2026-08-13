"""Breach analysis with a staged provision channel.

Mirrors the parent :mod:`breach` module, with one change: the provision term of ``Psi``
carries the per-cell multiplier from :mod:`stage2.staging`.

    Psi_staged(p) = (1 - kappa_tax) * ell * lambda * p  +  12.5 * R_star * K(p)

**Only the numerator moves.** Basel computes the IRB charge on the 12-month PD whatever
IFRS 9 stage a loan sits in, so ``K(p)`` is untouched and the RWA channel is the Stage 1
one. That is what keeps the change this small.

**The cushion is not redefined here.** The SICR reference is the same-date baseline, so
the reference narrative compares to itself, never triggers, and carries ``lambda = 1``.
``H[n]`` is therefore exactly the Stage 1 cushion and :func:`breach.cushion` is called
unchanged -- as are :func:`breach.calibrate_cet1_for_ratio` and the whole of
:mod:`portfolio`, :mod:`scenarios` and :mod:`config`.

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


def erosion_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Climate capital erosion in euros, shape ``(n_scenario, n_date)``.

    ``sum_g E[g,n] * (Psi_staged(p[s,g,n]) - Psi(p0[g,n]))``. The baseline term carries
    no multiplier: it is Stage 1 by construction.
    """
    staged = psi_staged(scenarios.pd_cube, cfg, multiplier)
    baseline = regulatory.psi_from_config(scenarios.baseline_pd, cfg)
    return np.einsum("gn,sgn->sn", portfolio.exposure, staged - baseline[None, :, :])


def distance_to_breach_staged(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    multiplier: NDArray[np.float64],
    check_cushion: bool = False,
) -> NDArray[np.float64]:
    """``H[n] - Erosion[s,n]``. Breach is certain where this is negative."""
    cushion = breach.cushion(portfolio, scenarios, cfg, check=check_cushion)
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
