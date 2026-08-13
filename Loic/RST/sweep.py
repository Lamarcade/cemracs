"""Sweep the brown/green split of the loan book and see what it moves.

The Level 1 balance sheet is static, so the share of the book sitting in the
carbon-intensive bucket is the natural comparative-statics knob -- the static analogue
of the origination control ``q[g,n]`` that the underlying note leaves to the upstream
stochastic problem. This module varies it and reports the distance to breach.

Computation only; the figures live in :mod:`report`, as ``breach`` is to ``report``.

Three findings the sweep was built around, all measured on the current defaults. They
matter because two of them are counter-intuitive and a chart alone would mislead:

1. **The share is a weak lever.** Moving it from 0 to 100% shifts the distance to
   breach by ~1.5 bn EUR, where moving ``target_ratio`` from 0.150 to 0.115 shifts it
   by ~23 bn. A factor of about 15, which is why
   :func:`sweep_share_and_target` exists alongside the one-dimensional sweep.
2. **On EU27 nothing crosses.** DAPS is the only scenario that bites there and its
   distance *rises* with the brown share, so no breach appears or disappears anywhere
   on the range. Crossings do exist on wider region sets.
3. **Recalibrating the balance sheet at each share, or freezing it, gives the same
   answer** (DAPS 4.58 -> 5.73 against 4.39 -> 6.17 over the full range). The sweep
   recalibrates, consistently with the rest of the chain.

Units: shares are fractions of the book, distances are euros.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import breach
import portfolio as pf
from config import RstConfig
from scenarios import ScenarioSet

#: Keeps the endpoints inside ``0 < high_carbon_share < 1``, which
#: :func:`portfolio.stylised_hl_portfolio` enforces. Bounding here rather than relaxing
#: the constructor: a portfolio with an empty bucket is a real error everywhere else.
SHARE_EPS = 1e-9

#: Default target ratios of the two-dimensional grid, spanning the breach frontier.
DEFAULT_TARGETS = (0.150, 0.140, 0.130, 0.125, 0.120, 0.115)


@dataclass(frozen=True)
class ShareSweep:
    """Distance to breach as a function of the carbon-intensive share.

    Attributes
    ----------
    shares : ndarray, shape (n_share,)
        Fractions of the book in bucket ``H``.
    scenarios : tuple of str
        Scenario names, in the column order of ``distance``.
    distance : ndarray, shape (n_share, n_scenario)
        ``min_n (H[n] - Erosion[s,n])`` in euros, the first-passage reading of the
        horizon. Negative means the scenario breaches at some date.
    cet1_0 : ndarray, shape (n_share,)
        What the calibration produced at each share, in euros. Nearly flat in practice;
        worth carrying so a reader can check the calibration is not doing the work.
    min_cushion : ndarray, shape (n_share,)
        Smallest ``H[n]`` at each share. Non-positive entries mean the exercise was
        inadmissible there (assumption 8) and the row should not be read.
    region_label, cfg_label : str
        What was swept, for figure titles.
    """

    shares: NDArray[np.float64]
    scenarios: tuple[str, ...]
    distance: NDArray[np.float64]
    cet1_0: NDArray[np.float64]
    min_cushion: NDArray[np.float64]
    region_label: str
    cfg_label: str

    @property
    def worst_scenario(self) -> list[str]:
        """Name of the tightest scenario at each share."""
        return [self.scenarios[i] for i in self.distance.argmin(axis=1)]

    @property
    def inadmissible(self) -> NDArray[np.bool_]:
        """Shares where the cushion was non-positive."""
        return self.min_cushion <= 0.0


def sweep_carbon_share(
    scenarios: ScenarioSet,
    cfg: RstConfig,
    shares: NDArray[np.float64] | list[float] | None = None,
    n_points: int = 21,
    target_ratio: float = 0.13,
    region_label: str = "",
) -> ShareSweep:
    """Vary the share of the book in bucket ``H`` and record the distance to breach.

    Parameters
    ----------
    scenarios : ScenarioSet
        A **two-bucket** set, ``("H", "L")``, already clipped. Built once by the caller
        and reused across every share: the PD cube does not depend on the split, only
        the portfolio does, which is what makes the two-dimensional grid affordable.
    cfg : RstConfig
    shares : array_like, optional
        Shares to evaluate. Defaults to ``n_points`` values spanning ``[0, 1]``.
    n_points : int, optional
        Grid size when ``shares`` is not given. Default 21.
    target_ratio : float, optional
        Baseline CET1 ratio the balance sheet is pinned to at *every* share, via
        :func:`breach.calibrate_cet1_for_ratio`. Default 0.13.
    region_label : str, optional
        Free-text label carried to the figures.

    Returns
    -------
    ShareSweep

    Notes
    -----
    **The share is a secondary lever.** Over the whole range it moves the distance by
    roughly a tenth of what ``target_ratio`` does. Reading a share sweep on its own
    invites the conclusion that portfolio composition decides the outcome; it does not,
    capitalisation does. :func:`sweep_share_and_target` puts the two side by side.

    **Distances are not comparable across ``target_ratio``.** The balance sheet is
    recalibrated at every point, so a sweep at 0.13 and a sweep at 0.12 sit on
    different capital bases -- the same trap already documented in
    :func:`breach.calibrate_cet1_for_ratio`.
    """
    if scenarios.n_buckets != 2:
        raise ValueError(
            f"the share sweep is defined for the two-bucket book, got "
            f"{scenarios.n_buckets}: {list(scenarios.buckets)}. Use --granularity carbon."
        )

    grid = (
        np.linspace(0.0, 1.0, n_points) if shares is None else np.asarray(shares, float)
    )
    distance = np.empty((grid.size, scenarios.n_scenarios), dtype=float)
    cet1_0 = np.empty(grid.size, dtype=float)
    min_cushion = np.empty(grid.size, dtype=float)

    for i, share in enumerate(grid):
        port = pf.stylised_hl_portfolio(
            scenarios.dates,
            high_carbon_share=float(np.clip(share, SHARE_EPS, 1.0 - SHARE_EPS)),
        )
        port = breach.calibrate_cet1_for_ratio(
            port, scenarios, cfg, target_ratio=target_ratio
        )
        # check=False: a sweep reports inadmissibility, it does not stop on it
        cushion = breach.cushion(port, scenarios, cfg, check=False)
        erosion = breach.erosion(port, scenarios, cfg)
        distance[i] = (cushion[None, :] - erosion).min(axis=1)
        cet1_0[i] = port.cet1_0
        min_cushion[i] = cushion.min()

    return ShareSweep(
        shares=grid,
        scenarios=scenarios.scenarios,
        distance=distance,
        cet1_0=cet1_0,
        min_cushion=min_cushion,
        region_label=region_label,
        cfg_label=cfg.label,
    )


def critical_shares(sweep: ShareSweep) -> pd.DataFrame:
    """Share at which each scenario crosses into breach, if it ever does.

    Returns
    -------
    DataFrame
        Indexed by scenario, columns ``critical_share`` (linearly interpolated at the
        first sign change, ``NaN`` when there is none), ``direction``, and the distance
        at each end of the range.

        ``direction`` is ``"breaches above"`` when the distance falls through zero as
        the brown share rises, ``"breaches below"`` when it rises through zero, and
        ``"always"`` / ``"never"`` when there is no crossing.

    Notes
    -----
    **A rising distance is not a bug.** ``carbon_bucket_transition`` is derived from
    HWTP, so it orders the *transition* narratives and has no reason to order a
    physical or stagnation shock. On EU27, DAPS -- the only scenario that bites there --
    gets **safer** as the brown share rises, and no scenario crosses at all. That
    inversion is the out-of-sample signal the mapping header warns about, not a defect.
    """
    rows = []
    for j, name in enumerate(sweep.scenarios):
        d = sweep.distance[:, j]
        sign_change = np.nonzero(np.diff(np.signbit(d)))[0]

        if sign_change.size == 0:
            direction = "always" if (d < 0).all() else "never"
            crossing = np.nan
        else:
            k = int(sign_change[0])
            lo, hi = d[k], d[k + 1]
            # linear interpolation between the bracketing grid points
            weight = lo / (lo - hi)
            crossing = float(sweep.shares[k] + weight * (sweep.shares[k + 1] - sweep.shares[k]))
            direction = "breaches above" if hi < lo else "breaches below"

        rows.append(
            {
                "scenario": name,
                "critical_share": crossing,
                "direction": direction,
                "distance_at_0_bn": d[0] / 1e9,
                "distance_at_1_bn": d[-1] / 1e9,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def sweep_share_and_target(
    scenarios: ScenarioSet,
    cfg: RstConfig,
    shares: NDArray[np.float64] | list[float] | None = None,
    targets: tuple[float, ...] | list[float] = DEFAULT_TARGETS,
    n_points: int = 21,
) -> pd.DataFrame:
    """Distance of the tightest scenario over a ``share x target_ratio`` grid.

    The context the one-dimensional sweep needs: it shows that the breach frontier runs
    almost horizontally, i.e. capitalisation decides the outcome and composition only
    modulates it.

    Returns
    -------
    DataFrame
        Index ``target_ratio``, columns ``share``, values in euros. Negative means the
        worst scenario breaches somewhere on the horizon. Rows at or below
        ``cfg.r_star`` are all ``NaN``, see below.

    Warns
    -----
    UserWarning
        Naming the target ratios dropped for being at or below the breach level. Pinning
        the bank *below* the threshold it is measured against starts it in breach and
        makes the exercise vacuous -- :func:`breach.calibrate_cet1_for_ratio` refuses
        it. This bites the relative convention in particular, whose ``R_star`` sits just
        under a plausible target: at ``R_star = 0.1261`` every target from 0.125 down is
        out. The threshold is held fixed while capitalisation varies, which is the
        meaningful reading -- recomputing ``R_star`` from each target would make the
        grid constant by construction.
    """
    grid = (
        np.linspace(0.0, 1.0, n_points) if shares is None else np.asarray(shares, float)
    )
    values = np.full((len(targets), grid.size), np.nan, dtype=float)

    dropped = []
    for i, target in enumerate(targets):
        if float(target) <= cfg.r_star:
            dropped.append(float(target))
            continue
        swept = sweep_carbon_share(
            scenarios, cfg, shares=grid, target_ratio=float(target)
        )
        values[i] = swept.distance.min(axis=1)

    if dropped:
        warnings.warn(
            f"{len(dropped)} target ratio(s) dropped from the grid under "
            f"{cfg.label}: {dropped}. Pinning the baseline ratio at or below R_star "
            "starts the bank in breach, so the exercise has nothing to measure there.",
            stacklevel=2,
        )
    if np.isnan(values).all():
        raise ValueError(
            f"every target ratio is at or below R_star={cfg.r_star:.4f} under "
            f"{cfg.label}: nothing to plot. Raise --sweep-targets."
        )

    return pd.DataFrame(
        values,
        index=pd.Index([float(t) for t in targets], name="target_ratio"),
        columns=pd.Index(grid, name="share"),
    )
