"""The SICR trigger and the lifetime multiplier it switches on.

Under Stage 1 the impairment is ``ell * E * p``, the 12-month expected loss. Under
Stage 2 it becomes ``ell * E * LT``, the expected loss over the remaining life. Writing
``lambda = LT / p``, the provision channel is simply multiplied by ``lambda`` wherever a
bucket has migrated -- which is what keeps the whole affine breach machinery intact.

Two choices here decide whether the extension measures climate or measures artefacts of
the source file. Both were settled by measurement, not by preference.

**The lifetime window is fixed at 2.5 years, not "to the end of the data".** Reading
"remaining life" as "however many years the spreadsheet still has" makes the multiplier
a function of the calendar rather than of credit: it runs 4.5 at 2024 down to exactly
1.00 at 2030, and barely differs between BAU (4.51) and DAPS (5.15). At that setting the
reference narrative itself breaches, which would make the exercise vacuous. A fixed
window of :data:`LIFETIME_WINDOW` -- the F-IRB maturity ``M`` that ``MA(p)`` already
uses -- gives a stable ~2.2 and discriminates properly.

**The SICR reference is the same-date baseline PD, not the PD at ``t_0``.** CLIMACRED's
BAU path triples between 2022 and 2023, so a trigger relative to ``t_0`` fires on
baseline drift alone: 100% of buckets migrate in 2023, under BAU included. Comparing to
the same-date baseline measures the *climate* increment, which is what the rest of the
model already does everywhere (``Erosion = Psi(p) - Psi(p0)``).

That second choice has a consequence worth stating plainly: this is **not** IFRS 9 SICR,
which is judged against origination. It is a climate-attribution SICR -- "PD doubled
versus the counterfactual". The upside is that the reference narrative can never trigger
(its ratio is exactly 1), so the cushion ``H[n]`` is untouched by staging and
admissibility (assumption 8) is preserved by construction.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

#: Lifetime window in years. The F-IRB default maturity ``M`` of the capital charge,
#: reused here so the two horizons of the model cannot drift apart.
LIFETIME_WINDOW = 2.5

#: Default ``Thresh``: a bucket migrates when its PD reaches this multiple of the
#: baseline. 2.0 mirrors the usual IFRS 9 practice of a doubling.
SICR_THRESHOLD = 2.0


def lifetime_pd(
    pd_cube: NDArray[np.float64], window: float = LIFETIME_WINDOW
) -> NDArray[np.float64]:
    """Cumulative default probability over a fixed forward window.

    ``LT[s,g,n] = sum_m S[m] * p[s,g,m] * step``, where ``S`` is the survival probability
    accumulated from date ``n`` and the final step is fractional when the window is not a
    whole number of years.

    Parameters
    ----------
    pd_cube : ndarray, shape (n_scenario, n_bucket, n_date)
        PDs as fractions, already clipped onto the admissible domain.
    window : float, optional
        Window in years. Default :data:`LIFETIME_WINDOW`.

    Returns
    -------
    ndarray
        Same shape as ``pd_cube``.

    Notes
    -----
    **The last PD is held flat beyond the end of the grid.** Without it the window
    truncates against the data rather than against the loan: ``lambda`` would fall to
    exactly 1.00 in the final year, making Stage 2 vanish at the end of the horizon for
    no reason other than where the file stops. Holding the last projection flat is the
    standard assumption and keeps the multiplier a property of the credit path.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    n_dates = pd_cube.shape[2]
    n_extra = int(np.ceil(window))
    extended = np.concatenate(
        [pd_cube, np.repeat(pd_cube[:, :, -1:], n_extra, axis=2)], axis=2
    )

    out = np.zeros_like(pd_cube)
    for n in range(n_dates):
        accumulated = np.zeros(pd_cube.shape[:2])
        survival = np.ones(pd_cube.shape[:2])
        remaining, m = float(window), n
        while remaining > 1e-12:
            step = min(1.0, remaining)
            marginal = extended[:, :, m] * step
            accumulated += survival * marginal
            survival *= 1.0 - marginal
            remaining -= step
            m += 1
        out[:, :, n] = accumulated
    return out


def sicr_flags(
    pd_cube: NDArray[np.float64],
    baseline_pd: NDArray[np.float64],
    thresh: float = SICR_THRESHOLD,
) -> NDArray[np.bool_]:
    """Which ``(scenario, bucket, date)`` cells have migrated to Stage 2.

    ``p[s,g,n] / p0[g,n] > thresh``, **re-tested at every date**. A bucket returns to
    Stage 1 as soon as its ratio falls back below the threshold, which IFRS 9 permits
    and which matters here: the DAPS shock is transient, migrating 36% of buckets in
    2027 and none in 2028. An absorbing rule would keep them in Stage 2 to the end of
    the horizon and hide that.

    Returns
    -------
    ndarray of bool
        Same shape as ``pd_cube``. Identically False on the reference narrative.
    """
    if thresh <= 1.0:
        raise ValueError(
            f"thresh must exceed 1, got {thresh}: at or below 1 the reference narrative "
            "triggers against itself and the cushion stops being Stage 1."
        )
    return pd_cube > thresh * baseline_pd[None, :, :]


def provision_multiplier(
    pd_cube: NDArray[np.float64],
    baseline_pd: NDArray[np.float64],
    thresh: float = SICR_THRESHOLD,
    window: float = LIFETIME_WINDOW,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The per-cell factor on the provision channel, and the flags that produced it.

    Returns
    -------
    multiplier : ndarray
        ``1.0`` where the bucket is Stage 1, ``LT/p`` where it is Stage 2.
    flags : ndarray of bool
        The Stage 2 mask, returned alongside so callers can report migration without
        recomputing it.

    Warns
    -----
    UserWarning
        When a migrated cell carries a multiplier far above the window. ``window`` is
        the ceiling only on a *flat* path -- there ``LT`` is ``window`` years of the same
        PD. On a rising path ``LT/p`` is unbounded, and legitimately so: a bucket whose
        PD is about to jump has a lifetime provision far above ``window`` times its
        current 12-month one. That is the anticipation effect Stage 2 exists to capture,
        but a very large value usually means the *denominator* is small, so the figure
        says more about today's PD than about the loss ahead.

    Notes
    -----
    Two invariants hold by construction and are asserted rather than trusted:

    ``multiplier >= 1`` everywhere, since ``LT`` opens with the full 12-month term.

    ``multiplier == 1`` on the reference narrative, which compares to itself and so can
    never trigger. That is what keeps ``H[n]`` the Stage 1 cushion and lets
    :func:`breach.cushion` and :func:`breach.calibrate_cet1_for_ratio` be reused as they
    stand.
    """
    flags = sicr_flags(pd_cube, baseline_pd, thresh)
    ratio = lifetime_pd(pd_cube, window) / pd_cube
    multiplier = np.where(flags, ratio, 1.0)

    if not np.all(multiplier >= 1.0 - 1e-12):
        raise AssertionError("provision multiplier below 1: lifetime ECL under 12-month ECL")

    if flags.any():
        worst = float(multiplier[flags].max())
        if worst > 1.5 * window:
            warnings.warn(
                f"a migrated bucket carries a provision multiplier of {worst:.1f}, well "
                f"above the {window:g} a flat PD path would give. Its PD is rising "
                "steeply inside the window, so the lifetime provision is driven by "
                "anticipation rather than by the current level -- check the PD path "
                "before quoting that cell.",
                stacklevel=2,
            )
    return multiplier, flags


def migration_table(
    flags: NDArray[np.bool_], scenarios: tuple[str, ...], dates: NDArray[np.int64]
):
    """Share of buckets in Stage 2, one row per scenario and one column per date."""
    import pandas as pd

    return pd.DataFrame(
        flags.mean(axis=1),
        index=pd.Index(scenarios, name="scenario"),
        columns=pd.Index(dates, name="date"),
    )
