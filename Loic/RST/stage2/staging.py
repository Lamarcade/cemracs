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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Lifetime window in years. The F-IRB default maturity ``M`` of the capital charge,
#: reused here so the two horizons of the model cannot drift apart.
LIFETIME_WINDOW = 2.5

#: Default ``Thresh``: a bucket migrates when its PD reaches this multiple of the
#: baseline. 2.0 mirrors the usual IFRS 9 practice of a doubling.
SICR_THRESHOLD = 2.0


@dataclass(frozen=True)
class SicrRule:
    """A significant-increase-in-credit-risk trigger.

    Attributes
    ----------
    thresh : float
        Multiple of the reference PD at which the relative trigger fires. 3.0 is the
        supervisory backstop -- a threefold increase, i.e. a 200% rise.
    absolute_backstop : float or None
        Reporting-date 12-month PD above which SICR is triggered **on its own**,
        whatever the ratio. 0.20 in the ECB reading. This is an ``OR``, not a filter:
        a deeply impaired obligor is in Stage 2 even if its PD has not tripled.
    low_risk_floor : float or None
        Low credit risk exemption. The *relative* trigger applies only where the
        initial **or** the current PD sits above this level; below it, a tripling is
        immaterial in absolute terms and does not by itself signal SICR. 0.003 in the
        ECB reading. Does not affect the absolute backstop.
    reference : {"origination", "same_date"}
        What the current PD is compared against.

        ``origination`` is IFRS 9 as written: ``scenario_pd[n] / scenario_pd[t_0]``,
        the total PD now against the total PD when the book was written. The reference
        narrative is judged against its own origination too, so **it can migrate** --
        and on this data it does, on 88% of cells, because the CLIMACRED baseline
        triples between 2022 and 2023. That is not an artefact to suppress: a real
        book whose PD triples does move to Stage 2, whatever the cause.

        ``same_date`` compares to the contemporaneous baseline,
        ``p_scenario[n] / p_baseline[n]``, isolating the climate increment. The
        reference narrative then has a ratio of exactly 1 and never migrates.
    name : str
        Label carried into the reports.

    Notes
    -----
    The two PD levels pull in **opposite directions**, which is easy to get backwards.
    The 20% figure *adds* cells to Stage 2 on its own; it is not a ceiling above which
    the relative test stops applying. The 0.3% figure *removes* cells, and only from the
    relative test -- an obligor above 20% is in Stage 2 regardless.
    """

    thresh: float
    absolute_backstop: float | None = 0.20
    low_risk_floor: float | None = 0.003
    reference: str = "origination"
    name: str = ""

    def __post_init__(self) -> None:
        if self.thresh <= 1.0:
            raise ValueError(f"thresh must exceed 1, got {self.thresh}")
        if self.reference not in ("origination", "same_date"):
            raise ValueError(
                f"reference must be 'origination' or 'same_date', got {self.reference!r}"
            )
        for name, value in (("absolute_backstop", self.absolute_backstop),
                            ("low_risk_floor", self.low_risk_floor)):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be a fraction in (0, 1), got {value}")

    @property
    def stages_baseline(self) -> bool:
        """Whether the reference narrative can itself migrate.

        True under ``origination``, and it is what forces the cushion and the
        calibration to be recomputed on a staged baseline -- see
        :func:`stage2.breach2.cushion_staged`.
        """
        return self.reference == "origination"

    @property
    def label(self) -> str:
        """Human-readable description, for figure subtitles and the run header."""
        if self.name:
            return self.name
        denominator = "p(t0)" if self.stages_baseline else "p_baseline(t)"
        parts = [f"p / {denominator} > {self.thresh:g}"]
        if self.low_risk_floor is not None:
            parts[0] += f" (if PD > {self.low_risk_floor:.1%})"
        if self.absolute_backstop is not None:
            parts.append(f"p > {self.absolute_backstop:.0%}")
        return " OR ".join(parts)


#: A plain relative trigger, with neither backstop nor exemption. The reference point
#: for measuring what the two supervisory levels actually add.
SIMPLE_RULE = SicrRule(SICR_THRESHOLD, None, None, "origination",
                       "doubling since origination, no backstop")

#: The ECB reading: a threefold rise since initial recognition, subject to the low
#: credit risk exemption, **or** a reporting-date PD above 20% on its own.
ECB_RULE = SicrRule(3.0, 0.20, 0.003, "origination", "")

#: Climate-attribution variant: the trigger measures the increment over the
#: contemporaneous baseline, so the reference narrative never migrates.
CLIMATE_RULE = SicrRule(SICR_THRESHOLD, None, None, "same_date",
                        "doubling vs the same-date baseline")

#: Selectable presets, for the driver.
SICR_RULES = {"simple": SIMPLE_RULE, "ecb": ECB_RULE, "climate": CLIMATE_RULE}


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
    rule: SicrRule = SIMPLE_RULE,
) -> NDArray[np.bool_]:
    """Which ``(scenario, bucket, date)`` cells have migrated to Stage 2.

    ``p[s,g,n] / p0[g,n] > thresh``, optionally restricted to an applicability band on
    ``p``, and **re-tested at every date**. A bucket returns to Stage 1 as soon as it
    falls back below the trigger, which IFRS 9 permits and which matters here: the DAPS
    shock is transient, migrating 36% of buckets in 2027 and none in 2028. An absorbing
    rule would keep them in Stage 2 to the end of the horizon and hide that.

    Returns
    -------
    ndarray of bool
        Same shape as ``pd_cube``. Under ``same_date`` it is identically False on the
        reference narrative; under ``origination`` the reference narrative migrates
        like any other, which is the point of that rule.
    """
    if rule.stages_baseline:
        # (baseline + pd_adjustment) now, over the same quantity at origination. At
        # t_0 the adjustment is identically zero, so every narrative shares the same
        # denominator -- the book was written before any of them diverged.
        reference = pd_cube[:, :, :1]
    else:
        reference = baseline_pd[None, :, :]

    relative = pd_cube > rule.thresh * reference
    if rule.low_risk_floor is not None:
        # low credit risk exemption: the relative test needs the initial OR the current
        # PD to be material. Both below the floor and a tripling means nothing.
        floor = rule.low_risk_floor
        relative &= (reference > floor) | (pd_cube > floor)

    triggered = relative
    if rule.absolute_backstop is not None:
        # a backstop ADDS cells, it does not filter them: past this level the obligor
        # is in Stage 2 whatever its ratio has done
        triggered = triggered | (pd_cube > rule.absolute_backstop)
    return triggered


def provision_multiplier(
    pd_cube: NDArray[np.float64],
    baseline_pd: NDArray[np.float64],
    rule: SicrRule = SIMPLE_RULE,
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
    ``multiplier >= 1`` everywhere and is asserted, since ``LT`` opens with the full
    12-month term.

    Whether the **reference narrative** carries a multiplier of 1 depends on the rule.
    Under ``same_date`` it does, and the cushion is then the Stage 1 one. Under
    ``origination`` it does not: the baseline migrates too, so the cushion and the
    calibration must both be recomputed on the staged baseline provision --
    :func:`stage2.breach2.cushion_staged` and
    :func:`stage2.breach2.calibrate_staged` exist for exactly that.
    """
    flags = sicr_flags(pd_cube, baseline_pd, rule)
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
