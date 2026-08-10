"""Configuration objects for the CET1 climate reverse stress test (Level 1).

The three parameter categories of the specification are three distinct types, so
that the status of any number used downstream is visible from its container:

- :class:`RegulatoryConstants` -- frozen by regulation. Changing these is not a
  sensitivity, it is a different regulatory regime.
- :class:`SensitivityParams` -- declared sensitivities. Every one of them is meant
  to be varied, and ``r_star`` in particular is meant to be run under both
  conventions in parallel.
- Calibrated inputs (``E[g,n]``, ``CET1_0``, ``RWA_oth``, ``pbar[s,g,n]``) are not
  here: they live in :mod:`portfolio` and :mod:`scenarios` because they are data,
  not settings.

No magic number may appear in the body of any other module: everything flows
through :class:`RstConfig`.

Units convention, checked at construction: exposures and CET1 are in euros, PDs
and ``r_star`` are fractions, never percentages.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

# Numerical facts established analytically in the specification (section 4).
# They are constants, not tunables: do not re-derive them at runtime.

#: Pole of the maturity adjustment, where ``1 - 1.5 * b(p) = 0``. Below it ``MA``
#: turns negative and the capital charge changes sign, which is meaningless.
MA_POLE = 2.927e-06

#: Argmax of the IRB capital charge ``K``. Beyond it ``K`` is *decreasing*: the
#: charge is not monotone in PD, which is the usual source of surprise here.
K_ARGMAX = 0.2962

#: Root of ``Psi'``, i.e. the hard upper bound of the invertibility domain. The
#: provision channel ``ell`` absorbs the negative slope of ``K`` up to this point.
PSI_MONOTONE_MAX = 0.7202

#: Prudential relative-depletion benchmark: 300 basis points off the current ratio.
RELATIVE_DEPLETION = 0.03

RStarConvention = Literal["absolute", "relative"]


@dataclass(frozen=True)
class RegulatoryConstants:
    """Parameters frozen by the Basel III / CRR3 framework.

    Attributes
    ----------
    ell : float
        Loss given default. 0.40 is the F-IRB supervisory value for senior
        unsecured corporate exposures (CRR3 art. 161(1); it was 0.45 before).
        Held constant across buckets under assumption 3 -- no stranded-collateral
        channel, no downturn PD-LGD correlation.
    maturity : float
        Effective maturity ``M`` in years. 2.5 is the F-IRB default, and it is what
        collapses the maturity adjustment to ``MA(p) = 1 / (1 - 1.5 * b(p))``.
    rwa_factor : float
        ``12.5 = 1 / 0.08``, converting a capital charge into risk-weighted assets.
    pd_floor : float
        Basel regulatory PD floor, 3 basis points. Also the lower bound of the
        admissible domain: it sits 102x above :data:`MA_POLE`.
    """

    ell: float = 0.40
    maturity: float = 2.5
    rwa_factor: float = 12.5
    pd_floor: float = 3e-04

    def __post_init__(self) -> None:
        if not 0.0 < self.ell <= 1.0:
            raise ValueError(f"ell must be a fraction in (0, 1], got {self.ell}")
        if self.pd_floor < MA_POLE:
            raise ValueError(
                f"pd_floor={self.pd_floor:.3e} is below the maturity-adjustment pole "
                f"{MA_POLE:.3e}: the capital charge changes sign there."
            )
        if self.maturity != 2.5:
            raise ValueError(
                "maturity != 2.5 requires the full maturity factor "
                "(1 + (M - 2.5) * b(p)) / (1 - 1.5 * b(p)), which regulatory.py does "
                "not implement. Level 1 fixes M at the F-IRB default."
            )


@dataclass(frozen=True)
class SensitivityParams:
    """Parameters declared as sensitivities, to be varied deliberately.

    Attributes
    ----------
    r_star : float
        Breach level of the CET1 ratio, as a fraction. Two conventions are in use
        and they do *not* rank scenarios the same way -- that disagreement is a
        result in itself, so both are meant to be run in parallel. See
        :meth:`RstConfig.r_star_conventions`.
    r_star_convention : {"absolute", "relative"}
        Which convention ``r_star`` came from. Carried along for labelling only;
        nothing in the maths reads it.
    pd_max : float
        Upper bound of the admissible PD domain. 0.30 is an *economic credibility*
        bound, not a mathematical one: the hard bound is :data:`PSI_MONOTONE_MAX`,
        so there is a factor ~2.4 of headroom.
    baseline_scenario : str or None
        Reference narrative ``s_0`` defining ``p0[g,n]``. Mathematically a pure
        normalisation (the breach set is invariant), but not neutral for
        interpretation. ``None`` means "let the caller pick at load time".
    phi : float
        Fraction of earnings distributed. ``1.0`` is the Level 1 baseline
        (assumption 10): all earnings paid out, so ``CET1_RE[n]`` is flat and the
        pure climate effect is isolated. ``phi < 1`` is a declared hook, see
        :meth:`portfolio.Portfolio.cet1_re`.
    kappa_tax : float
        Tax shield on the provision term. ``0.0`` in baseline (assumption 5: the
        model is pre-tax); ``kappa_tax > 0`` multiplies the provision channel by
        ``1 - kappa_tax``.
    """

    r_star: float = 0.105
    r_star_convention: RStarConvention = "absolute"
    pd_max: float = 0.30
    baseline_scenario: str | None = None
    phi: float = 1.0
    kappa_tax: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.r_star < 1.0:
            raise ValueError(
                f"r_star must be a fraction, not a percentage, got {self.r_star}"
            )
        if self.pd_max > PSI_MONOTONE_MAX:
            raise ValueError(
                f"pd_max={self.pd_max} exceeds {PSI_MONOTONE_MAX}, beyond which Psi "
                "stops being increasing and is no longer invertible."
            )
        if not 0.0 <= self.phi <= 1.0:
            raise ValueError(f"phi must be a fraction in [0, 1], got {self.phi}")
        if not 0.0 <= self.kappa_tax < 1.0:
            raise ValueError(f"kappa_tax must be in [0, 1), got {self.kappa_tax}")


@dataclass(frozen=True)
class RstConfig:
    """Full configuration: frozen constants plus declared sensitivities."""

    regulatory: RegulatoryConstants = RegulatoryConstants()
    sensitivity: SensitivityParams = SensitivityParams()

    def __post_init__(self) -> None:
        lo, hi = self.pd_bounds
        if lo >= hi:
            raise ValueError(f"empty admissible PD domain [{lo}, {hi}]")

    # -- convenience accessors, so callers never reach two levels deep -----------

    @property
    def ell(self) -> float:
        """Loss given default, shared by all buckets under assumption 3."""
        return self.regulatory.ell

    @property
    def r_star(self) -> float:
        """Breach level of the CET1 ratio, as a fraction."""
        return self.sensitivity.r_star

    @property
    def pd_bounds(self) -> tuple[float, float]:
        """Admissible PD domain ``[p_min, p_max]`` on which ``Psi`` is invertible."""
        return self.regulatory.pd_floor, self.sensitivity.pd_max

    @property
    def erosion_coefficient(self) -> float:
        """``12.5 * r_star``, the weight of the RWA channel inside ``Psi``."""
        return self.regulatory.rwa_factor * self.sensitivity.r_star

    @property
    def label(self) -> str:
        """Short human-readable tag, used for figure legends and table columns."""
        return f"R*={self.r_star:.4f} ({self.sensitivity.r_star_convention})"

    # -- variants ---------------------------------------------------------------

    def with_r_star(self, r_star: float, convention: RStarConvention) -> RstConfig:
        """Copy with a different breach level."""
        return replace(
            self,
            sensitivity=replace(
                self.sensitivity, r_star=r_star, r_star_convention=convention
            ),
        )

    def with_baseline_scenario(self, scenario: str) -> RstConfig:
        """Copy with a different reference narrative ``s_0``.

        Useful as a consistency check: the breach set is invariant under this
        change, since ``H[n]`` and ``Erosion[n]`` shift by the same amount.
        """
        return replace(
            self, sensitivity=replace(self.sensitivity, baseline_scenario=scenario)
        )

    def r_star_conventions(
        self, current_ratio: float, absolute: float = 0.105
    ) -> list[RstConfig]:
        """The two breach-level conventions, to be run side by side.

        Parameters
        ----------
        current_ratio : float
            ``R_0``, the bank's CET1 ratio on the reference narrative at date 0.
            Compute it with :func:`breach.cet1_ratio`.
        absolute : float, optional
            Pillar 1 minimum plus combined buffer. Default 0.105.

        Returns
        -------
        list of RstConfig
            ``[absolute convention, relative convention]``, the latter using
            ``R_0 * (1 - 0.03)``, the 300 basis point prudential benchmark.

        Notes
        -----
        The two conventions do not induce the same scenario ranking. Producing that
        comparison is a deliverable, not a robustness check -- see
        :func:`report.compare_r_star_conventions`.
        """
        relative = current_ratio * (1.0 - RELATIVE_DEPLETION)
        return [
            self.with_r_star(absolute, "absolute"),
            self.with_r_star(relative, "relative"),
        ]


#: Level 1 baseline: everything at its specified default.
DEFAULT_CONFIG = RstConfig()
