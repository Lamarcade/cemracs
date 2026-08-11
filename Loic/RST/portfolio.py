"""Bank balance sheet: bucket exposures, retained-earnings capital, other RWA.

This module holds the *calibrated* inputs of the exercise (specification section 6):
``E[g,n]``, ``CET1_0`` and ``RWA_oth``. It knows nothing about the regulatory
formulas and nothing about scenarios.

Assumptions carried by this module (specification section 5):

- **9, static balance sheet.** ``E[g,n] = E[g,0]`` for every ``n``, the EBA
  convention. Exposures do not depend on the scenario, which is what makes the
  cancellation of ``CET1_RE`` exact in the breach inequality. The origination
  control ``q[g,n]`` and the runoff rate ``delta_g`` of the underlying note play no
  role at Level 1 -- they belong to the upstream stochastic control problem.
- **10, flat baseline retained earnings.** ``phi = 1``, all earnings distributed, so
  ``CET1_RE[n] = CET1_0``. This isolates the pure climate effect: any drift in the
  ratio then comes from the PD channel and nothing else. The general recursion is
  implemented for ``phi < 1``, as a declared sensitivity hook.
- **2, EAD equals exposure.** No undrawn commitments, so the credit conversion
  factor plays no role and ``EAD[g,n] = E[g,n]``.

Units: exposures, ``cet1_0`` and ``rwa_oth`` are all in euros. Mixing units here is
the single most likely way to get a plausible-looking but meaningless ratio, so
:meth:`Portfolio.validate` checks orders of magnitude explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import RstConfig

BASE_DIR = Path(__file__).resolve().parent

#: Bucket labels of the stylised two-bucket portfolio of the underlying note.
HIGH_CARBON = "H"
LOW_CARBON = "L"


@dataclass(frozen=True)
class Portfolio:
    """Exposures and capital of the bank, on the annual date grid.

    Attributes
    ----------
    buckets : tuple of str
        Bucket keys, in the row order of ``exposure``. A bucket is a
        ``(sector, region)`` pair at the granularity of the available PD
        projections, rendered as a single composite string.
    dates : ndarray of int
        Years ``t_0 ... t_N``, strictly increasing.
    exposure : ndarray, shape (n_bucket, n_date)
        ``E[g,n]`` in euros. Under assumption 9 every column is identical, but the
        shape stays two-dimensional so that a dynamic balance sheet can be dropped
        in without touching any consumer.
    cet1_0 : float
        CET1 capital at date 0, in euros, before the IFRS 9 provision is deducted.
    rwa_oth : float
        Market and operational risk-weighted assets, in euros. A constant add-on:
        the climate shock does not touch them.
    """

    buckets: tuple[str, ...]
    dates: NDArray[np.int64]
    exposure: NDArray[np.float64]
    cet1_0: float
    rwa_oth: float

    def __post_init__(self) -> None:
        self.validate()

    # -- shape helpers ----------------------------------------------------------

    @property
    def n_buckets(self) -> int:
        """Number of buckets."""
        return len(self.buckets)

    @property
    def n_dates(self) -> int:
        """Number of dates on the annual grid."""
        return len(self.dates)

    @property
    def total_exposure(self) -> NDArray[np.float64]:
        """Total exposure per date, shape ``(n_date,)``."""
        return self.exposure.sum(axis=0)

    def bucket_index(self, bucket: str) -> int:
        """Row index of ``bucket``."""
        try:
            return self.buckets.index(bucket)
        except ValueError:
            raise KeyError(
                f"unknown bucket {bucket!r}; available: {list(self.buckets)}"
            ) from None

    def date_index(self, year: int) -> int:
        """Column index of ``year``."""
        matches = np.nonzero(self.dates == year)[0]
        if matches.size == 0:
            raise KeyError(f"year {year} not on the grid {self.dates.tolist()}")
        return int(matches[0])

    # -- capital ----------------------------------------------------------------

    def cet1_re(self, cfg: RstConfig, margin_rate: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Retained-earnings capital ``CET1_RE[n]``, *before* the provision.

        Under assumption 10 (``phi = 1``) this is flat at ``cet1_0``. The general
        recursion of the note is::

            CET1_RE[n+1] = CET1_RE[n] + (1 - phi) * (r_n(E_n) - c_n(q_n, q_{n-1}))

        with ``c_n = 0`` at Level 1 -- there is no active origination control, so
        there are no adjustment costs -- and ``r_n(E_n) = sum_g s[g,n] * E[g,n]``
        the pre-provision credit margin.

        Parameters
        ----------
        cfg : RstConfig
            Supplies ``phi``.
        margin_rate : ndarray, shape (n_bucket, n_date), optional
            ``s[g,n]``, the net credit margin rate per unit of exposure. Required
            when ``phi < 1``, ignored otherwise.

        Returns
        -------
        ndarray, shape (n_date,)
            Capital in euros, excluding the provision.

        Notes
        -----
        Note that ``CET1_RE[n]`` cancels exactly between the two sides of the breach
        inequality (specification section 3, property 1): it enters only through the
        level of the cushion ``H[n]``, never through the erosion. So ``phi < 1``
        moves the cushion but leaves the erosion function untouched.
        """
        phi = cfg.sensitivity.phi
        if phi == 1.0:
            return np.full(self.n_dates, float(self.cet1_0))

        if margin_rate is None:
            raise ValueError(
                "phi < 1 requires margin_rate s[g,n]: with earnings retained, "
                "CET1_RE is no longer flat and the pre-provision result must be "
                "supplied. Level 1 baseline is phi = 1."
            )
        margin = np.asarray(margin_rate, dtype=float)
        if margin.shape != self.exposure.shape:
            raise ValueError(
                f"margin_rate has shape {margin.shape}, expected {self.exposure.shape}"
            )

        # pre-provision retained result of period n, accumulated forward
        result = (margin * self.exposure).sum(axis=0)
        out = np.empty(self.n_dates, dtype=float)
        out[0] = float(self.cet1_0)
        for n in range(self.n_dates - 1):
            out[n + 1] = out[n] + (1.0 - phi) * result[n]
        return out

    # -- validation -------------------------------------------------------------

    def validate(self) -> None:
        """Check shapes, signs, ordering and units.

        Raises
        ------
        ValueError
            On any structural inconsistency.
        """
        if self.exposure.shape != (self.n_buckets, self.n_dates):
            raise ValueError(
                f"exposure has shape {self.exposure.shape}, expected "
                f"({self.n_buckets}, {self.n_dates})"
            )
        if len(set(self.buckets)) != self.n_buckets:
            raise ValueError("duplicate bucket keys")
        if np.any(np.diff(self.dates) <= 0):
            raise ValueError(f"dates must be strictly increasing, got {self.dates}")
        if np.any(self.exposure < 0):
            raise ValueError("exposures must be non-negative")
        if not np.all(np.isfinite(self.exposure)):
            raise ValueError("exposures contain NaN or infinity")
        if self.cet1_0 <= 0 or self.rwa_oth < 0:
            raise ValueError(
                f"cet1_0 must be positive and rwa_oth non-negative, got "
                f"{self.cet1_0} and {self.rwa_oth}"
            )
        # Unit guard: everything is in euros, so a capital-to-exposure ratio far
        # outside the plausible band almost always means one of the two was entered
        # in millions and the other in units.
        leverage = self.cet1_0 / max(self.total_exposure.max(), 1.0)
        if not 1e-3 < leverage < 1.0:
            raise ValueError(
                f"cet1_0 / total exposure = {leverage:.2e} is implausible; exposures "
                "and capital must both be in euros (specification section 10)."
            )


# -- builders ------------------------------------------------------------------


def static_balance_sheet(
    exposure_0: NDArray[np.float64], n_dates: int
) -> NDArray[np.float64]:
    """Broadcast date-0 exposures over the whole grid (assumption 9).

    Parameters
    ----------
    exposure_0 : ndarray, shape (n_bucket,)
        Exposures at date 0, in euros.
    n_dates : int
        Length of the date grid.

    Returns
    -------
    ndarray, shape (n_bucket, n_date)
        ``E[g,n] = E[g,0]``, the EBA static balance sheet convention.
    """
    return np.repeat(np.asarray(exposure_0, dtype=float)[:, None], n_dates, axis=1)


def stylised_portfolio(
    buckets: tuple[str, ...] | list[str],
    dates: NDArray[np.int64] | list[int],
    weights: NDArray[np.float64] | None = None,
    corporate_book: float = 300e9,
    cet1_0: float = 45e9,
    rwa_oth: float = 60e9,
) -> Portfolio:
    """Stylised static balance sheet over an arbitrary bucket set.

    The generic builder behind :func:`stylised_hl_portfolio`, used when the exercise
    runs at full ``(sector, region)`` granularity rather than on the two carbon
    buckets.

    Parameters
    ----------
    buckets : sequence of str
        Bucket keys, in the row order they will keep. Must match
        ``ScenarioSet.buckets`` exactly.
    dates : array_like of int
        Years of the annual grid.
    weights : ndarray, shape (n_bucket,), optional
        Shares of ``corporate_book`` per bucket, normalised internally. Defaults to
        uniform, which is the honest placeholder when no sector-level book exists --
        and the first assumption a reader should attack.
    corporate_book, cet1_0, rwa_oth : float
        Totals in euros. ``cet1_0`` is usually overridden by
        :func:`breach.calibrate_cet1_for_ratio`.

    Returns
    -------
    Portfolio
    """
    keys = tuple(buckets)
    grid = np.asarray(dates, dtype=np.int64)

    if weights is None:
        share = np.full(len(keys), 1.0 / len(keys))
    else:
        share = np.asarray(weights, dtype=float)
        if share.shape != (len(keys),):
            raise ValueError(f"weights has shape {share.shape}, expected ({len(keys)},)")
        if np.any(share < 0) or not share.sum() > 0:
            raise ValueError("weights must be non-negative and not all zero")
        share = share / share.sum()

    return Portfolio(
        buckets=keys,
        dates=grid,
        exposure=static_balance_sheet(corporate_book * share, grid.size),
        cet1_0=cet1_0,
        rwa_oth=rwa_oth,
    )


def stylised_hl_portfolio(
    dates: NDArray[np.int64] | list[int],
    corporate_book: float = 300e9,
    high_carbon_share: float = 0.60,
    cet1_0: float = 45e9,
    rwa_oth: float = 60e9,
) -> Portfolio:
    """Two-bucket stylised balance sheet, in the shape of the underlying note.

    Order of magnitude of a mid-to-large European bank, chosen so that the exercise
    is admissible but not trivial: the starting CET1 ratio lands around 11-12%, above
    the 10.5% absolute breach level, and the cushion ``H[n]`` is positive but
    breakable by a plausible PD shock.

    Parameters
    ----------
    dates : array_like of int
        Years of the annual grid.
    corporate_book : float, optional
        Total corporate exposure in euros. Default 300 billion.
    high_carbon_share : float, optional
        Fraction of the book in the carbon-intensive bucket ``H``. Default 0.20.
    cet1_0 : float, optional
        CET1 capital at date 0, in euros. Default 45 billion.
    rwa_oth : float, optional
        Market and operational RWA, in euros. Default 60 billion.

    Returns
    -------
    Portfolio
        Buckets ``("H", "L")``, static over the grid.

    Notes
    -----
    These are calibration *placeholders*. They are declared here rather than buried
    in a driver so that swapping in a real balance sheet is a one-line change --
    see :func:`from_csv`. ``cet1_0`` in particular is usually overridden by
    :func:`breach.calibrate_cet1_for_ratio`, because the NGFS projections carry PD
    *levels* an order of magnitude above textbook corporate PDs and a hand-set
    capital figure is then almost always inadmissible.
    """
    if not 0.0 < high_carbon_share < 1.0:
        raise ValueError(f"high_carbon_share must be in (0, 1), got {high_carbon_share}")

    return stylised_portfolio(
        buckets=(HIGH_CARBON, LOW_CARBON),
        dates=dates,
        weights=np.array([high_carbon_share, 1.0 - high_carbon_share]),
        corporate_book=corporate_book,
        cet1_0=cet1_0,
        rwa_oth=rwa_oth,
    )


def from_tidy(df: pd.DataFrame, cet1_0: float, rwa_oth: float) -> Portfolio:
    """Build a portfolio from the tidy schema of specification section 7.6.

    Parameters
    ----------
    df : DataFrame
        Columns ``bucket``, ``date``, ``exposure``. Extra columns (``scenario``,
        ``pd``) are ignored, so the same tidy frame can feed both this and
        :func:`scenarios.from_tidy`. Exposure must not depend on the scenario
        (assumption 9); this is checked.
    cet1_0, rwa_oth : float
        Capital and other RWA, in euros.

    Returns
    -------
    Portfolio
    """
    required = {"bucket", "date", "exposure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tidy frame is missing columns {sorted(missing)}")

    unique = df.groupby(["bucket", "date"])["exposure"].nunique()
    if (unique > 1).any():
        offenders = unique[unique > 1].index.tolist()[:5]
        raise ValueError(
            f"exposure varies across scenarios for {offenders}: assumption 9 requires "
            "a scenario-independent balance sheet."
        )

    wide = (
        df.drop_duplicates(subset=["bucket", "date"])
        .pivot(index="bucket", columns="date", values="exposure")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if wide.isna().to_numpy().any():
        raise ValueError("exposure grid has holes: every bucket needs every date")

    return Portfolio(
        buckets=tuple(wide.index.astype(str)),
        dates=wide.columns.to_numpy(dtype=np.int64),
        exposure=wide.to_numpy(dtype=float),
        cet1_0=cet1_0,
        rwa_oth=rwa_oth,
    )


def from_csv(path: str | Path, cet1_0: float, rwa_oth: float) -> Portfolio:
    """Read a tidy CSV (``bucket, date, exposure``) and build a portfolio.

    The hook for plugging in a real balance sheet in place of
    :func:`stylised_hl_portfolio`.
    """
    return from_tidy(pd.read_csv(path), cet1_0=cet1_0, rwa_oth=rwa_oth)


def check_bucket_alignment(portfolio: Portfolio, scenario_buckets: tuple[str, ...]) -> None:
    """Verify that portfolio and scenario buckets are the same set, in the same order.

    Raises
    ------
    ValueError
        Listing the buckets present on one side only. Misalignment here silently
        pairs the wrong PD with the wrong exposure, which no downstream check would
        catch, so it is raised rather than warned.
    """
    if portfolio.buckets == tuple(scenario_buckets):
        return
    only_portfolio = sorted(set(portfolio.buckets) - set(scenario_buckets))
    only_scenarios = sorted(set(scenario_buckets) - set(portfolio.buckets))
    if only_portfolio or only_scenarios:
        raise ValueError(
            f"bucket mismatch -- only in portfolio: {only_portfolio}, "
            f"only in scenarios: {only_scenarios}"
        )
    raise ValueError(
        f"same buckets in different order: portfolio {list(portfolio.buckets)} vs "
        f"scenarios {list(scenario_buckets)}"
    )
