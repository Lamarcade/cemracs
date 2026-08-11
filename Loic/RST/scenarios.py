"""Default-probability projections: the ``pbar[s,g,n]`` cube and its calibration checks.

The climate scenario enters the model through exactly one channel, the PD. This
module turns the available projections into a dense ``(scenario, bucket, date)``
array of *fractions*, records what had to be clipped, and runs the two calibration
checks that catch a mis-specified input before it silently produces a plausible
answer.

Two traps in the CLIMACRED source, both handled in :func:`from_climacred`:

1. **The file is in percentage points**, not fractions -- a median PD of 5.8 means
   5.8%. Everything downstream assumes fractions (specification section 10).
2. **A bucket is a ``(sector, region)`` pair**, so the composite key has to be built
   explicitly and kept consistent with the portfolio row order.

Clipping is material, not cosmetic. On the current file, about 2% of values sit
below the 3 basis point regulatory floor, and the share above the credibility bound
runs from about 17% at ``p_max = 0.30`` down to about 2% at the default 0.50, with a
maximum above 100%. That is why :func:`clip_pd` returns a report and warns with
numbers rather than clipping quietly (specification section 7.3).

The censoring is also very unevenly spread across regions: on the six default regions
every clipped extreme comes from Brazil, whose baseline PD peaks near 45% against 3-15%
for the EU, Japan, China and the US. Any statement about how much the domain censors
is a statement about a handful of regions, not about the file as a whole.

Choice of reference narrative ``s_0``. Mathematically a pure normalisation: shifting
``p0`` moves ``H[n]`` and ``Erosion[n]`` by the same amount, so the breach set, the
distance to breach and the critical PDs are all strictly invariant. Not neutral for
interpretation, though, since it decides what "climate erosion" is measured against.

**The default is the NGFS business-as-usual path**, ``p0 = baseline_pd``, exposed as
the pseudo-scenario :data:`BAU_LABEL`. CLIMACRED ships the BAU as a variable in its
own right and defines every scenario as ``scenario_pd = baseline_pd + pd_adjustment``,
so taking the BAU as ``p0`` makes ``Erosion[n]`` *exactly* the effect of the
NGFS-supplied climate increment ``pd_adjustment`` -- which is what the underlying note
means by "l'increment climatique, fourni par la NGFS". Picking one of the four
narratives as ``p0`` instead would measure every other scenario against that
narrative, which is a different and much less natural question.

Beware the terminology collision: CLIMACRED's ``baseline_pd`` is the BAU *level*,
while ``RstConfig.sensitivity.baseline_scenario`` is the reference *narrative* ``s_0``
of the specification. They coincide only because the default of the latter is now the
former. The two other candidates of specification section 7.3 remain available by
passing any scenario name: the bank's own internal planning scenario gives the ICAAP
reading and is the most consistent with how ``CET1_RE`` is built, and a frozen
point-in-time observed PD gives the "shock relative to today" reading.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import regulatory
from config import PSI_MONOTONE_MAX, RstConfig

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "mapping.csv"

#: CLIMACRED stores PDs in percentage points; this divides them into fractions.
PERCENT_POINTS = 100.0

#: A PD cube whose maximum exceeds this is taken to be a unit error rather than data.
#: Values just above 1 do occur -- the source saturates a few series at 100 percentage
#: points and rounds to 100.02 -- and those are for :func:`clip_pd` to deal with, not
#: for the constructor to reject. A forgotten division by 100 lands near 100, far
#: above this threshold, so the two cases separate cleanly.
UNIT_ERROR_THRESHOLD = 2.0

#: Separator of the composite ``(sector, region)`` bucket key.
BUCKET_SEP = " | "

#: Name of the business-as-usual pseudo-scenario built from CLIMACRED ``baseline_pd``.
#: Not one of the four NGFS narratives: it is the counterfactual they are defined
#: against, and the default reference narrative ``s_0``.
BAU_LABEL = "BAU"


def make_bucket_key(sector: str, region: str) -> str:
    """Composite bucket key from a ``(sector, region)`` pair."""
    return f"{sector}{BUCKET_SEP}{region}"


@dataclass(frozen=True)
class ClipReport:
    """What :func:`clip_pd` had to change, and where.

    Attributes
    ----------
    n_total : int
        Number of PD values examined.
    n_below, n_above : int
        Values clipped up to ``p_min`` and down to ``p_max``.
    bounds : tuple of float
        The ``(p_min, p_max)`` applied.
    observed_range : tuple of float
        Min and max of the input, *before* clipping. The interesting number: a
        maximum above 1.0 means the source is not in fractions.
    by_scenario : DataFrame
        One row per scenario, columns ``n_below``, ``n_above``, ``share_clipped``.
    """

    n_total: int
    n_below: int
    n_above: int
    bounds: tuple[float, float]
    observed_range: tuple[float, float]
    by_scenario: pd.DataFrame

    @property
    def share_clipped(self) -> float:
        """Fraction of values that were modified."""
        return (self.n_below + self.n_above) / max(self.n_total, 1)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        lo, hi = self.bounds
        obs_lo, obs_hi = self.observed_range
        return (
            f"clipped {self.n_below + self.n_above}/{self.n_total} PD values "
            f"({self.share_clipped:.1%}) onto [{lo:.1e}, {hi:.2f}]: "
            f"{self.n_below} below the floor, {self.n_above} above the cap. "
            f"Observed input range [{obs_lo:.2e}, {obs_hi:.3f}]."
        )


@dataclass(frozen=True)
class ScenarioSet:
    """PD projections indexed by scenario, bucket and date.

    Attributes
    ----------
    scenarios : tuple of str
        Scenario names, in the first-axis order of ``pd_cube``.
    buckets : tuple of str
        Composite ``(sector, region)`` keys, in the second-axis order. Must match
        ``Portfolio.buckets`` exactly -- see :func:`portfolio.check_bucket_alignment`.
    dates : ndarray of int
        Years, strictly increasing, in the third-axis order.
    pd_cube : ndarray, shape (n_scenario, n_bucket, n_date)
        ``pbar[s,g,n]`` as **fractions**, never percentages.
    baseline_scenario : str
        The reference narrative ``s_0`` defining ``p0[g,n]``.
    """

    scenarios: tuple[str, ...]
    buckets: tuple[str, ...]
    dates: NDArray[np.int64]
    pd_cube: NDArray[np.float64]
    baseline_scenario: str

    def __post_init__(self) -> None:
        self.validate()

    # -- shape helpers ----------------------------------------------------------

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios."""
        return len(self.scenarios)

    @property
    def n_buckets(self) -> int:
        """Number of buckets."""
        return len(self.buckets)

    @property
    def n_dates(self) -> int:
        """Number of dates."""
        return len(self.dates)

    @property
    def baseline_index(self) -> int:
        """First-axis index of the reference narrative."""
        return self.scenarios.index(self.baseline_scenario)

    @property
    def baseline_pd(self) -> NDArray[np.float64]:
        """``p0[g,n]``, shape ``(n_bucket, n_date)``."""
        return self.pd_cube[self.baseline_index]

    def scenario_index(self, scenario: str) -> int:
        """First-axis index of ``scenario``."""
        try:
            return self.scenarios.index(scenario)
        except ValueError:
            raise KeyError(
                f"unknown scenario {scenario!r}; available: {list(self.scenarios)}"
            ) from None

    def date_index(self, year: int) -> int:
        """Third-axis index of ``year``."""
        matches = np.nonzero(self.dates == year)[0]
        if matches.size == 0:
            raise KeyError(f"year {year} not on the grid {self.dates.tolist()}")
        return int(matches[0])

    # -- validation and calibration checks --------------------------------------

    def validate(self) -> None:
        """Check shapes, ordering, and that PDs are fractions."""
        expected = (self.n_scenarios, self.n_buckets, self.n_dates)
        if self.pd_cube.shape != expected:
            raise ValueError(
                f"pd_cube has shape {self.pd_cube.shape}, expected {expected}"
            )
        if self.baseline_scenario not in self.scenarios:
            raise ValueError(
                f"baseline scenario {self.baseline_scenario!r} is not among "
                f"{list(self.scenarios)}"
            )
        if np.any(np.diff(self.dates) <= 0):
            raise ValueError(f"dates must be strictly increasing, got {self.dates}")
        if not np.all(np.isfinite(self.pd_cube)):
            raise ValueError("pd_cube contains NaN or infinity")
        if np.any(self.pd_cube < 0.0):
            raise ValueError("PDs must be non-negative")

        # Unit check, deliberately separate from the domain check. Rejecting anything
        # above 1 here would make the cube unconstructible on the real file -- a few
        # series saturate at 100 percentage points and round to 100.02 -- and would
        # therefore block clip_pd, which exists precisely to censor such values.
        peak = float(self.pd_cube.max())
        if peak > UNIT_ERROR_THRESHOLD:
            raise ValueError(
                f"PD maximum is {peak:.2f}: PDs must be fractions, and a maximum this "
                "far above 1 is the signature of a source expressed in percentage "
                "points -- divide by 100."
            )
        if peak > 1.0:
            warnings.warn(
                f"PD maximum is {peak:.4f}, marginally above 1 -- the source saturates "
                "a few series at 100 percentage points. clip_pd will censor them at "
                "p_max; they are not usable as probabilities as they stand.",
                stacklevel=3,
            )

    def check_present_anchor(self, tol: float = 1e-12) -> None:
        """Verify that every scenario shares the same PD at date 0.

        The present is observed, not projected, so ``p0[g,0]`` must equal ``p[g,0]``
        for every scenario. If it does not, ``H[0]`` is not the capital margin the
        bank actually reports, and the whole horizon is anchored on the wrong level.

        Warns
        -----
        UserWarning
            With the worst offending bucket and the size of the discrepancy. Warns
            rather than raises: some calibrations legitimately start diverging at
            date 0, and the exercise still runs -- but the reader must be told.
        """
        gap = np.abs(self.pd_cube[:, :, 0] - self.baseline_pd[:, 0][None, :])
        worst = float(gap.max())
        if worst > tol:
            s, g = np.unravel_index(int(gap.argmax()), gap.shape)
            warnings.warn(
                f"scenarios disagree at date {int(self.dates[0])}: max |p - p0| = "
                f"{worst:.3e} on scenario {self.scenarios[s]!r}, bucket "
                f"{self.buckets[g]!r}. The present is observed, not projected, so "
                f"H[0] is not the reported capital margin.",
                stacklevel=2,
            )

    def with_baseline(self, scenario: str) -> ScenarioSet:
        """Copy with a different reference narrative.

        The breach set is invariant under this change (specification section 3,
        property 2), which makes it the cheapest available consistency check.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"unknown scenario {scenario!r}")
        return ScenarioSet(
            scenarios=self.scenarios,
            buckets=self.buckets,
            dates=self.dates,
            pd_cube=self.pd_cube,
            baseline_scenario=scenario,
        )


# -- clipping ------------------------------------------------------------------


def clip_pd(scenarios: ScenarioSet, cfg: RstConfig, warn: bool = True) -> tuple[ScenarioSet, ClipReport]:
    """Clip the PD cube onto the admissible domain, and report what moved.

    Parameters
    ----------
    scenarios : ScenarioSet
        Input projections, as fractions.
    cfg : RstConfig
        Supplies ``pd_bounds``.
    warn : bool, optional
        Emit a ``UserWarning`` when anything was clipped. Default True.

    Returns
    -------
    ScenarioSet
        Clipped copy.
    ClipReport
        Counts, overall and per scenario.

    Notes
    -----
    Never clips silently. The lower bound is the Basel PD floor, which also keeps the
    domain 102x away from the pole of the maturity adjustment; the upper bound is an
    economic credibility bound. Both mean the clipped values are *censored*, and a
    scenario whose extreme PDs were flattened onto the cap will look less severe than
    it is -- which is exactly what the caller needs to know.
    """
    lo, hi = cfg.pd_bounds
    cube = scenarios.pd_cube
    below = cube < lo
    above = cube > hi

    per_scenario = pd.DataFrame(
        {
            "n_below": below.sum(axis=(1, 2)),
            "n_above": above.sum(axis=(1, 2)),
        },
        index=pd.Index(scenarios.scenarios, name="scenario"),
    )
    per_cell = scenarios.n_buckets * scenarios.n_dates
    per_scenario["share_clipped"] = (
        per_scenario["n_below"] + per_scenario["n_above"]
    ) / per_cell

    report = ClipReport(
        n_total=cube.size,
        n_below=int(below.sum()),
        n_above=int(above.sum()),
        bounds=(lo, hi),
        observed_range=(float(cube.min()), float(cube.max())),
        by_scenario=per_scenario,
    )

    if warn and (report.n_below or report.n_above):
        warnings.warn(f"PD clipping: {report.summary()}", stacklevel=2)

    clipped = ScenarioSet(
        scenarios=scenarios.scenarios,
        buckets=scenarios.buckets,
        dates=scenarios.dates,
        pd_cube=np.clip(cube, lo, hi),
        baseline_scenario=scenarios.baseline_scenario,
    )
    return clipped, report


# -- sector selection ----------------------------------------------------------

#: Ranking criteria understood by :func:`select_sectors`.
SECTOR_CRITERIA = ("erosion", "amplitude", "dispersion")


@dataclass(frozen=True)
class SectorSelection:
    """Sectors retained for a restricted study, and the scores that chose them.

    Attributes
    ----------
    sectors : tuple of str
        The retained sectors, best first.
    scores : DataFrame
        One row per sector of the source, columns ``erosion``, ``amplitude``,
        ``dispersion`` and their ranks. Kept whole rather than filtered so the
        rejected sectors stay inspectable -- the selection is an assumption, and an
        assumption you cannot see the alternatives to is not reviewable.
    criterion : str
        Which column drove the ranking.
    n_saturated : int
        Number of *retained* sectors whose ``erosion`` score hit
        :attr:`saturation_value`.
    saturation_value : float
        ``Psi(p_max) - Psi(p_min)``, the largest erosion increment the admissible
        domain can express.
    """

    sectors: tuple[str, ...]
    scores: pd.DataFrame
    criterion: str
    n_saturated: int
    saturation_value: float

    @property
    def selected_scores(self) -> pd.DataFrame:
        """``scores`` restricted to the retained sectors, in selection order."""
        return self.scores.loc[list(self.sectors)]


def select_sectors(
    df: pd.DataFrame,
    cfg: RstConfig,
    n: int = 12,
    criterion: str = "erosion",
    regions: list[str] | None = None,
    warn: bool = True,
) -> SectorSelection:
    """Rank sectors by how much their PD moves, and keep the top ``n``.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`.
    cfg : RstConfig
        Supplies ``ell``, ``r_star`` and ``pd_bounds`` for the ``erosion`` score.
    n : int, optional
        How many sectors to keep. Default 12.
    criterion : {"erosion", "amplitude", "dispersion"}, optional
        ``erosion`` (default) ranks by ``|Psi(scenario_pd) - Psi(baseline_pd)|``, the
        quantity that actually drives the breach; ``amplitude`` ranks by
        ``|pd_adjustment|``, the literal size of the PD move; ``dispersion`` ranks by
        the spread across narratives, which picks the sectors that *discriminate*
        scenarios rather than the ones that move most.
    regions : list of str, optional
        Restrict the scoring to these regions. Should be the same set the study will
        run on, otherwise sectors are chosen on evidence the study never sees.
    warn : bool, optional
        Report saturation of the ``erosion`` score. Default True.

    Returns
    -------
    SectorSelection

    Warns
    -----
    UserWarning
        When retained sectors saturate the ``erosion`` score, i.e. their PD sweeps the
        whole admissible domain and the clipping bounds -- not the data -- decide their
        score. Their relative order is then meaningless and is settled by ``amplitude``.

    Notes
    -----
    Two properties worth knowing before reading any restricted result.

    **Scores are peaks over the horizon, never last-year values.** The DAPS shock is
    transient (2026-2027) and largely resorbed by 2030, so ranking on the final year
    would select sectors that are intact by then. Same reasoning as
    :func:`pd.climacred_loader.top_sectors`, generalised here to pool over all
    narratives and regions instead of one ``(scenario, region)`` pair.

    **A restricted study is a concentration study.** The stylised book is redistributed
    over the retained sectors alone, so it models a bank concentrated in them, not the
    corresponding sub-portfolio of a wider book. Distances to breach are therefore not
    comparable across different sector universes -- the same trap as the ``p0``
    dependence documented in :func:`breach.calibrate_cet1_for_ratio`.
    """
    if criterion not in SECTOR_CRITERIA:
        raise ValueError(f"criterion must be one of {SECTOR_CRITERIA}, got {criterion!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    sel = df if regions is None else df[df["region"].isin(regions)]
    if sel.empty:
        raise ValueError(f"no rows for regions={regions}")

    lo, hi = cfg.pd_bounds
    work = sel.loc[:, ["scenario", "region", "sector", "year", "baseline_pd",
                       "pd_adjustment", "scenario_pd"]].copy()

    # erosion: the capital-relevant move. Psi is only defined on the admissible
    # domain, so the inputs are clipped first -- which is exactly what censors this
    # score at the top, see the saturation warning below.
    p_scenario = np.clip(work["scenario_pd"].to_numpy() / PERCENT_POINTS, lo, hi)
    p_baseline = np.clip(work["baseline_pd"].to_numpy() / PERCENT_POINTS, lo, hi)
    work["erosion"] = np.abs(
        regulatory.psi_from_config(p_scenario, cfg)
        - regulatory.psi_from_config(p_baseline, cfg)
    )
    work["amplitude"] = work["pd_adjustment"].abs()

    scores = work.groupby("sector")[["erosion", "amplitude"]].max()

    # dispersion: how far apart the narratives are at a given (region, date)
    by_cell = work.groupby(["sector", "region", "year"])["scenario_pd"]
    scores["dispersion"] = (by_cell.max() - by_cell.min()).groupby("sector").max()

    for column in SECTOR_CRITERIA:
        scores[f"rank_{column}"] = scores[column].rank(ascending=False).astype(int)

    # amplitude breaks ties: the erosion score saturates, and without a second key the
    # order among saturated sectors would be whatever the sort happened to produce
    ordered = scores.sort_values([criterion, "amplitude"], ascending=False)
    retained = tuple(ordered.index[:n])

    saturation = float(
        regulatory.psi_from_config(hi, cfg) - regulatory.psi_from_config(lo, cfg)
    )
    saturated = [s for s in retained if scores.loc[s, "erosion"] >= saturation - 1e-9]

    if warn and saturated and criterion == "erosion":
        warnings.warn(
            f"{len(saturated)} of the {len(retained)} retained sectors saturate the "
            f"erosion score at {saturation:.4f} = Psi({hi:.2f}) - Psi({lo:.1e}): their "
            f"PD sweeps the whole admissible domain, so the clipping bounds rather than "
            f"the data set their score, and their relative order is settled by "
            f"amplitude alone. Sectors: {saturated}. p_max is an economic credibility "
            f"bound, not a mathematical one -- the hard bound is {PSI_MONOTONE_MAX} -- "
            f"so raising it (e.g. --pd-max 0.5) loosens the censoring.",
            stacklevel=2,
        )

    return SectorSelection(
        sectors=retained,
        scores=ordered,
        criterion=criterion,
        n_saturated=len(saturated),
        saturation_value=saturation,
    )


# -- bucket mapping ------------------------------------------------------------


def load_mapping(path: str | Path = MAPPING_PATH) -> pd.DataFrame:
    """Read the versioned sector -> carbon bucket mapping artefact.

    Returns
    -------
    DataFrame
        Columns ``ngfs_sector``, ``sector_group``, ``carbon_bucket``, ``nace_codes``.
    """
    mapping = pd.read_csv(path, comment="#")
    expected = {"ngfs_sector", "sector_group", "carbon_bucket", "nace_codes"}
    missing = expected - set(mapping.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    unknown = set(mapping["carbon_bucket"]) - {"H", "L"}
    if unknown:
        raise ValueError(f"carbon_bucket must be H or L, found {sorted(unknown)}")
    return mapping


def check_mapping_coverage(df: pd.DataFrame, path: str | Path = MAPPING_PATH) -> None:
    """Verify that ``mapping.csv`` covers exactly the sectors of the source file.

    Mirrors :func:`pd.climacred_loader.check_sector_groups`, and additionally checks
    that ``sector_group`` agrees with ``climacred_loader.SECTOR_GROUPS`` -- the two
    artefacts must not drift apart.

    Raises
    ------
    ValueError
        Listing the sectors present on one side only, or the group disagreements.
    """
    from pd.climacred_loader import SECTOR_GROUPS

    mapping = load_mapping(path)
    in_file = set(df["sector"].unique())
    in_mapping = set(mapping["ngfs_sector"])
    missing = sorted(in_file - in_mapping)
    extra = sorted(in_mapping - in_file)
    if missing or extra:
        raise ValueError(
            f"{path} out of sync with the source -- missing: {missing}, extra: {extra}"
        )

    disagreements = [
        (row.ngfs_sector, row.sector_group, SECTOR_GROUPS[row.ngfs_sector])
        for row in mapping.itertuples()
        if SECTOR_GROUPS[row.ngfs_sector] != row.sector_group
    ]
    if disagreements:
        raise ValueError(
            f"sector_group disagrees with climacred_loader.SECTOR_GROUPS for "
            f"{disagreements}"
        )


# -- builders ------------------------------------------------------------------


def warn_dropped_scenarios(
    source: pd.DataFrame,
    filtered: pd.DataFrame,
    regions: list[str] | None = None,
    sectors: list[str] | None = None,
) -> tuple[str, ...]:
    """Report narratives that the region/sector filter removed entirely.

    Filtering by region can silently delete a whole narrative, because CLIMACRED does
    not cover every region under every scenario. The cube that comes out is still
    dense and every downstream check still passes, so nothing else in the chain would
    notice -- the reverse stress test would simply be run against a smaller set of
    narratives than the caller believes.

    The live case: **DAPS covers 52 of the 53 regions, all but ``World``**, because it
    is assembled from six geographic slices and none of them spans the world aggregate.
    Selecting ``World`` alone therefore drops the scenario that drives the breach in
    every run so far.

    Returns
    -------
    tuple of str
        The dropped scenario names, empty when nothing was lost.
    """
    before = set(source["scenario"].unique())
    after = set(filtered["scenario"].unique())
    dropped = tuple(sorted(before - after))
    if dropped:
        warnings.warn(
            f"the selection dropped {len(dropped)} narrative(s) entirely: {list(dropped)}. "
            f"regions={regions}, sectors={'all' if sectors is None else len(sectors)}. "
            "CLIMACRED does not cover every region under every scenario -- DAPS in "
            "particular has no 'World' rows -- so the reverse stress test will compare "
            "fewer narratives than the file contains.",
            stacklevel=3,
        )
    return dropped


def warn_duplicate_scenarios(scenarios: ScenarioSet, tol: float = 1e-12) -> list[tuple[str, str]]:
    """Report narratives that are numerically identical over the whole selection.

    The companion trap to :func:`warn_dropped_scenarios`. A narrative can survive the
    region filter and still carry no information, because CLIMACRED gives some regions
    the same projection under two scenarios: **DIRE and HWTP are identical on 39 of the
    53 regions**, including every elementary European one. A study run on France would
    silently rank four narratives of which two are the same series.

    Aggregates escape it -- ``EU27`` mixes in countries where the two diverge -- which
    is one reason the driver defaults to an aggregate rather than a country.

    Returns
    -------
    list of tuple
        The colliding pairs, empty when every narrative is distinct.
    """
    collisions = []
    for i in range(scenarios.n_scenarios):
        for j in range(i + 1, scenarios.n_scenarios):
            gap = float(np.abs(scenarios.pd_cube[i] - scenarios.pd_cube[j]).max())
            if gap <= tol:
                collisions.append((scenarios.scenarios[i], scenarios.scenarios[j]))
    if collisions:
        pairs = ", ".join(f"{a} == {b}" for a, b in collisions)
        warnings.warn(
            f"{len(collisions)} narrative pair(s) are identical over this selection: "
            f"{pairs}. They will receive the same erosion and the same rank, which is a "
            "property of the source file on these regions, not a result. Widen the "
            "region set -- aggregates such as EU27 keep them apart.",
            stacklevel=3,
        )
    return collisions


def with_bau_scenario(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """Append the business-as-usual path to the CLIMACRED long table as a scenario.

    CLIMACRED carries the BAU as its own variable and builds every narrative on top of
    it, ``scenario_pd = baseline_pd + pd_adjustment``. Materialising it as a row with
    ``scenario == BAU_LABEL`` lets it be used as the reference narrative ``p0`` through
    the ordinary machinery, with no special case anywhere downstream.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`.
    warn : bool, optional
        Report the disagreement described below. Default True.

    Returns
    -------
    DataFrame
        ``df`` plus one BAU row per ``(region, sector, year)``.

    Warns
    -----
    UserWarning
        The BAU is **not** strictly scenario-invariant. It is identical across
        narratives for the 47 elementary regions, but differs for the six aggregates
        (``climacred_loader.AGGREGATE_REGIONS``), whose BAU is a weighted mix of
        member-country BAUs and therefore inherits each narrative's slightly different
        sectoral weights and country coverage -- the spread reaches several percentage
        points. The BAU row is the mean across narratives, and the warning reports how
        far apart they were, because on an aggregate region that spread is a
        calibration artefact sitting directly inside ``H[n]``.
    """
    key = ["region", "sector", "year"]
    grouped = df.groupby(key)["baseline_pd"]
    bau = grouped.mean().reset_index()

    if warn:
        spread = (grouped.max() - grouped.min()).reset_index(name="spread")
        worst = spread.loc[spread["spread"].idxmax()]
        if worst["spread"] > 0:
            n_keys = int((spread["spread"] > 0).sum())
            affected = sorted(spread.loc[spread["spread"] > 0, "region"].unique())
            warnings.warn(
                f"the BAU is not scenario-invariant on {n_keys}/{len(spread)} keys, "
                f"all in aggregate regions {affected}: worst spread "
                f"{worst['spread']:.3f} pp on {worst['region']!r}/{worst['sector']!r} "
                f"in {int(worst['year'])}. The BAU row uses the mean across narratives.",
                stacklevel=2,
            )

    bau["scenario"] = BAU_LABEL
    bau["daps_zone"] = None
    bau["pd_adjustment"] = 0.0
    bau["scenario_pd"] = bau["baseline_pd"]
    return pd.concat([df, bau[df.columns]], ignore_index=True)


def from_tidy(df: pd.DataFrame, baseline_scenario: str) -> ScenarioSet:
    """Build a cube from the tidy schema of specification section 7.6.

    Parameters
    ----------
    df : DataFrame
        Columns ``scenario``, ``bucket``, ``date``, ``pd``, the last as a fraction.
    baseline_scenario : str
        Reference narrative ``s_0``.

    Returns
    -------
    ScenarioSet
    """
    required = {"scenario", "bucket", "date", "pd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tidy frame is missing columns {sorted(missing)}")

    scenarios = tuple(sorted(df["scenario"].unique()))
    buckets = tuple(sorted(df["bucket"].unique()))
    dates = np.sort(df["date"].unique()).astype(np.int64)

    index = pd.MultiIndex.from_product(
        [scenarios, buckets, dates], names=["scenario", "bucket", "date"]
    )
    aligned = (
        df.set_index(["scenario", "bucket", "date"])["pd"]
        .groupby(level=[0, 1, 2])
        .mean()
        .reindex(index)
    )
    if aligned.isna().any():
        holes = int(aligned.isna().sum())
        raise ValueError(
            f"{holes} missing (scenario, bucket, date) combinations: the cube must be "
            "dense before it can be evaluated by direct substitution."
        )

    cube = aligned.to_numpy(dtype=float).reshape(
        len(scenarios), len(buckets), len(dates)
    )
    built = ScenarioSet(
        scenarios=scenarios,
        buckets=buckets,
        dates=dates,
        pd_cube=cube,
        baseline_scenario=baseline_scenario,
    )
    warn_duplicate_scenarios(built)
    return built


def from_climacred(
    df: pd.DataFrame,
    baseline_scenario: str = BAU_LABEL,
    regions: list[str] | None = None,
    sectors: list[str] | None = None,
    include_bau: bool = True,
) -> ScenarioSet:
    """Build a cube from the CLIMACRED long table, at ``(sector, region)`` granularity.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`, with ``scenario_pd`` **in
        percentage points**.
    baseline_scenario : str, optional
        Reference narrative ``s_0``. Defaults to the business-as-usual path
        :data:`BAU_LABEL`, so that the erosion is exactly the NGFS climate increment.
    regions, sectors : list of str, optional
        Restrict the cube. ``None`` keeps everything, which is 50 x 53 = 2650 buckets.
    include_bau : bool, optional
        Append the BAU pseudo-scenario, see :func:`with_bau_scenario`. Default True.

    Returns
    -------
    ScenarioSet
        PDs converted to fractions, buckets keyed by :func:`make_bucket_key`.

    Notes
    -----
    Not every scenario covers every region -- DAPS does not cover ``World``, for
    instance -- so restricting to a region set that all scenarios share is usually
    necessary to obtain a dense cube.
    """
    source = with_bau_scenario(df) if include_bau else df
    sel = source
    if regions is not None:
        sel = sel[sel["region"].isin(regions)]
    if sectors is not None:
        sel = sel[sel["sector"].isin(sectors)]
    if sel.empty:
        raise ValueError(
            f"no rows for regions={regions} and sectors={sectors}; check the labels "
            "with pd.climacred_loader.resolve_region."
        )
    warn_dropped_scenarios(source, sel, regions, sectors)

    tidy = pd.DataFrame(
        {
            "scenario": sel["scenario"].to_numpy(),
            "bucket": [
                make_bucket_key(s, r)
                for s, r in zip(sel["sector"].to_numpy(), sel["region"].to_numpy())
            ],
            "date": sel["year"].to_numpy(),
            "pd": sel["scenario_pd"].to_numpy() / PERCENT_POINTS,
        }
    )
    return from_tidy(tidy, baseline_scenario=baseline_scenario)


def aggregate_to_carbon_buckets(
    df: pd.DataFrame,
    baseline_scenario: str = BAU_LABEL,
    sector_weights: pd.Series | None = None,
    regions: list[str] | None = None,
    sectors: list[str] | None = None,
    mapping_path: str | Path = MAPPING_PATH,
    include_bau: bool = True,
) -> ScenarioSet:
    """Collapse the CLIMACRED sectors onto the two carbon buckets ``H`` and ``L``.

    The aggregation is exposure-weighted: the PD of a carbon bucket is the average of
    its sector PDs weighted by the bank's exposure to each sector. Passing uniform
    weights is the honest default when no sector-level book is available, and it is
    the assumption a reader will want to attack first.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`.
    baseline_scenario : str, optional
        Reference narrative ``s_0``. Defaults to the business-as-usual path
        :data:`BAU_LABEL`.
    sector_weights : Series, optional
        Indexed by NGFS sector name, values proportional to exposure. Defaults to
        uniform weights within each carbon bucket.
    regions : list of str, optional
        Regions to pool, also uniformly. ``None`` keeps all of them.
    sectors : list of str, optional
        Restrict to these sectors before collapsing, typically
        ``select_sectors(...).sectors``. Both carbon buckets are then built from the
        retained sectors only, which sharpens the H/L contrast -- and makes the result
        a concentration study, not comparable to the full-universe run.
    mapping_path : path, optional
        The versioned mapping artefact.
    include_bau : bool, optional
        Append the BAU pseudo-scenario, see :func:`with_bau_scenario`. Default True.

    Returns
    -------
    ScenarioSet
        Buckets ``("H", "L")``, PDs as fractions.
    """
    mapping = load_mapping(mapping_path)
    bucket_of = dict(zip(mapping["ngfs_sector"], mapping["carbon_bucket"]))

    source = with_bau_scenario(df) if include_bau else df
    sel = source
    if regions is not None:
        sel = sel[sel["region"].isin(regions)]
    if sectors is not None:
        sel = sel[sel["sector"].isin(sectors)]
    if sel.empty:
        raise ValueError(f"no rows for regions={regions} and sectors={sectors}")
    warn_dropped_scenarios(source, sel, regions, sectors)

    present = set(sel["sector"]).intersection(
        load_mapping(mapping_path).query("carbon_bucket == 'H'")["ngfs_sector"]
    )
    if not present or len(set(sel["sector"])) == len(present):
        raise ValueError(
            "the retained sectors all fall in the same carbon bucket, so the H/L "
            "collapse would produce a single bucket. Use --granularity sector, or "
            "widen the selection."
        )

    work = sel.loc[:, ["scenario", "sector", "year", "scenario_pd"]].copy()
    work["bucket"] = work["sector"].map(bucket_of)
    if work["bucket"].isna().any():
        unmapped = sorted(work.loc[work["bucket"].isna(), "sector"].unique())
        raise ValueError(f"sectors absent from {mapping_path}: {unmapped}")

    if sector_weights is None:
        work["weight"] = 1.0
    else:
        work["weight"] = work["sector"].map(sector_weights)
        if work["weight"].isna().any():
            unweighted = sorted(work.loc[work["weight"].isna(), "sector"].unique())
            raise ValueError(f"sector_weights has no entry for {unweighted}")

    work["weighted_pd"] = work["scenario_pd"] * work["weight"]
    grouped = work.groupby(["scenario", "bucket", "year"], as_index=False).agg(
        weighted_pd=("weighted_pd", "sum"), weight=("weight", "sum")
    )
    tidy = pd.DataFrame(
        {
            "scenario": grouped["scenario"],
            "bucket": grouped["bucket"],
            "date": grouped["year"],
            "pd": grouped["weighted_pd"] / grouped["weight"] / PERCENT_POINTS,
        }
    )
    return from_tidy(tidy, baseline_scenario=baseline_scenario)
