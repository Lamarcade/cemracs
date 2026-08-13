"""Outputs of the reverse stress test: rankings, tornado, iso-breach curve, R* table.

Four deliverables, in the order a reader needs them:

1. Which scenarios breach, and by how much capital they miss -- :func:`rank_scenarios`.
2. How far a single bucket would have to move on its own -- :func:`plot_critical_pd_tornado`.
3. The full two-bucket breach boundary -- :func:`plot_iso_breach_frontier`.
4. Whether the two ``R_star`` conventions agree on the ranking --
   :func:`compare_r_star_conventions`. They generally do not, and that disagreement
   is a result of the exercise rather than a robustness caveat.

Figure style is reused from :mod:`pd.viz_style` rather than redefined, so the RST
figures sit next to the descriptive PD figures without a visual seam. The categorical
palette holds six series; beyond that :func:`pd.viz_style.series_colors` raises rather
than recycling hues, which is the intended behaviour -- reduce the selection instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:  # avoids a cycle: sweep imports breach, report imports both
    from sweep import ShareSweep

import breach
import regulatory
from config import K_ARGMAX, PSI_MONOTONE_MAX, RstConfig
from pd import viz_style as vs
from portfolio import Portfolio
from scenarios import ScenarioSet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

#: Euros per billion, for axis labels.
BILLION = 1e9


@dataclass(frozen=True)
class RunContext:
    """The settings a figure was produced under, for its context strip.

    Every figure varies one or two things and holds the rest fixed, but which is which
    is not recoverable from the axes alone -- a distance-to-breach chart looks the same
    whatever region, brown share or breach convention produced it. Each plot declares
    what *it* varies and renders the rest as a footer, so no figure can be read out of
    context or mistaken for another run.

    Attributes
    ----------
    regions : tuple of str
        Perimeter of the study. Rendered as a count past three entries.
    baseline_scenario : str
        Reference narrative ``s_0`` defining ``p0``.
    brown_share : float or None
        Fraction of the book in bucket ``H``.
    target_ratio : float or None
        Baseline CET1 ratio the balance sheet is pinned to.
    bucket_rule, aggregation, granularity : str
        How sectors were split and collapsed. See :mod:`scenarios`.
    n_buckets : int or None
        Bucket count actually used.
    pd_bounds : tuple of float or None
        Admissible PD domain.
    """

    regions: tuple[str, ...] = ()
    baseline_scenario: str = ""
    brown_share: float | None = None
    target_ratio: float | None = None
    bucket_rule: str = ""
    aggregation: str = ""
    granularity: str = ""
    n_buckets: int | None = None
    pd_bounds: tuple[float, float] | None = None

    def caption(self, varying: tuple[str, ...] = ()) -> str:
        """One-line summary, marking the fields the figure sweeps as ``varies``.

        Parameters
        ----------
        varying : tuple of str
            Any of ``"regions"``, ``"scenario"``, ``"share"``, ``"target"``,
            ``"bucket"`` -- the caption fields, and only those. What a figure puts on
            its axes otherwise (dates, narratives, buckets) is already legible from the
            axes and the legend; this strip is for what is *not* visible.
        """
        parts: list[str] = []

        def add(key: str, label: str, value: str | None) -> None:
            if key in varying:
                parts.append(f"{label}: varies")
            elif value:
                parts.append(f"{label}: {value}")

        if len(self.regions) > 3:
            where = f"{len(self.regions)} regions"
        else:
            where = ", ".join(self.regions)
        add("regions", "perimeter", where)
        add("scenario", "p0", self.baseline_scenario)
        add("share", "brown share",
            None if self.brown_share is None else f"{self.brown_share:.0%}")
        add("target", "pinned at",
            None if self.target_ratio is None else f"{self.target_ratio:.1%}")

        book = " / ".join(x for x in (self.granularity, self.bucket_rule, self.aggregation) if x)
        if self.n_buckets is not None:
            book = f"{self.n_buckets} ({book})" if book else str(self.n_buckets)
        add("bucket", "buckets", book or None)
        if self.pd_bounds is not None:
            parts.append(f"PD in [{self.pd_bounds[0]:.0e}, {self.pd_bounds[1]:.2f}]")
        # neutral prefix: the strip lists both what is pinned and what the figure
        # sweeps, so "held fixed" would contradict its own "varies" entries
        return "run settings — " + "  |  ".join(parts) if parts else ""


def _context_strip(fig: Figure, context: RunContext | None, varying: tuple[str, ...] = ()) -> None:
    """Render the run settings under a figure. No-op when no context is supplied."""
    if context is None:
        return
    text = context.caption(varying)
    if text:
        fig.text(0.0, -0.015, text, fontsize=8, color=vs.INK_MUTED, ha="left", va="top")


# -- tables --------------------------------------------------------------------


def rank_scenarios(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    date_index: int | None = None,
    check_cushion: bool = True,
) -> pd.DataFrame:
    """Rank scenarios by distance to breach, most severe first.

    Parameters
    ----------
    date_index : int, optional
        Date to rank on. ``None`` (default) ranks on each scenario's *worst* date,
        the first-passage reading of the horizon.

    Returns
    -------
    DataFrame
        Indexed by scenario, columns ``distance_bn`` (billions of euros, negative
        means breached), ``erosion_bn``, ``cushion_bn``, ``worst_year``, ``breached``.
    """
    h = breach.cushion(portfolio, scenarios, cfg, check=check_cushion)
    ero = breach.erosion(portfolio, scenarios, cfg)
    dist = h[None, :] - ero

    if date_index is None:
        worst = dist.argmin(axis=1)
    else:
        worst = np.full(scenarios.n_scenarios, date_index, dtype=int)
    rows = np.arange(scenarios.n_scenarios)

    table = pd.DataFrame(
        {
            "distance_bn": dist[rows, worst] / BILLION,
            "erosion_bn": ero[rows, worst] / BILLION,
            "cushion_bn": h[worst] / BILLION,
            "worst_year": scenarios.dates[worst],
            "breached": dist[rows, worst] < 0.0,
        },
        index=pd.Index(scenarios.scenarios, name="scenario"),
    )
    return table.sort_values("distance_bn")


def compare_r_star_conventions(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    configs: list[RstConfig],
    date_index: int | None = None,
    check_cushion: bool = True,
) -> pd.DataFrame:
    """Side-by-side scenario ranking under several breach-level conventions.

    Parameters
    ----------
    configs : list of RstConfig
        Typically the output of :meth:`config.RstConfig.r_star_conventions`.

    Returns
    -------
    DataFrame
        Indexed by scenario, with ``distance_bn``, ``rank`` and ``breached`` per
        convention, plus two disagreement columns. ``rank_disagreement`` is the spread
        of ranks; ``breach_disagreement`` is True where the conventions do not even
        agree on whether the scenario breaches. The second is usually the sharper of
        the two: the conventions often preserve the ordering while disagreeing
        completely on how many scenarios are in the breach set.
    """
    if not configs:
        raise ValueError("configs is empty")

    distances, breached = {}, {}
    for cfg in configs:
        table = rank_scenarios(portfolio, scenarios, cfg, date_index, check_cushion)
        distances[cfg.label] = table["distance_bn"]
        breached[cfg.label] = table["breached"]

    out = pd.DataFrame(distances)
    ranks = out.rank(method="min", ascending=True).astype(int)
    breach_flags = pd.DataFrame(breached)
    for label in out.columns:
        out[f"rank [{label}]"] = ranks[label]
        out[f"breached [{label}]"] = breach_flags[label]
    out["rank_disagreement"] = ranks.max(axis=1) - ranks.min(axis=1)
    out["breach_disagreement"] = breach_flags.any(axis=1) & ~breach_flags.all(axis=1)
    return out.sort_values(out.columns[0])


def critical_pd_table(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    scenario: str,
    date_index: int,
    check_cushion: bool = True,
) -> pd.DataFrame:
    """Per-bucket critical PD at one (scenario, date), with the limiting-case flags.

    Returns
    -------
    DataFrame
        Indexed by bucket, columns ``baseline_pd``, ``scenario_pd``, ``critical_pd``,
        ``headroom`` (critical minus scenario PD), ``residual_cushion_bn``, ``status``.
        ``status`` is one of ``ok``, ``already_breached``, ``unreachable``,
        ``zero_exposure``.
    """
    result = breach.critical_pd(portfolio, scenarios, cfg, check_cushion=check_cushion)
    s = scenarios.scenario_index(scenario)

    status = np.full(scenarios.n_buckets, "ok", dtype=object)
    status[result.already_breached[s, :, date_index]] = "already_breached"
    status[result.unreachable[s, :, date_index]] = "unreachable"
    status[result.zero_exposure[s, :, date_index]] = "zero_exposure"

    scenario_pd = scenarios.pd_cube[s, :, date_index]
    crit = result.critical_pd[s, :, date_index]
    return pd.DataFrame(
        {
            "baseline_pd": scenarios.baseline_pd[:, date_index],
            "scenario_pd": scenario_pd,
            "critical_pd": crit,
            "headroom": crit - scenario_pd,
            "residual_cushion_bn": result.residual_cushion[s, :, date_index] / BILLION,
            "reach_margin_bn": result.reach_margin[s, :, date_index] / BILLION,
            "status": status,
        },
        index=pd.Index(scenarios.buckets, name="bucket"),
    )


# -- figures -------------------------------------------------------------------


def _mark_domain(ax: plt.Axes, cfg: RstConfig, label: bool = False) -> None:
    """Shade the admissible PD domain and mark the three bounds that define it."""
    lo, hi = cfg.pd_bounds
    ax.axvspan(lo, hi, color=vs.GRID, alpha=0.45, zorder=0)
    for x, text, style in (
        (lo, f"Basel floor {lo:.0e}", ":"),
        (hi, f"p_max {hi:.2f}", "--"),
        (PSI_MONOTONE_MAX, f"hard bound {PSI_MONOTONE_MAX}", "-."),
    ):
        ax.axvline(x, color=vs.INK_MUTED, linewidth=1.1, linestyle=style, zorder=2)
        if label:
            ax.annotate(
                text, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -11), textcoords="offset points",
                color=vs.INK_MUTED, fontsize=8, rotation=90, va="top",
            )


def plot_sector_max_pd(
    df: pd.DataFrame,
    cfg: RstConfig,
    regions: list[str] | None = None,
    n_sectors: int | None = None,
    include_bau: bool = True,
    statistic: str = "max",
    ax: plt.Axes | None = None,
    context: RunContext | None = None,
) -> Figure:
    """PD reached by each sector under each scenario, against the domain bounds.

    One row per sector, one dot per scenario, aggregating ``scenario_pd`` over every
    ``(region, date)`` cell retained. The vertical rules are the candidate values of
    ``p_max``, so the chart answers directly: which sectors does the admissible domain
    censor, and under which narrative.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`, PDs in percentage points.
    cfg : RstConfig
        Supplies the current ``p_max``, drawn as the binding rule.
    regions : list of str, optional
        Restrict to these regions. ``None`` keeps all 53, which makes the ``max``
        statistic a statement about the single worst country in the file. Should match
        the study, otherwise the censoring shown is not the censoring it experiences.
    n_sectors : int, optional
        Keep only the ``n`` highest-ranking sectors. ``None`` (default) shows all 50.
    include_bau : bool, optional
        Add the business-as-usual path as a scenario. Default True.
    statistic : {"max", "median"}, optional
        How to collapse the ``(region, date)`` cells. **Read both.** ``max`` is an
        extreme and is routinely set by one outlier region -- on the six default
        regions every peak comes from Brazil, whose baseline PD runs three times the
        next highest -- so a ``max`` chart says what the worst cell does, not what the
        portfolio does. ``median`` shows the typical level.

    Returns
    -------
    Figure

    Notes
    -----
    Under ``max``, the BAU column is usually flat across sectors: CLIMACRED's
    ``baseline_pd`` does not vary by sector for the elementary regions, so a single
    region-year supplies every sector's peak and the dots line up. That flatness is a
    property of the *argmax cell*, not evidence that the baseline is high everywhere --
    check the per-region spread before concluding anything about censoring from it.
    """
    from scenarios import PERCENT_POINTS, with_bau_scenario

    if statistic not in {"max", "median"}:
        raise ValueError(f"statistic must be 'max' or 'median', got {statistic!r}")

    sel = df if regions is None else df[df["region"].isin(regions)]
    if sel.empty:
        raise ValueError(f"no rows for regions={regions}")
    if include_bau:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # BAU spread already reported by the driver
            sel = with_bau_scenario(sel)

    peak = (
        sel.groupby(["sector", "scenario"])["scenario_pd"].agg(statistic).unstack()
        / PERCENT_POINTS
    )
    peak = peak.loc[peak.max(axis=1).sort_values().index]

    # attribute the extreme: a max chart whose peaks all come from one region is a
    # statement about that region, and the reader has to be told which one
    driver = ""
    if statistic == "max":
        worst = sel.loc[sel["scenario_pd"].idxmax()]
        by_region = sel.groupby("region")["scenario_pd"].max()
        share = float(by_region.max() / by_region.nlargest(2).iloc[-1]) if len(by_region) > 1 else 1.0
        driver = f" — peak set by {worst['region']} {int(worst['year'])}"
        if share > 1.5:
            driver += f", {share:.1f}x the next region"
    if n_sectors is not None:
        peak = peak.tail(n_sectors)

    names = list(peak.columns)
    colours = vs.series_colors(len(names))

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, max(4.0, 0.24 * len(peak) + 2.6)), facecolor=vs.SURFACE
        )
    else:
        fig = ax.figure

    lo, hi = cfg.pd_bounds
    ax.axvspan(0.0, hi, color=vs.GRID, alpha=0.45, zorder=0)
    for x, text, style in (
        (hi, f"p_max = {hi:.2f}", "--"),
        (0.50, "0.50", ":"),
        (PSI_MONOTONE_MAX, f"hard bound {PSI_MONOTONE_MAX}", "-."),
    ):
        if abs(x - hi) > 1e-9 or text.startswith("p_max"):
            ax.axvline(x, color=vs.INK_MUTED, linewidth=1.2, linestyle=style, zorder=2)
            ax.annotate(
                text, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -4), textcoords="offset points",
                color=vs.INK_MUTED, fontsize=8, rotation=90, va="top",
            )

    y = np.arange(len(peak))
    # the min-to-max connector makes 50 rows readable: without it the eye cannot tell
    # which dots belong to the same sector
    ax.hlines(y, peak.min(axis=1), peak.max(axis=1), color=vs.AXIS, linewidth=1.2, zorder=3)

    # BAU last and with its own marker: it is the reference every other dot is read
    # against, and drawing it first buries it under whichever narrative happens to
    # coincide with it -- which is most of them, since many sectors never exceed BAU
    bau_name = "BAU" if "BAU" in names else None
    for colour, name in zip(colours, names):
        if name == bau_name:
            continue
        ax.plot(peak[name].to_numpy(), y, "o", color=colour, markersize=6,
                label=name, zorder=4)

    if bau_name is not None:
        bau = peak[bau_name]
        colour = colours[names.index(bau_name)]
        ax.plot(bau.to_numpy(), y, "D", color=colour, markersize=5,
                markeredgecolor=vs.SURFACE, markeredgewidth=0.6,
                label=bau_name, zorder=6)
        # when the BAU peak is flat across sectors -- CLIMACRED's baseline_pd does not
        # vary by sector for elementary regions -- say so with a rule: every horizontal
        # deviation from it is then climate increment, not baseline level
        if bau.max() - bau.min() < 0.01 * max(bau.max(), 1e-12):
            ax.axvline(float(bau.mean()), color=colour, linewidth=1.3,
                       linestyle="--", alpha=0.7, zorder=5)
            ax.annotate(
                f"BAU peak {bau.mean():.3f}, flat across sectors",
                xy=(float(bau.mean()), 0.0), xycoords=("data", "axes fraction"),
                xytext=(-4, 6), textcoords="offset points",
                color=colour, fontsize=8, rotation=90, ha="right", va="bottom",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(peak.index, fontsize=8)
    ax.set_ylim(-0.8, len(peak) - 0.2)
    vs.apply_style(ax, grid_axis="x")
    ax.set_xlabel(f"{statistic} PD over the retained (region, date) cells (fraction)",
                  color=vs.INK_SECONDARY, fontsize=10)
    # centre left: the admissible band is usually empty of data here, so the legend
    # costs nothing there, whereas above the axes it collides with the subtitle
    vs.legend(ax, loc="center left")

    overall = peak.max(axis=1)
    vs.titre(
        ax,
        f"How high does each sector's PD go? ({statistic})",
        f"{int((overall > hi).sum())} of {len(peak)} sectors above p_max={hi:.2f}, "
        f"{int((overall > PSI_MONOTONE_MAX).sum())} beyond the hard bound{driver}",
    )
    _context_strip(fig, context)
    return fig


def plot_regulatory_functions(
    cfg: RstConfig,
    scenarios: ScenarioSet | None = None,
    n_points: int = 600,
    context: RunContext | None = None,
) -> Figure:
    """The regulatory layer as a function of PD: ``Psi``, ``K``, their derivatives.

    Four panels on a shared logarithmic PD axis -- the domain spans four orders of
    magnitude, so a linear axis would collapse everything below 1% into the origin:

    1. ``Psi`` split into its two channels. The provision channel is linear in ``p``,
       the RWA channel carries all the curvature.
    2. ``K``, with its maximum marked. It is *not* monotone.
    3. ``Psi'`` and ``K'``. This is the monotonicity lemma in one picture: ``K'``
       crosses zero at ``p = 0.2962`` while ``Psi'`` stays strictly positive, because
       the provision slope absorbs it. The sufficient condition
       ``K' > -ell/(12.5 R*)`` is drawn.
    4. The auxiliary functions ``R(p)``, ``MA(p)`` and the 99.9% conditional loss rate.

    Parameters
    ----------
    cfg : RstConfig
        Supplies ``ell``, ``r_star``, ``kappa_tax`` and the domain bounds.
    scenarios : ScenarioSet, optional
        When given, the PDs actually used by the study are drawn as a rug under panel
        1. This is the panel that connects the theory to the data: the NGFS PDs live
        around 5-20%, where ``Psi' ~ 0.6-1.3``, far from the ``Psi' ~ 27`` of the
        regulatory floor. That gap is the conditioning problem Levels 2 and 3 inherit.
    n_points : int, optional
        Resolution of the curves. Default 600.

    Returns
    -------
    Figure
    """
    lo, _ = cfg.pd_bounds
    grid = np.geomspace(lo, PSI_MONOTONE_MAX, n_points)
    ell, r_star = cfg.ell, cfg.r_star

    k = regulatory.capital_charge(grid, ell)
    psi = regulatory.psi_from_config(grid, cfg)
    provision = (1.0 - cfg.sensitivity.kappa_tax) * ell * grid
    rwa_channel = cfg.regulatory.rwa_factor * r_star * k

    colours = vs.series_colors(3)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor=vs.SURFACE)

    # 1. Psi and its two channels
    ax = axes[0, 0]
    ax.plot(grid, psi, color=vs.INK, linewidth=2.4, label=r"$\Psi(p)$", zorder=5)
    ax.plot(grid, provision, color=colours[0], linewidth=1.9,
            label=r"provision channel $\ell p$", zorder=4)
    ax.plot(grid, rwa_channel, color=colours[1], linewidth=1.9,
            label=r"RWA channel $12.5 R^* K(p)$", zorder=4)
    _mark_domain(ax, cfg, label=True)
    if scenarios is not None:
        values = scenarios.pd_cube.ravel()
        ax.plot(values, np.full(values.size, ax.get_ylim()[0]), "|",
                color=colours[2], markersize=7, alpha=0.35, zorder=3,
                label=f"PDs used ({values.size} values)")
    ax.set_xscale("log")
    vs.apply_style(ax, grid_axis="both")
    vs.legend(ax, loc="upper left")
    vs.titre(ax, "Erosion function", "both channels, per unit of exposure")

    # 2. K alone
    ax = axes[0, 1]
    ax.plot(grid, k, color=vs.INK, linewidth=2.4, label=r"$K(p)$", zorder=5)
    k_max = float(regulatory.capital_charge(K_ARGMAX, ell))
    ax.plot([K_ARGMAX], [k_max], "o", color=colours[1], markersize=9, zorder=6)
    ax.annotate(
        f"max {k_max:.4f}\nat p = {K_ARGMAX}",
        xy=(K_ARGMAX, k_max), xytext=(-70, -34), textcoords="offset points",
        color=vs.INK_SECONDARY, fontsize=9,
        arrowprops=dict(arrowstyle="-", color=vs.AXIS, linewidth=1.0),
    )
    _mark_domain(ax, cfg)
    ax.set_xscale("log")
    vs.apply_style(ax, grid_axis="both")
    vs.legend(ax, loc="upper left")
    vs.titre(ax, "IRB capital charge", "unexpected loss only — decreasing past its peak")

    # 3. derivatives: the monotonicity lemma
    ax = axes[1, 0]
    d_psi = regulatory.d_psi_from_config(grid, cfg)
    d_k = regulatory.d_capital_charge(grid, ell)
    sufficient = -ell / (cfg.regulatory.rwa_factor * r_star)
    ax.plot(grid, d_psi, color=vs.INK, linewidth=2.4, label=r"$\Psi'(p)$", zorder=5)
    ax.plot(grid, d_k, color=colours[1], linewidth=1.9, label=r"$K'(p)$", zorder=4)
    ax.axhline(0.0, color=vs.INK_SECONDARY, linewidth=1.2, zorder=3)
    ax.axhline(sufficient, color=colours[0], linewidth=1.3, linestyle="--", zorder=3,
               label=rf"sufficient: $K' > {sufficient:.4f}$")
    _mark_domain(ax, cfg)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    vs.apply_style(ax, grid_axis="both")
    vs.legend(ax, loc="upper right")
    vs.titre(
        ax,
        "Why Psi stays invertible",
        rf"$K'$ turns negative at p={K_ARGMAX}, $\Psi'$ never does",
    )

    # 4. auxiliary functions
    ax = axes[1, 1]
    ax.plot(grid, regulatory.asset_correlation(grid), color=colours[0], linewidth=1.9,
            label=r"$R(p)$ asset correlation")
    ax.plot(grid, regulatory.maturity_adjustment(grid), color=colours[1], linewidth=1.9,
            label=r"$MA(p)$ maturity adjustment")
    ax.plot(grid, regulatory.conditional_loss_rate(grid), color=colours[2], linewidth=1.9,
            label="99.9% conditional loss rate")
    _mark_domain(ax, cfg)
    ax.set_xscale("log")
    vs.apply_style(ax, grid_axis="both")
    vs.legend(ax, loc="upper left")
    vs.titre(ax, "Auxiliary functions", "the ingredients of the IRB formula")

    for ax in axes.ravel():
        ax.set_xlabel("PD (fraction, log scale)", color=vs.INK_SECONDARY, fontsize=10)

    fig.tight_layout(pad=2.2)
    _context_strip(fig, context)
    return fig


def plot_distance_to_breach(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    check_cushion: bool = True,
    ax: plt.Axes | None = None,
    context: RunContext | None = None,
) -> Figure:
    """Distance to breach over the horizon, one line per scenario.

    The zero line is the breach boundary: a scenario is in breach wherever its curve
    goes below it.
    """
    dist = breach.distance_to_breach(portfolio, scenarios, cfg, check_cushion)
    colours = vs.series_colors(scenarios.n_scenarios)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=vs.SURFACE)
    else:
        fig = ax.figure

    ax.axhline(0.0, color=vs.INK, linewidth=1.6, linestyle="--", zorder=3)
    ax.annotate(
        f"breach boundary ({cfg.label})",
        xy=(scenarios.dates[0], 0.0),
        xytext=(2, 6),
        textcoords="offset points",
        color=vs.INK_SECONDARY,
        fontsize=9,
    )
    for colour, name, row in zip(colours, scenarios.scenarios, dist):
        ax.plot(scenarios.dates, row / BILLION, color=colour, linewidth=2.0, label=name, zorder=4)
        ax.plot(scenarios.dates[-1:], row[-1:] / BILLION, "o", color=colour, markersize=8, zorder=5)

    vs.apply_style(ax)
    ax.set_xlabel("")
    ax.set_ylabel("H[n] - Erosion[n]  (bn EUR)", color=vs.INK_SECONDARY, fontsize=10)
    vs.legend(ax, loc="best")
    vs.titre(
        ax,
        "Distance to CET1 breach",
        f"Capital left before Ratio < R*, reference narrative {scenarios.baseline_scenario}",
    )
    _context_strip(fig, context)
    return fig


def plot_critical_pd_tornado(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    scenario: str,
    date_index: int,
    n_buckets: int = 12,
    check_cushion: bool = True,
    ax: plt.Axes | None = None,
    context: RunContext | None = None,
) -> Figure:
    """Tornado of per-bucket headroom: how far each bucket's PD is from its critical level.

    Bars run from the scenario PD to the critical PD. A short bar is a bucket close to
    breaking the ratio on its own; buckets flagged ``unreachable`` cannot break it at
    all and are drawn hatched at the domain cap, because omitting them would hide the
    reason the ratio holds.
    """
    table = critical_pd_table(
        portfolio, scenarios, cfg, scenario, date_index, check_cushion
    )
    # rank on reach_margin, not headroom: headroom is NaN for every unreachable bucket,
    # and on a granular portfolio *every* bucket is unreachable, so a headroom sort
    # would silently return an arbitrary slice instead of the most threatening buckets
    table = table.sort_values("reach_margin_bn", ascending=False).head(n_buckets)[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * len(table) + 2.2)), facecolor=vs.SURFACE)
    else:
        fig = ax.figure

    _, pd_max = cfg.pd_bounds
    y = np.arange(len(table))
    for i, (_, row) in enumerate(table.iterrows()):
        start = row["scenario_pd"]
        if row["status"] == "unreachable":
            ax.barh(i, pd_max - start, left=start, color=vs.INK_MUTED, alpha=0.30,
                    hatch="//", edgecolor=vs.INK_MUTED, zorder=3)
        elif row["status"] == "already_breached":
            ax.barh(i, pd_max - start, left=start, color=vs.CATEGORICAL[1], alpha=0.55, zorder=3)
        else:
            ax.barh(i, row["critical_pd"] - start, left=start,
                    color=vs.CATEGORICAL[0], alpha=0.85, zorder=3)
        ax.plot([start], [i], "o", color=vs.INK, markersize=6, zorder=5)
        # On a granular portfolio every bucket is unreachable and every bar runs to
        # p_max, so the geometry alone carries no information. The capital still
        # missing at maximum PD is what separates them.
        if row["status"] == "unreachable":
            ax.annotate(
                f"{-row['reach_margin_bn']:,.1f} bn short",
                xy=(pd_max, i), xytext=(-6, 0), textcoords="offset points",
                color=vs.INK_SECONDARY, fontsize=8, ha="right", va="center", zorder=6,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(table.index, fontsize=9)
    vs.apply_style(ax, grid_axis="x")
    ax.set_xlabel("PD (fraction): dot = scenario, bar end = critical level",
                  color=vs.INK_SECONDARY, fontsize=10)
    n_unreachable = int((table["status"] == "unreachable").sum())
    subtitle = f"{scenario}, {int(scenarios.dates[date_index])}"
    if n_unreachable == len(table):
        subtitle += (
            f" — every bucket unreachable at p_max={pd_max:.2f}: no single bucket of "
            f"this {portfolio.n_buckets}-bucket book can break the ratio alone"
        )
    else:
        subtitle += f" — hatched = unreachable even at p_max={pd_max:.2f}"
    vs.titre(ax, "How far is each bucket from breaking the ratio alone?", subtitle)
    _context_strip(fig, context)
    return fig


#: Colours of the three-way breach classification, in increasing severity.
BREACH_COLOURS = {"neither": 0, "relative only": 3, "both": 1}


def plot_region_shock(table: pd.DataFrame, n_regions: int = 20, context: RunContext | None = None) -> Figure:
    """Rank regions by the size of the climate shock, coloured by whether they breach.

    The shock is the largest CET1 ratio drop against BAU, in percentage points -- the
    scale that is comparable across regions, unlike a euro distance, which also carries
    the region's risk density.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`regions.scan_regions`.
    n_regions : int, optional
        How many regions to show, worst first. Default 20.

    Returns
    -------
    Figure
    """
    import regions as rg

    ok = table[table["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("no region ran successfully")
    ok["class"] = rg.breach_class(ok)
    shown = ok.nlargest(n_regions, "shock_pp").iloc[::-1]

    palette = vs.series_colors(max(BREACH_COLOURS.values()) + 1)
    fig, ax = plt.subplots(
        figsize=(11, max(4.0, 0.34 * len(shown) + 2.4)), facecolor=vs.SURFACE
    )

    y = np.arange(len(shown))
    for i, row in enumerate(shown.itertuples()):
        colour = palette[BREACH_COLOURS[shown["class"].iloc[i]]]
        ax.barh(i, row.shock_pp, color=colour, alpha=0.9, zorder=3)
        ax.annotate(
            f"{row.worst_scenario} {int(row.worst_year)}",
            xy=(row.shock_pp, i), xytext=(6, 0), textcoords="offset points",
            color=vs.INK_MUTED, fontsize=8, va="center",
        )

    median = float(ok["shock_pp"].median())
    ax.axvline(median, color=vs.INK, linewidth=1.4, linestyle="--", zorder=4)
    ax.annotate(
        f"median of all {len(ok)} regions: {median:.2f} pp",
        xy=(median, 1.0), xycoords=("data", "axes fraction"),
        xytext=(4, -12), textcoords="offset points",
        color=vs.INK_SECONDARY, fontsize=8,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(shown["region"], fontsize=9)
    ax.set_xlim(0, float(shown["shock_pp"].max()) * 1.18)
    vs.apply_style(ax, grid_axis="x")
    ax.set_xlabel("largest CET1 ratio drop vs BAU (percentage points)",
                  color=vs.INK_SECONDARY, fontsize=10)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=palette[BREACH_COLOURS[k]], label=f"breaches: {k}")
        for k in ("both", "relative only", "neither")
    ]
    vs.legend(ax, handles=handles, loc="lower right")
    counts = ok["breach_abs"].sum(), ok["breach_rel"].sum()
    vs.titre(
        ax,
        "Which regions take the biggest hit?",
        f"{int(counts[0])} of {len(ok)} regions breach under the absolute convention, "
        f"{int(counts[1])} under the relative one",
    )
    _context_strip(fig, context, ('regions',))
    return fig


def plot_region_distance(table: pd.DataFrame, n_regions: int = 20, context: RunContext | None = None) -> Figure:
    """Distance to breach per region under both conventions, as a dumbbell.

    Shows how far the two thresholds are apart region by region, and how many regions
    sit between them -- safe on one convention, breached on the other.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`regions.scan_regions`.
    n_regions : int, optional
        How many regions to show, most breached first. Default 20.

    Returns
    -------
    Figure
    """
    ok = table[table["status"] == "ok"]
    if ok.empty:
        raise ValueError("no region ran successfully")
    shown = ok.nsmallest(n_regions, "dist_rel_bn").iloc[::-1]

    colours = vs.series_colors(2)
    fig, ax = plt.subplots(
        figsize=(11, max(4.0, 0.34 * len(shown) + 2.4)), facecolor=vs.SURFACE
    )

    y = np.arange(len(shown))
    ax.axvline(0.0, color=vs.INK, linewidth=1.6, linestyle="--", zorder=4)
    ax.hlines(y, shown["dist_rel_bn"], shown["dist_abs_bn"],
              color=vs.AXIS, linewidth=1.4, zorder=3)
    ax.plot(shown["dist_abs_bn"], y, "o", color=colours[0], markersize=7,
            label="absolute R* = 10.50 %", zorder=5)
    ax.plot(shown["dist_rel_bn"], y, "o", color=colours[1], markersize=7,
            label="relative R* = baseline − 300 bp", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(shown["region"], fontsize=9)
    vs.apply_style(ax, grid_axis="x")
    ax.set_xlabel("distance to breach, worst date (bn EUR) — negative is a breach",
                  color=vs.INK_SECONDARY, fontsize=10)
    # upper right: rows are sorted worst-first, so the top rows sit far left and leave
    # that corner clear, whereas the bottom right is where the safe regions land
    vs.legend(ax, loc="upper right")

    split = int((~ok["breach_abs"].fillna(False) & ok["breach_rel"].fillna(False)).sum())
    vs.titre(
        ax,
        "How much does the breach convention matter?",
        f"{split} of {len(ok)} regions sit between the two thresholds — safe under the "
        "absolute reading, breached under the relative one",
    )
    _context_strip(fig, context, ('regions',))
    return fig


def plot_share_sweep(sweeps: list["ShareSweep"], cfg: RstConfig, context: RunContext | None = None) -> Figure:
    """Distance to breach against the carbon-intensive share, one panel per region set.

    The contrast between panels is the point: how much the split matters depends
    entirely on the geographic perimeter. On EU27 the curves are near flat and DAPS
    *rises*; on a wider basket several narratives cross zero and a critical share
    exists.

    Parameters
    ----------
    sweeps : list of sweep.ShareSweep
        One per region set, drawn side by side on a shared y axis.
    cfg : RstConfig
        Supplies the breach-level label.

    Returns
    -------
    Figure
    """
    import sweep as sw

    if not sweeps:
        raise ValueError("no sweep to plot")

    fig, axes = plt.subplots(
        1, len(sweeps), figsize=(7.0 * len(sweeps), 6.5), squeeze=False,
        sharey=True, facecolor=vs.SURFACE,
    )
    colours = vs.series_colors(len(sweeps[0].scenarios))

    for ax, swept in zip(axes[0], sweeps):
        table = sw.critical_shares(swept)
        ax.axhline(0.0, color=vs.INK, linewidth=1.6, linestyle="--", zorder=3)

        for colour, name, column in zip(colours, swept.scenarios, swept.distance.T):
            ax.plot(swept.shares, column / BILLION, color=colour, linewidth=2.0,
                    label=name, zorder=4)
            crossing = table.loc[name, "critical_share"]
            if np.isfinite(crossing):
                ax.plot([crossing], [0.0], "o", color=colour, markersize=9, zorder=6)
                ax.annotate(
                    f"{crossing:.0%}", xy=(crossing, 0.0), xytext=(0, -16),
                    textcoords="offset points", color=colour, fontsize=9,
                    ha="center", fontweight="bold",
                )

        if swept.inadmissible.any():
            ax.axvspan(swept.shares[swept.inadmissible].min(),
                       swept.shares[swept.inadmissible].max(),
                       color=vs.INK_MUTED, alpha=0.20, zorder=1)

        vs.apply_style(ax, grid_axis="both")
        ax.set_xlabel("share of the book in bucket H (brown)",
                      color=vs.INK_SECONDARY, fontsize=10)
        n_crossing = int(np.isfinite(table["critical_share"]).sum())
        vs.titre(
            ax,
            swept.region_label or "carbon share sweep",
            f"{n_crossing} of {len(table)} narratives cross zero"
            if n_crossing else "no narrative crosses zero on this perimeter",
        )

    axes[0][0].set_ylabel("distance to breach, worst date (bn EUR)",
                          color=vs.INK_SECONDARY, fontsize=10)
    vs.legend(axes[0][-1], loc="best")
    fig.suptitle(
        f"Does the brown/green split decide the outcome?  —  {cfg.label}",
        color=vs.INK, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _context_strip(fig, context, ('share', 'regions'))
    return fig


def plot_share_target_map(grid: pd.DataFrame, cfg: RstConfig, context: RunContext | None = None) -> Figure:
    """Worst-scenario distance over a ``share x target_ratio`` grid, with the zero contour.

    The honest companion to :func:`plot_share_sweep`: the breach frontier runs almost
    horizontally, so capitalisation decides the outcome and the brown/green split only
    modulates it.

    Parameters
    ----------
    grid : DataFrame
        Output of :func:`sweep.sweep_share_and_target`, in euros.
    cfg : RstConfig

    Returns
    -------
    Figure
    """
    values = grid.to_numpy() / BILLION
    shares = grid.columns.to_numpy(dtype=float)
    targets = grid.index.to_numpy(dtype=float)

    # rows at or below R_star are all-NaN by construction: pinning the bank below the
    # threshold it is judged against starts it in breach. They render blank.
    usable = ~np.isnan(values).all(axis=1)
    n_dropped = int((~usable).sum())

    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=vs.SURFACE)

    # Diverging ramp centred on zero: the sign is the whole message, and viz_style's
    # neutral midpoint keeps "no breach margin" from looking like a value. Reversed so
    # that breach reads red and headroom reads blue -- the ramp runs blue-to-red, which
    # would otherwise paint the safe corner in alarm colours.
    limit = float(np.nanmax(np.abs(values)))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    mesh = ax.pcolormesh(shares, targets, values, cmap=vs.DIVERGING.reversed(),
                         norm=norm, shading="nearest")
    contour = ax.contour(shares, targets, values, levels=[0.0],
                         colors=[vs.INK], linewidths=2.4)
    ax.clabel(contour, fmt={0.0: "breach frontier"}, fontsize=9)

    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("distance to breach (bn EUR)", color=vs.INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=vs.INK_MUTED, labelsize=8)

    ax.set_xlabel("share of the book in bucket H (brown)",
                  color=vs.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("baseline CET1 ratio it is pinned to",
                  color=vs.INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=vs.INK_MUTED, labelsize=9, length=0)

    kept = values[usable]
    span_share = float(np.nanmax(np.abs(kept[:, -1] - kept[:, 0])))
    span_target = float(np.nanmax(np.abs(kept[-1, :] - kept[0, :])))
    ratio = span_target / max(span_share, 1e-12)
    # kept on two short lines: a single long subtitle stretches the figure under
    # bbox_inches="tight" and leaves half the canvas empty
    note = (
        f"\n{n_dropped} row(s) blank: pinned at or below R*, the bank starts in breach"
        if n_dropped else ""
    )
    vs.titre(
        ax,
        "Composition or capitalisation?",
        f"{cfg.label} — the target ratio moves the distance {ratio:.0f}x more than the "
        f"brown share ({span_target:.1f} vs {span_share:.1f} bn){note}",
    )
    _context_strip(fig, context, ('share', 'target'))
    return fig


def plot_iso_breach_frontier(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    date_index: int,
    check_cushion: bool = True,
    ax: plt.Axes | None = None,
    context: RunContext | None = None,
) -> Figure:
    """Iso-breach curve in the ``(p_H, p_L)`` plane, with the scenarios plotted on it.

    Everything above and to the right of the curve is a breach. Only defined for a
    two-bucket portfolio.
    """
    p_high, p_low = breach.iso_breach_frontier(
        portfolio, scenarios, cfg, date_index, check_cushion=check_cushion
    )
    finite = np.isfinite(p_low)
    if not finite.any():
        raise ValueError(
            "the iso-breach frontier lies entirely outside the admissible PD domain "
            f"at date {int(scenarios.dates[date_index])}: no point of "
            f"[{cfg.pd_bounds[0]:.1e}, {cfg.pd_bounds[1]:.2f}]^2 sits on the boundary."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 7.0), facecolor=vs.SURFACE)
    else:
        fig = ax.figure

    ax.plot(p_high[finite], p_low[finite], color=vs.INK, linewidth=2.2, zorder=4,
            label="iso-breach frontier")
    ax.fill_between(p_high[finite], p_low[finite], cfg.pd_bounds[1],
                    color=vs.CATEGORICAL[1], alpha=0.15, zorder=1, label="breach region")

    colours = vs.series_colors(scenarios.n_scenarios)
    for colour, name in zip(colours, scenarios.scenarios):
        s = scenarios.scenario_index(name)
        ax.plot(scenarios.pd_cube[s, 0, date_index], scenarios.pd_cube[s, 1, date_index],
                "o", color=colour, markersize=9, zorder=5, label=name)

    vs.apply_style(ax, grid_axis="both")
    ax.set_xlabel(f"PD of bucket {portfolio.buckets[0]}", color=vs.INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"PD of bucket {portfolio.buckets[1]}", color=vs.INK_SECONDARY, fontsize=10)
    vs.legend(ax, loc="upper right")
    # name the date and whether it is the binding one: this is a single-date snapshot,
    # and drawn at the wrong date it appears to contradict the distance-to-breach chart
    binding = breach.binding_cell(portfolio, scenarios, cfg, check_cushion=check_cushion)[1]
    when = (
        "the binding date" if binding == date_index
        else f"NOT the binding date, which is {int(scenarios.dates[binding])}"
    )
    vs.titre(
        ax,
        f"Iso-breach frontier at {int(scenarios.dates[date_index])}",
        f"{cfg.label} — {when}. Decreasing level set of a separable increasing function",
    )
    _context_strip(fig, context)
    return fig
