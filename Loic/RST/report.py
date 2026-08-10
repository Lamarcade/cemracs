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

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import breach
from config import RstConfig
from pd import viz_style as vs
from portfolio import Portfolio
from scenarios import ScenarioSet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

#: Euros per billion, for axis labels.
BILLION = 1e9


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
            "status": status,
        },
        index=pd.Index(scenarios.buckets, name="bucket"),
    )


# -- figures -------------------------------------------------------------------


def plot_distance_to_breach(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    check_cushion: bool = True,
    ax: plt.Axes | None = None,
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
    table = table.reindex(
        table["headroom"].fillna(np.inf).sort_values(ascending=False).index
    ).tail(n_buckets)

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

    ax.set_yticks(y)
    ax.set_yticklabels(table.index, fontsize=9)
    vs.apply_style(ax, grid_axis="x")
    ax.set_xlabel("PD (fraction): dot = scenario, bar end = critical level",
                  color=vs.INK_SECONDARY, fontsize=10)
    vs.titre(
        ax,
        "How far is each bucket from breaking the ratio alone?",
        f"{scenario}, {int(scenarios.dates[date_index])} — hatched = unreachable even at p_max={pd_max:.2f}",
    )
    return fig


def plot_iso_breach_frontier(
    portfolio: Portfolio,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    date_index: int,
    check_cushion: bool = True,
    ax: plt.Axes | None = None,
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
    vs.titre(
        ax,
        "Iso-breach frontier",
        f"{int(scenarios.dates[date_index])}, {cfg.label} — decreasing level set of a separable increasing function",
    )
    return fig
