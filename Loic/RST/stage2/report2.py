"""Figures specific to the Stage 2 extension.

Two, both answering "what did the switch change" rather than "what does Stage 2 say":
the Stage 1 result is the reference every reader already has, so showing Stage 2 alone
would hide the only quantity of interest.

Style and the run-context strip are reused from the parent :mod:`report` and
:mod:`pd.viz_style`, so these sit alongside the Level 1 figures without a seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import report
from config import RstConfig
from pd import viz_style as vs
from scenarios import ScenarioSet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

BILLION = 1e9


def plot_stage_migration(
    migration: pd.DataFrame,
    scenarios: ScenarioSet,
    thresh: float,
    context: report.RunContext | None = None,
) -> Figure:
    """Share of buckets sitting in Stage 2, over time, one line per scenario.

    The mechanism made visible. Because the SICR test is re-run at every date, a bucket
    returns to Stage 1 when its ratio falls back -- so a transient shock shows up as a
    spike rather than a step, which is the whole reason the per-date rule was chosen.

    Parameters
    ----------
    migration : DataFrame
        Output of :func:`stage2.staging.migration_table`, values in ``[0, 1]``.
    """
    colours = vs.series_colors(len(migration.index))
    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor=vs.SURFACE)

    for colour, name in zip(colours, migration.index):
        share = migration.loc[name].to_numpy() * 100
        ax.plot(scenarios.dates, share, color=colour, linewidth=2.0, label=name, zorder=4)
        peak = int(share.argmax())
        if share[peak] > 1.0:
            ax.plot([scenarios.dates[peak]], [share[peak]], "o", color=colour,
                    markersize=8, zorder=5)
            ax.annotate(
                f"{share[peak]:.0f}%", xy=(scenarios.dates[peak], share[peak]),
                xytext=(0, 8), textcoords="offset points", color=colour,
                fontsize=9, fontweight="bold", ha="center",
            )

    vs.apply_style(ax)
    ax.set_ylabel("share of buckets in Stage 2 (%)", color=vs.INK_SECONDARY, fontsize=10)
    ax.set_ylim(bottom=-1.0)
    vs.legend(ax, loc="upper left")
    vs.titre(
        ax,
        "When does the book migrate to lifetime ECL?",
        f"SICR at p / p_baseline > {thresh:g}, re-tested each date so a bucket can cure "
        "back to Stage 1",
    )
    report._context_strip(fig, context)
    return fig


def plot_stage1_vs_stage2(
    distance_stage1: np.ndarray,
    distance_stage2: np.ndarray,
    scenarios: ScenarioSet,
    cfg: RstConfig,
    context: report.RunContext | None = None,
) -> Figure:
    """Distance to breach under both conventions of impairment, on one axis.

    Solid is Stage 1, dashed is Stage 2, one colour per scenario. **The gap between a
    pair of lines is the Stage 2 effect** -- that comparison is the deliverable, so the
    two are drawn together rather than in separate figures.

    Parameters
    ----------
    distance_stage1, distance_stage2 : ndarray, shape (n_scenario, n_date)
        Distances in euros.
    """
    colours = vs.series_colors(scenarios.n_scenarios)
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=vs.SURFACE)

    ax.axhline(0.0, color=vs.INK, linewidth=1.6, linestyle="--", zorder=3)
    ax.annotate(
        f"breach boundary ({cfg.label})", xy=(scenarios.dates[0], 0.0),
        xytext=(2, 6), textcoords="offset points", color=vs.INK_SECONDARY, fontsize=9,
    )

    for colour, name, one, two in zip(
        colours, scenarios.scenarios, distance_stage1, distance_stage2
    ):
        ax.plot(scenarios.dates, one / BILLION, color=colour, linewidth=2.0,
                label=name, zorder=4)
        ax.plot(scenarios.dates, two / BILLION, color=colour, linewidth=2.0,
                linestyle=(0, (4, 2)), zorder=4)
        gap = (one - two) / BILLION
        worst = int(gap.argmax())
        if gap[worst] > 0.2:
            ax.annotate(
                "", xy=(scenarios.dates[worst], two[worst] / BILLION),
                xytext=(scenarios.dates[worst], one[worst] / BILLION),
                arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.2, alpha=0.7),
            )

    handles = [
        plt.Line2D([], [], color=vs.INK_SECONDARY, linewidth=2.0, label="Stage 1"),
        plt.Line2D([], [], color=vs.INK_SECONDARY, linewidth=2.0,
                   linestyle=(0, (4, 2)), label="Stage 2"),
    ]
    first = vs.legend(ax, loc="upper left")
    ax.add_artist(first)
    vs.legend(ax, handles=handles, loc="lower right")

    vs.apply_style(ax)
    ax.set_ylabel("distance to breach (bn EUR)", color=vs.INK_SECONDARY, fontsize=10)
    worst_gap = float((distance_stage1 - distance_stage2).max() / BILLION)
    vs.titre(
        ax,
        "What does the Stage 2 switch cost?",
        f"{cfg.label} — lifetime ECL removes up to {worst_gap:.2f} bn of headroom at the "
        "dates where buckets migrate",
    )
    report._context_strip(fig, context)
    return fig
