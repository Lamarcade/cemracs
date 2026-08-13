"""Run the reverse stress test region by region and rank the results.

Every other driver studies one perimeter. This one sweeps the geography: it runs the
whole chain once per CLIMACRED region and returns a table of how hard each is hit and
whether it breaches. Computation only; the figures live in :mod:`report`.

What the table is, and is not. Each region is calibrated **independently** to the same
starting CET1 ratio, so a row describes *a hypothetical bank concentrated in that
region*, not that country's actual banking system. Comparing rows compares climate
exposure at equal capitalisation, which is the point -- but no row is a statement about
a real balance sheet.

Two measures are reported because they rank differently:

- ``shock_pp`` -- the largest CET1 ratio drop against BAU, in percentage points. The
  natural scale for "how hard is this region hit", and comparable across regions.
- ``dist_*_bn`` -- the distance to breach in euros, which also scales with the region's
  risk density. A region with high baseline PDs carries more RWA, so the same ratio drop
  costs more euros there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import breach
import portfolio as pf
import scenarios as sc
from config import RstConfig

#: Columns of the scan table, in report order.
COLUMNS = [
    "region", "shock_pp", "worst_scenario", "worst_year",
    "dist_abs_bn", "breach_abs", "dist_rel_bn", "breach_rel",
    "bau_min_ratio_pct", "n_narratives", "status",
]


def scan_regions(
    df: pd.DataFrame,
    cfg: RstConfig,
    regions: list[str] | None = None,
    target_ratio: float = 0.13,
    high_carbon_share: float | None = None,
    absolute_r_star: float = 0.105,
) -> pd.DataFrame:
    """Run the chain once per region and collect the outcome.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`pd.climacred_loader.load_pd`.
    cfg : RstConfig
        Base configuration; its ``r_star`` is replaced per convention below.
    regions : list of str, optional
        Regions to scan. ``None`` (default) scans all of them.
    target_ratio : float, optional
        Starting CET1 ratio every region is pinned to. Default 0.13.
    high_carbon_share : float, optional
        Brown share of the book. ``None`` uses the
        :func:`portfolio.stylised_hl_portfolio` default.
    absolute_r_star : float, optional
        Pillar 1 plus combined buffer. Default 0.105. The relative convention is derived
        per region from its own baseline ratio.

    Returns
    -------
    DataFrame
        One row per region, columns :data:`COLUMNS`. Regions that could not be run carry
        ``status`` set to the exception type and ``NaN`` elsewhere -- a region can lack
        coverage, or put every retained sector in one carbon bucket.

    Notes
    -----
    Warnings are suppressed inside the loop. Every one of them -- the non-invariant BAU,
    narratives dropped by the region filter, DIRE and HWTP colliding on elementary
    European regions, sector PDs clipped before the collapse -- fires once per region
    and would bury the result. They still apply, and a single-region run through
    ``run_rst.py`` surfaces them.
    """
    names = sorted(df["region"].unique()) if regions is None else list(regions)
    abs_cfg = cfg.with_r_star(absolute_r_star, "absolute")
    share = {} if high_carbon_share is None else {"high_carbon_share": high_carbon_share}

    rows = []
    for region in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scen, _ = sc.clip_pd(
                    sc.aggregate_to_carbon_buckets(df, regions=[region], cfg=abs_cfg),
                    abs_cfg, warn=False,
                )
                port = pf.stylised_hl_portfolio(scen.dates, **share)
                port = breach.calibrate_cet1_for_ratio(port, scen, abs_cfg, target_ratio)
        except Exception as exc:
            rows.append({"region": region, "status": type(exc).__name__})
            continue

        bau = breach.cet1_ratio(port, scen, abs_cfg, scenario=sc.BAU_LABEL)
        shock, worst, year = -np.inf, None, None
        for name in scen.scenarios:
            if name == sc.BAU_LABEL:
                continue
            drop = (bau - breach.cet1_ratio(port, scen, abs_cfg, scenario=name)) * 100
            if drop.max() > shock:
                shock, worst, year = float(drop.max()), name, int(scen.dates[drop.argmax()])

        row = {
            "region": region, "status": "ok", "shock_pp": shock,
            "worst_scenario": worst, "worst_year": year,
            "bau_min_ratio_pct": float(bau.min()) * 100,
            "n_narratives": scen.n_scenarios - 1,
        }
        conventions = [("abs", abs_cfg),
                       ("rel", cfg.r_star_conventions(float(bau.min()))[1])]
        for tag, candidate in conventions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dist = breach.distance_to_breach(port, scen, candidate, check_cushion=False)
            row[f"dist_{tag}_bn"] = float(dist.min()) / 1e9
            row[f"breach_{tag}"] = bool((dist < 0).any())
        rows.append(row)

    table = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return table[COLUMNS]


def breach_class(table: pd.DataFrame) -> pd.Series:
    """Label each region by which conventions it breaches under.

    The three-way split the figures colour by: ``"both"``, ``"relative only"`` or
    ``"neither"``. There is no "absolute only" class -- the relative threshold sits
    above the absolute one on every region here, so breaching the absolute one implies
    breaching the relative one.
    """
    both = table["breach_abs"].fillna(False) & table["breach_rel"].fillna(False)
    relative = ~table["breach_abs"].fillna(False) & table["breach_rel"].fillna(False)
    return pd.Series(
        np.where(both, "both", np.where(relative, "relative only", "neither")),
        index=table.index,
    )


def summarise(table: pd.DataFrame) -> str:
    """One-paragraph summary of a scan, for the driver to print."""
    ok = table[table["status"] == "ok"]
    skipped = table[table["status"] != "ok"]
    worst = ok.loc[ok["shock_pp"].idxmax()]
    lines = [
        f"{len(ok)} regions scanned"
        + (f", {len(skipped)} skipped ({sorted(skipped['region'])})" if len(skipped) else ""),
        f"shock: median {ok['shock_pp'].median():.2f} pp, "
        f"max {worst['shock_pp']:.2f} pp on {worst['region']} "
        f"({worst['worst_scenario']} {int(worst['worst_year'])})",
        f"breaches: {int(ok['breach_abs'].sum())}/{len(ok)} absolute, "
        f"{int(ok['breach_rel'].sum())}/{len(ok)} relative",
        "worst narrative: "
        + ", ".join(f"{k} {v}" for k, v in ok["worst_scenario"].value_counts().items()),
    ]
    return "\n".join("  " + line for line in lines)
