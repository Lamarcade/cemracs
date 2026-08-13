"""Rank every CLIMACRED region by climate shock and distance to breach.

A dedicated driver rather than a flag on ``run_rst.py``: this runs the whole chain 53
times and answers a different question -- not "what happens to this book" but "where is
the exposure worst". It writes two figures to Figs/ and prints the table.

Each region is calibrated independently to the same starting CET1 ratio, so a row is
*a hypothetical bank concentrated in that region*, not that country's banking system.

Examples
--------
    python run_region_scan.py
    python run_region_scan.py --target-ratio 0.12 --n-regions 25
    python run_region_scan.py --regions EU27 "China - CHN" "Poland - POL"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import portfolio as pf
import regions as rg
import report
import scenarios as sc
from config import DEFAULT_CONFIG
from pd.climacred_loader import load_pd, resolve_region

BASE_DIR = Path(__file__).resolve().parent
FIGS_DIR = BASE_DIR / "Figs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=None,
                        help="regions to scan (default: all 53)")
    parser.add_argument("--baseline-scenario", default=sc.BAU_LABEL,
                        help="reference narrative s_0 defining p0")
    parser.add_argument("--target-ratio", type=float, default=0.13,
                        help="starting CET1 ratio every region is pinned to")
    parser.add_argument("--high-carbon-share", type=float,
                        default=pf.DEFAULT_HIGH_CARBON_SHARE,
                        help="brown share of the book")
    parser.add_argument("--pd-max", type=float, default=None,
                        help="upper bound of the admissible PD domain")
    parser.add_argument("--n-regions", type=int, default=20,
                        help="how many regions each figure shows")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-read the .xlsx")
    args = parser.parse_args()

    df = load_pd(refresh=args.refresh)
    sc.check_mapping_coverage(df)
    names = None if args.regions is None else [resolve_region(df, r) for r in args.regions]

    cfg = DEFAULT_CONFIG.with_baseline_scenario(args.baseline_scenario)
    if args.pd_max is not None:
        cfg = cfg.with_pd_max(args.pd_max)

    table = rg.scan_regions(
        df, cfg, regions=names, target_ratio=args.target_ratio,
        high_carbon_share=args.high_carbon_share,
    )

    print(f"p0 (reference) : {args.baseline_scenario}")
    print(f"PD domain      : [{cfg.pd_bounds[0]:.1e}, {cfg.pd_bounds[1]:.2f}]")
    print(f"pinned at      : {args.target_ratio:.1%} baseline CET1 ratio\n")
    print(rg.summarise(table))

    ok = table[table["status"] == "ok"]
    show = ["region", "shock_pp", "worst_scenario", "worst_year",
            "dist_abs_bn", "breach_abs", "dist_rel_bn", "breach_rel"]
    with pd.option_context("display.width", 200):
        print(f"\nBiggest shocks (top {args.n_regions}):")
        print(ok.nlargest(args.n_regions, "shock_pp")[show].round(2).to_string(index=False))
        print(f"\nDeepest breaches, relative convention (top {args.n_regions}):")
        print(ok.nsmallest(args.n_regions, "dist_rel_bn")[show].round(2).to_string(index=False))

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    # the perimeter is the x/y axis here, so it is marked as varying by the plots
    context = report.RunContext(
        regions=tuple(names or sorted(df["region"].unique())),
        baseline_scenario=args.baseline_scenario,
        brown_share=args.high_carbon_share,
        target_ratio=args.target_ratio,
        bucket_rule="transition", aggregation="certainty_equivalent",
        granularity="carbon", n_buckets=2, pd_bounds=cfg.pd_bounds,
    )
    written = []
    for name, plot in [("rst_region_shock", report.plot_region_shock),
                       ("rst_region_distance", report.plot_region_distance)]:
        fig = plot(table, n_regions=args.n_regions, context=context)
        path = FIGS_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    print()
    for path in written:
        print(f"written: {path.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
