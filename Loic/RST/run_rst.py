"""Run the Level 1 CET1 climate reverse stress test end to end.

Chains: load CLIMACRED PDs -> aggregate to carbon buckets -> clip onto the admissible
domain (reporting what moved) -> check the date-0 anchor -> calibrate the stylised
balance sheet -> check admissibility (H > 0) -> rank scenarios under both R* conventions
-> write figures to Figs/.

Examples
--------
    python run_rst.py
    python run_rst.py --baseline-scenario DIRE --target-ratio 0.14
    python run_rst.py --regions EU27 "China - CHN" --r-star-convention absolute
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import breach
import portfolio as pf
import report
import scenarios as sc
from config import DEFAULT_CONFIG
from pd.climacred_loader import SCENARIO_LABELS, load_pd, resolve_region

BASE_DIR = Path(__file__).resolve().parent
FIGS_DIR = BASE_DIR / "Figs"

# Six regions covered by all four scenarios, and deliberately not EU-only: DIRE and
# HWTP have *identical* PD projections on 39 of the 53 regions, including every
# European one. Ranking scenarios on a European-only book would therefore show two
# of the four as indistinguishable -- a property of the source file, not of the model.
DEFAULT_REGIONS = ["EU27", "USA - USA", "China - CHN", "India - IND", "Japan - JPN", "Brazil - BRA"]

#: Reference narrative. Current-policies reading; see the module docstring of
#: scenarios.py for the three candidate conventions.
DEFAULT_BASELINE = "HWTP"


def slug(text: str) -> str:
    """Filesystem-safe name, matching the convention of ``run_pd_analysis.py``."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scenario", default=DEFAULT_BASELINE,
                        help="reference narrative s_0 defining p0")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="regions to pool into the carbon buckets")
    parser.add_argument("--year", type=int, default=None,
                        help="date for the tornado and frontier figures (default: last)")
    parser.add_argument("--r-star-convention", choices=["absolute", "relative", "both"],
                        default="both", help="breach level convention(s) to run")
    parser.add_argument("--target-ratio", type=float, default=0.13,
                        help="baseline CET1 ratio the stylised balance sheet is pinned to")
    parser.add_argument("--r0-year", type=int, default=None,
                        help="year defining R_0 for the relative R* convention "
                             "(default: the tightest date of the reference path)")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-read the .xlsx")
    args = parser.parse_args()

    df = load_pd(refresh=args.refresh)
    sc.check_mapping_coverage(df)
    regions = [resolve_region(df, r) for r in (args.regions or DEFAULT_REGIONS)]

    cfg = DEFAULT_CONFIG.with_baseline_scenario(args.baseline_scenario)
    raw = sc.aggregate_to_carbon_buckets(df, args.baseline_scenario, regions=regions)

    print(f"Regions   : {regions}")
    print(f"Scenarios : {list(raw.scenarios)}")
    print(f"Buckets   : {list(raw.buckets)}  (mapping.csv, exposure-weighted)")
    print(f"Horizon   : {raw.dates[0]} -> {raw.dates[-1]}")
    print(f"Baseline  : {args.baseline_scenario} — {SCENARIO_LABELS.get(args.baseline_scenario, '')}")

    # -- clipping, never silent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scen, clip = sc.clip_pd(raw, cfg)
        scen.check_present_anchor()
    print(f"\nClipping  : {clip.summary()}")
    for w in caught:
        if "clipping" not in str(w.message):
            print(f"WARNING   : {w.message}")

    # -- balance sheet, pinned so the exercise is admissible by construction
    port = pf.stylised_hl_portfolio(scen.dates)
    port = breach.calibrate_cet1_for_ratio(port, scen, cfg, target_ratio=args.target_ratio)
    ratio0 = breach.cet1_ratio(port, scen, cfg)
    print(
        f"\nBalance sheet (stylised): exposure H={port.exposure[0, 0] / 1e9:.0f} bn, "
        f"L={port.exposure[1, 0] / 1e9:.0f} bn, CET1_0={port.cet1_0 / 1e9:.1f} bn, "
        f"RWA_oth={port.rwa_oth / 1e9:.0f} bn"
    )
    print(f"Baseline CET1 ratio: {np.round(ratio0, 4).tolist()}")

    # -- R* conventions.
    # R_0 anchors the relative convention. The reference CET1 ratio is *not* flat here
    # -- the CLIMACRED baseline PD is a near-zero placeholder in 2022 and jumps in
    # 2023 -- so anchoring on date 0 would put R* above the reference path at almost
    # every later date and make the relative convention vacuous (H[n] < 0 everywhere).
    # Anchoring on the tightest date of the reference path is the reading that keeps
    # "300 bp below where the bank is" meaningful over a whole horizon.
    r0_index = int(ratio0.argmin()) if args.r0_year is None else scen.date_index(args.r0_year)
    r0 = float(ratio0[r0_index])
    print(f"R_0 for the relative convention: {r0:.4f} (year {int(scen.dates[r0_index])})")

    if args.r_star_convention == "both":
        configs = cfg.r_star_conventions(current_ratio=r0)
    elif args.r_star_convention == "absolute":
        configs = [cfg.with_r_star(0.105, "absolute")]
    else:
        configs = [cfg.r_star_conventions(current_ratio=r0)[1]]

    # -- admissibility, per convention (H > 0 is equivalent to Ratio_0 > R*)
    usable = []
    for candidate in configs:
        try:
            h = breach.cushion(port, scen, candidate, check=True)
        except ValueError as exc:
            print(f"\nINADMISSIBLE under {candidate.label}: {exc}")
            continue
        print(f"\nCushion H[n] under {candidate.label} (bn EUR): {np.round(h / 1e9, 2).tolist()}")
        usable.append(candidate)

    if not usable:
        raise SystemExit(
            "no admissible convention: the bank is at or below the breach level on its "
            "own reference narrative at every date (assumption 8). Raise --target-ratio."
        )

    # -- rankings
    for candidate in usable:
        print(f"\nScenario ranking under {candidate.label} (worst date of the horizon):")
        print(report.rank_scenarios(port, scen, candidate).round(3).to_string())

    if len(usable) > 1:
        print("\nR* convention comparison:")
        comparison = report.compare_r_star_conventions(port, scen, usable)
        print(comparison.round(3).to_string())
        disagreement = int(comparison["rank_disagreement"].max())
        split = comparison.index[comparison["breach_disagreement"]].tolist()
        print(
            f"max rank disagreement: {disagreement}"
            + (" — the conventions order the scenarios differently."
               if disagreement else " — the conventions agree on the ordering.")
        )
        print(
            f"breach-set disagreement: {split}"
            if split
            else "breach-set disagreement: none — same breach set under both conventions."
        )

    # -- baseline-narrative invariance (specification section 3, property 2)
    main_cfg = usable[0]
    others = [s for s in scen.scenarios if s != scen.baseline_scenario]
    if others:
        alt = scen.with_baseline(others[0])
        reference = breach.distance_to_breach(port, scen, main_cfg, check_cushion=False)
        shifted = breach.distance_to_breach(port, alt, main_cfg, check_cushion=False)
        # relative, not absolute: these are euro amounts around 1e10, so float64
        # round-off alone is worth ~1e-5 euro and says nothing on its own
        gap = np.abs(reference - shifted).max() / np.abs(reference).max()
        print(
            f"\nBaseline-narrative invariance (p0 = {scen.baseline_scenario} vs "
            f"{others[0]}): max relative |distance difference| = {gap:.2e} "
            "(machine precision expected — the breach set does not depend on p0)"
        )

    # -- figures
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    date_index = scen.n_dates - 1 if args.year is None else scen.date_index(args.year)
    worst = report.rank_scenarios(port, scen, main_cfg).index[0]
    written = []

    fig = report.plot_distance_to_breach(port, scen, main_cfg)
    path = FIGS_DIR / f"rst_distance_to_breach_{slug(main_cfg.sensitivity.r_star_convention)}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig = report.plot_critical_pd_tornado(port, scen, main_cfg, worst, date_index)
    path = FIGS_DIR / f"rst_critical_pd_{slug(worst)}_{int(scen.dates[date_index])}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    if port.n_buckets == 2:
        fig = report.plot_iso_breach_frontier(port, scen, main_cfg, date_index)
        path = FIGS_DIR / f"rst_iso_breach_{int(scen.dates[date_index])}.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    print()
    for path in written:
        print(f"written: {path.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
