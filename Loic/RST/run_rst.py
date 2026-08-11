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

# EU27: the home region for the ICAAP reading, and the one aggregate that clears every
# trap at once. All four narratives cover it, unlike World, which has no DAPS rows. All
# four stay distinct on it, unlike the elementary European regions (France, Germany,
# Italy, Spain), where DIRE and HWTP are numerically identical -- EU27 is an aggregate
# and mixes in countries where they diverge, by up to 3.8 pp. Being an aggregate its
# baseline PD is also genuinely sectoral, and its PDs top out at 0.17, so nothing is
# censored at the upper bound.
DEFAULT_REGIONS = ["EU27"]

#: The previous default, kept as the ready-made alternative when DAPS is needed.
#: Covered by all four narratives, and deliberately not EU-only: DIRE and HWTP have
#: *identical* PD projections on 39 of the 53 regions, including every European one.
BASKET_REGIONS = ["EU27", "USA - USA", "China - CHN", "India - IND", "Japan - JPN", "Brazil - BRA"]

#: Region the standing global reference figure is drawn on, whatever --regions says.
WORLD_REGION = "World"

# Reference narrative s_0 defining p0. The NGFS business-as-usual path, not one of the
# four narratives: CLIMACRED defines every narrative as baseline_pd + pd_adjustment, so
# taking the BAU as p0 makes the erosion exactly the NGFS climate increment. Passing a
# narrative name instead measures every scenario against *that* narrative, which is a
# different question -- see the module docstring of scenarios.py.
DEFAULT_BASELINE = sc.BAU_LABEL


def slug(text: str) -> str:
    """Filesystem-safe name, matching the convention of ``run_pd_analysis.py``."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scenario", default=DEFAULT_BASELINE,
                        help="reference narrative s_0 defining p0 (default: the NGFS "
                             "business-as-usual path)")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="regions to pool (default: EU27; note 'World' has no DAPS "
                             "rows and the elementary European regions collapse DIRE "
                             "onto HWTP)")
    parser.add_argument("--basket", action="store_true",
                        help=f"shorthand for --regions {' '.join(BASKET_REGIONS)}")
    parser.add_argument("--sectors", nargs="+", default=None,
                        help="restrict the study to these sectors (overrides --top-sectors)")
    parser.add_argument("--top-sectors", type=int, default=None,
                        help="restrict to the N sectors whose PD moves most, "
                             "ranked by --sector-criterion")
    parser.add_argument("--sector-criterion", choices=list(sc.SECTOR_CRITERIA),
                        default="erosion",
                        help="how to rank sectors: erosion = |dPsi| (capital-relevant, "
                             "censored by p_max), amplitude = |pd_adjustment|, "
                             "dispersion = spread across narratives")
    parser.add_argument("--granularity", choices=["sector", "carbon"], default="sector",
                        help="one bucket per (sector x region), or collapse onto H/L")
    parser.add_argument("--pd-max", type=float, default=None,
                        help="upper bound of the admissible PD domain (declared "
                             "sensitivity; default 0.50, hard bound 0.7202)")
    parser.add_argument("--year", type=int, default=None,
                        help="date for the tornado and frontier figures (default: last)")
    parser.add_argument("--r-star-convention", choices=["absolute", "relative", "both"],
                        default="both", help="breach level convention(s) to run")
    parser.add_argument("--target-ratio", type=float, default=0.13,
                        help="CET1 ratio on p0 that the stylised balance sheet is pinned to")
    parser.add_argument("--r0-year", type=int, default=None,
                        help="year defining R_0 for the relative R* convention "
                             "(default: the tightest date of the reference path)")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-read the .xlsx")
    args = parser.parse_args()

    df = load_pd(refresh=args.refresh)
    sc.check_mapping_coverage(df)
    chosen = args.regions or (BASKET_REGIONS if args.basket else DEFAULT_REGIONS)
    regions = [resolve_region(df, r) for r in chosen]

    cfg = DEFAULT_CONFIG.with_baseline_scenario(args.baseline_scenario)
    if args.pd_max is not None:
        cfg = cfg.with_pd_max(args.pd_max)

    # -- sector universe. Scored on the same regions the study runs on, otherwise
    # sectors would be chosen on evidence the study never sees.
    selection = None
    sectors = args.sectors
    if sectors is None and args.top_sectors is not None:
        selection = sc.select_sectors(
            df, cfg, n=args.top_sectors, criterion=args.sector_criterion, regions=regions
        )
        sectors = list(selection.sectors)

    if args.granularity == "carbon":
        raw = sc.aggregate_to_carbon_buckets(
            df, args.baseline_scenario, regions=regions, sectors=sectors
        )
        bucket_note = "mapping.csv, exposure-weighted"
    else:
        raw = sc.from_climacred(
            df, args.baseline_scenario, regions=regions, sectors=sectors
        )
        bucket_note = "one per (sector x region)"

    label = SCENARIO_LABELS.get(
        args.baseline_scenario,
        "NGFS business-as-usual path (CLIMACRED baseline_pd)"
        if args.baseline_scenario == sc.BAU_LABEL
        else "",
    )
    print(f"Regions       : {regions}")
    print(f"Scenarios     : {list(raw.scenarios)}")
    print(f"Buckets       : {raw.n_buckets} ({bucket_note})")
    print(f"Horizon       : {raw.dates[0]} -> {raw.dates[-1]}")
    print(f"PD domain     : [{cfg.pd_bounds[0]:.1e}, {cfg.pd_bounds[1]:.2f}]")
    # "reference narrative", never "baseline": CLIMACRED already uses baseline_pd for
    # the BAU *level*, and conflating the two is how p0 silently becomes a narrative
    print(f"p0 (reference): {args.baseline_scenario} — {label}")

    if selection is not None:
        print(
            f"\nSectors retained ({selection.criterion}, {len(selection.sectors)} of "
            f"{len(selection.scores)}) — a restricted study is a CONCENTRATION study: "
            "the whole book is respread over these sectors, so distances are not "
            "comparable to another sector universe."
        )
        print(selection.selected_scores[list(sc.SECTOR_CRITERIA)].round(4).to_string())
        # only worth flagging when saturation actually drove the ranking: under the
        # other criteria the erosion column is reported but does not order anything
        if selection.n_saturated and selection.criterion == "erosion":
            print(
                f"  {selection.n_saturated} of them saturate the erosion score at "
                f"{selection.saturation_value:.4f}: p_max, not the data, sets their "
                "score. Try --pd-max 0.5."
            )
    elif sectors is not None:
        print(f"\nSectors retained (explicit): {sectors}")

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
    if args.granularity == "carbon":
        port = pf.stylised_hl_portfolio(scen.dates)
    else:
        # uniform weights across the retained buckets: the honest placeholder with no
        # sector-level book, and the first assumption a reader should attack
        port = pf.stylised_portfolio(scen.buckets, scen.dates)
    port = breach.calibrate_cet1_for_ratio(port, scen, cfg, target_ratio=args.target_ratio)
    ratio0 = breach.cet1_ratio(port, scen, cfg)
    print(
        f"\nBalance sheet (stylised): {port.n_buckets} buckets, "
        f"{port.total_exposure[0] / 1e9:.0f} bn total exposure, "
        f"CET1_0={port.cet1_0 / 1e9:.1f} bn, RWA_oth={port.rwa_oth / 1e9:.0f} bn"
    )
    # spell out the split when it is small enough to read: the carbon share is the
    # parameter most likely to be varied by hand, and it must not be invisible
    if port.n_buckets <= 6:
        split = ", ".join(
            f"{b}={e / 1e9:.0f} bn ({e / port.total_exposure[0]:.0%})"
            for b, e in zip(port.buckets, port.exposure[:, 0])
        )
        print(f"  exposure: {split}")
    print(f"CET1 ratio on p0: {np.round(ratio0, 4).tolist()}")

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

    fig = report.plot_regulatory_functions(main_cfg, scenarios=scen)
    path = FIGS_DIR / "rst_regulatory_functions.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # Standing global reference, always written and always on World, so a
    # region-restricted run can still be read against the world aggregate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DAPS-absent-from-World already reported above
        fig = report.plot_sector_max_pd(df, main_cfg, regions=[WORLD_REGION])
    path = FIGS_DIR / "rst_sector_world_pd.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # The study's own regions, only when they differ -- otherwise it duplicates the
    # World figure above. This is the one whose censoring matches what the run applies.
    if regions != [WORLD_REGION]:
        fig = report.plot_sector_max_pd(df, main_cfg, regions=regions)
        path = FIGS_DIR / "rst_sector_max_pd.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # The two breach figures are drawn for *every* admissible convention, not just the
    # first: the conventions preserve the scenario ranking but not the breach set, so a
    # single-convention figure shows only half the result. Filenames carry the
    # convention, otherwise the second run would overwrite the first.
    for candidate in usable:
        tag = slug(candidate.sensitivity.r_star_convention)

        fig = report.plot_distance_to_breach(port, scen, candidate)
        path = FIGS_DIR / f"rst_distance_to_breach_{tag}.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written.append(path)

        if port.n_buckets != 2:
            continue
        try:
            fig = report.plot_iso_breach_frontier(port, scen, candidate, date_index)
        except ValueError as exc:
            # a frontier can fall entirely outside the admissible PD square, which is a
            # finding about that convention, not a reason to abandon the other one
            print(f"\nno iso-breach frontier under {candidate.label}: {exc}")
            continue
        path = FIGS_DIR / f"rst_iso_breach_{tag}_{int(scen.dates[date_index])}.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    fig = report.plot_critical_pd_tornado(port, scen, main_cfg, worst, date_index)
    path = FIGS_DIR / f"rst_critical_pd_{slug(worst)}_{int(scen.dates[date_index])}.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    print()
    for path in written:
        print(f"written: {path.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
