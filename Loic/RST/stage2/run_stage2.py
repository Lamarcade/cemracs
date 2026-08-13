"""Run the reverse stress test with the IFRS 9 Stage 2 switch, and compare to Stage 1.

Sector granularity throughout: a bucket is one ``(sector, region)`` pair, homogeneous
enough for an all-or-nothing migration to mean something. The H/L collapse would move
half the book on a single trigger.

Examples
--------
    python -m stage2.run_stage2
    python -m stage2.run_stage2 --thresh 1.5
    python -m stage2.run_stage2 --regions "Latvia - LVA"
    python -m stage2.run_stage2 --thresh 99          # must reproduce Stage 1 exactly
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import breach
import portfolio as pf
import report
import scenarios as sc
from config import DEFAULT_CONFIG
from pd.climacred_loader import load_pd, resolve_region
from stage2 import breach2, report2, staging

STAGE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_DIR.parents[2]  # stage2 -> RST -> Loic -> cemracs

#: Stage 2 figures live with the package that produces them, not in the Level 1 Figs/.
#: They answer a different question and carry a different set of assumptions, so mixing
#: them into the main gallery invites reading a staged result as a Level 1 one.
FIGS_DIR = STAGE2_DIR / "Figs2"

#: Same default perimeter as the Level 1 driver: the one aggregate covered by all four
#: narratives that also keeps them distinct.
DEFAULT_REGIONS = ["EU27"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS,
                        help="regions to study (default: EU27)")
    parser.add_argument("--sicr-rule", choices=sorted(staging.SICR_RULES),
                        default="ecb",
                        help="SICR preset: ecb = tripling since origination with the "
                             "low credit risk exemption and the 20%% absolute backstop, "
                             "simple = bare doubling, climate = vs the same-date baseline")
    parser.add_argument("--thresh", type=float, default=None,
                        help="override the relative multiple")
    parser.add_argument("--absolute-backstop", type=float, default=None,
                        help="PD above which SICR triggers on its own (ECB: 0.20)")
    parser.add_argument("--no-absolute-backstop", action="store_true",
                        help="drop the absolute backstop")
    parser.add_argument("--low-risk-floor", type=float, default=None,
                        help="PD below which the relative trigger does not apply "
                             "(ECB: 0.003), tested on the initial or the current PD")
    parser.add_argument("--no-low-risk-exemption", action="store_true",
                        help="apply the relative trigger at any PD level")
    parser.add_argument("--window", type=float, default=staging.LIFETIME_WINDOW,
                        help="lifetime ECL window in years")
    parser.add_argument("--baseline-scenario", default=sc.BAU_LABEL,
                        help="reference narrative s_0 defining p0")
    parser.add_argument("--target-ratio", type=float, default=0.13,
                        help="CET1 ratio on p0 the balance sheet is pinned to")
    parser.add_argument("--r-star-convention", choices=["absolute", "relative", "both"],
                        default="both", help="breach level convention(s) to run")
    parser.add_argument("--pd-max", type=float, default=None,
                        help="upper bound of the admissible PD domain")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-read the .xlsx")
    args = parser.parse_args()

    df = load_pd(refresh=args.refresh)
    sc.check_mapping_coverage(df)
    regions = [resolve_region(df, r) for r in args.regions]

    cfg = DEFAULT_CONFIG.with_baseline_scenario(args.baseline_scenario)
    if args.pd_max is not None:
        cfg = cfg.with_pd_max(args.pd_max)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = sc.from_climacred(df, args.baseline_scenario, regions=regions)
        scen, clip = sc.clip_pd(raw, cfg)
    for w in caught:
        if "clipping" not in str(w.message):
            print(f"WARNING: {w.message}\n")

    rule = staging.SICR_RULES[args.sicr_rule]
    if args.thresh is not None:
        rule = replace(rule, thresh=args.thresh, name="")
    if args.no_absolute_backstop:
        rule = replace(rule, absolute_backstop=None, name="")
    elif args.absolute_backstop is not None:
        rule = replace(rule, absolute_backstop=args.absolute_backstop, name="")
    if args.no_low_risk_exemption:
        rule = replace(rule, low_risk_floor=None, name="")
    elif args.low_risk_floor is not None:
        rule = replace(rule, low_risk_floor=args.low_risk_floor, name="")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        multiplier, flags = staging.provision_multiplier(
            scen.pd_cube, scen.baseline_pd, rule=rule, window=args.window
        )
    for w in caught:
        print(f"WARNING: {w.message}\n")
    migration = staging.migration_table(flags, scen.scenarios, scen.dates)
    ones = np.ones_like(multiplier)

    # Two balance sheets, because they answer different questions and can disagree
    # sharply once the baseline itself migrates:
    #   stage1  the bank was capitalised for Stage 1, then staging hit -- the transition
    #           cost, which can leave the exercise inadmissible
    #   staged  the bank is capitalised for the accounting it actually applies
    port = pf.stylised_portfolio(scen.buckets, scen.dates)
    port_s1 = breach.calibrate_cet1_for_ratio(port, scen, cfg, target_ratio=args.target_ratio)
    port_s2 = (
        breach2.calibrate_staged(port, scen, cfg, multiplier, args.target_ratio)
        if rule.stages_baseline else port_s1
    )
    books = [("Stage 1 calibration", port_s1)]
    if rule.stages_baseline:
        books.append(("recalibrated on the staged baseline", port_s2))

    ratio0 = breach2.baseline_ratio_staged(port_s2, scen, cfg, multiplier)

    print(f"Regions       : {regions}")
    print(f"Buckets       : {scen.n_buckets} (one per sector x region)")
    print(f"Scenarios     : {list(scen.scenarios)}")
    print(f"p0 (reference): {args.baseline_scenario}")
    print(f"SICR          : {rule.label}, re-tested each date")
    print(f"Lifetime      : {args.window:g}-year window, last PD held flat past "
          f"{int(scen.dates[-1])}")
    print(f"Clipping      : {clip.summary()}")

    staged = multiplier[flags]
    lam0 = breach2.baseline_multiplier(scen, multiplier)
    print(f"\nProvision multiplier where triggered: "
          + (f"{staged.min():.2f} to {staged.max():.2f}" if staged.size else "never triggered"))
    print(f"Reference narrative multiplier: max {lam0.max():.3f}, "
          f"{(lam0 > 1).mean():.0%} of its cells staged"
          + ("  <- the baseline migrates too, so the cushion is not the Stage 1 one"
             if rule.stages_baseline else "  (never migrates by construction)"))
    print(f"CET1_0        : {port_s1.cet1_0 / 1e9:.1f} bn Stage 1 calibration"
          + (f", {port_s2.cet1_0 / 1e9:.1f} bn recalibrated" if rule.stages_baseline else ""))
    print(f"Staged baseline CET1 ratio: {np.round(ratio0, 4).tolist()}")

    # what each trigger contributes on its own: the two pull in opposite directions and
    # the totals alone would not say which one is doing the work
    if rule.absolute_backstop is not None or rule.low_risk_floor is not None:
        bare = replace(rule, absolute_backstop=None, low_risk_floor=None, name="")
        only_relative = staging.sicr_flags(scen.pd_cube, scen.baseline_pd, bare)
        parts = [f"relative alone {only_relative.mean():.1%}"]
        if rule.low_risk_floor is not None:
            kept = staging.sicr_flags(
                scen.pd_cube, scen.baseline_pd, replace(rule, absolute_backstop=None))
            parts.append(f"low-risk exemption removes {(only_relative & ~kept).mean():.1%}")
        if rule.absolute_backstop is not None:
            backstop = scen.pd_cube > rule.absolute_backstop
            parts.append(f"20% backstop adds {(backstop & ~only_relative).mean():.1%}")
        print(f"\nTrigger decomposition: " + ", ".join(parts)
              + f"  ->  {flags.mean():.1%} in Stage 2")

    print("\nShare of buckets in Stage 2 (%):")
    print((migration * 100).round(1).to_string())

    # The relative threshold has to come from the *staged* baseline ratio: built from
    # the unstaged one it would be compared against a staged path, and the reference
    # narrative would appear to breach purely from the mismatch.
    if args.r_star_convention == "both":
        configs = cfg.r_star_conventions(current_ratio=float(ratio0.min()))
    elif args.r_star_convention == "absolute":
        configs = [cfg.with_r_star(0.105, "absolute")]
    else:
        configs = [cfg.r_star_conventions(current_ratio=float(ratio0.min()))[1]]

    context = report.RunContext(
        regions=tuple(regions), baseline_scenario=args.baseline_scenario,
        target_ratio=args.target_ratio, granularity="sector",
        n_buckets=scen.n_buckets, pd_bounds=cfg.pd_bounds,
    )

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    fig = report2.plot_stage_migration(migration, scen, rule, context=context)
    path = FIGS_DIR / "stage2_migration.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    for candidate in configs:
      for book_label, book in books:
        d1 = breach2.distance_to_breach_staged(book, scen, candidate, ones)
        d2 = breach2.distance_to_breach_staged(book, scen, candidate, multiplier)
        check = breach2.check_forms_agree_staged(book, scen, candidate, multiplier)
        h2 = breach2.cushion_staged(book, scen, candidate, multiplier)

        table = pd.DataFrame(
            {
                "stage1_bn": d1.min(axis=1) / 1e9,
                "stage2_bn": d2.min(axis=1) / 1e9,
                "cost_bn": (d1.min(axis=1) - d2.min(axis=1)) / 1e9,
                "breach_s1": (d1 < 0).any(axis=1),
                "breach_s2": (d2 < 0).any(axis=1),
                "worst_year": scen.dates[d2.argmin(axis=1)],
            },
            index=pd.Index(scen.scenarios, name="scenario"),
        ).sort_values("stage2_bn")

        print(f"\n=== {candidate.label} — {book_label} ===")
        if (h2 <= 0).any():
            print(f"  INADMISSIBLE: staged H[n] <= 0 at "
                  f"{scen.dates[h2 <= 0].tolist()} (min {h2.min() / 1e9:+.2f} bn). "
                  "The reference narrative alone is at or below the threshold once its "
                  "own book is in Stage 2 — read the rows below as a diagnostic, not a "
                  "reverse stress test.")
        print(table.round(3).to_string())
        flipped = table.index[~table["breach_s1"] & table["breach_s2"]].tolist()
        print(f"  breach set  Stage 1: {table.index[table['breach_s1']].tolist() or 'none'}")
        print(f"              Stage 2: {table.index[table['breach_s2']].tolist() or 'none'}")
        print(f"  newly breaching under Stage 2: {flipped or 'none'}")
        print(f"  (11) vs (12) staged: max relative gap {check['max_rel_gap']:.2e}, "
              f"{check['n_staged_cells']} staged cells, "
              f"{check['n_breach_affine']}/{check['n_cells']} cells in breach")

        # only the book the run is really about gets a figure, otherwise every
        # convention would produce two near-identical charts
        if book is books[-1][1]:
            fig = report2.plot_stage1_vs_stage2(d1, d2, scen, candidate, context=context)
            tag = candidate.sensitivity.r_star_convention
            path = FIGS_DIR / f"stage2_distance_{tag}.png"
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    print()
    for path in written:
        print(f"written: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
