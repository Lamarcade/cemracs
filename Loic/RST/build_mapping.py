"""Regenerate the data-driven columns of ``mapping.csv``.

A dedicated maintenance command rather than a flag on ``run_rst.py``: rewriting the
bucket mapping is a deliberate act whose output is a diff to be read, not a side effect
of running an analysis. The specification calls the mapping the most contestable
assumption of the whole chain and asks for it to be a versioned artefact -- which only
works if regenerating it is explicit and its result is reviewable.

What the script owns and what it leaves alone:

- **rewrites** ``carbon_bucket_transition`` and ``median_pd_adjustment_hwtp``, both
  computed by :func:`scenarios.classify_transition_buckets`
- **preserves** the header comment block and every hand-written column --
  ``sector_group``, ``carbon_bucket_taxonomy``, ``nace_codes``

Examples
--------
    python build_mapping.py
    python build_mapping.py --regions EU27          # regional sensitivity
    python build_mapping.py --scenario DIRE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import scenarios as sc
from pd.climacred_loader import load_pd, resolve_region

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "mapping.csv"

#: Column order written back to the file.
COLUMNS = [
    "ngfs_sector",
    "sector_group",
    "carbon_bucket_taxonomy",
    "carbon_bucket_transition",
    "median_pd_adjustment_hwtp",
    "nace_codes",
]


def read_header(path: Path) -> str:
    """Return the leading ``#`` comment block verbatim, so rewriting keeps it."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=sc.BUCKET_SCENARIO,
                        help="classifying narrative (must be a transition scenario)")
    parser.add_argument("--regions", nargs="+", default=list(sc.BUCKET_REGIONS),
                        help="reference regions the classification is read on")
    parser.add_argument("--start-year", type=int, default=sc.BUCKET_START_YEAR,
                        help="first year retained (pd_adjustment is 0 before 2024)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the moves without rewriting the file")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-read the .xlsx")
    args = parser.parse_args()

    df = load_pd(refresh=args.refresh)
    regions = [resolve_region(df, r) for r in args.regions]

    current = pd.read_csv(MAPPING_PATH, comment="#")
    classified = sc.classify_transition_buckets(
        df, scenario=args.scenario, regions=regions, start_year=args.start_year
    )

    print(f"scenario   : {args.scenario}")
    print(f"regions    : {regions}")
    print(f"years      : {args.start_year} onwards")
    counts = classified["carbon_bucket_transition"].value_counts()
    print(f"split      : H={int(counts.get('H', 0))} L={int(counts.get('L', 0))}")

    merged = current.drop(
        columns=[c for c in ("carbon_bucket_transition", "median_pd_adjustment_hwtp")
                 if c in current.columns]
    ).merge(classified, on="ngfs_sector", how="right", validate="one_to_one")
    if merged[["sector_group", "nace_codes"]].isna().to_numpy().any():
        orphans = merged.loc[merged["sector_group"].isna(), "ngfs_sector"].tolist()
        raise ValueError(
            f"sectors present in the data but absent from {MAPPING_PATH.name}: {orphans}. "
            "Add their hand-written columns first -- this script never invents them."
        )
    merged = merged.rename(columns={"median_pd_adjustment": "median_pd_adjustment_hwtp"})
    merged["median_pd_adjustment_hwtp"] = merged["median_pd_adjustment_hwtp"].round(6)

    # the useful output: what actually moved, and how close each call was
    previous = (
        current.set_index("ngfs_sector")["carbon_bucket_transition"]
        if "carbon_bucket_transition" in current.columns
        else current.set_index("ngfs_sector")["carbon_bucket_taxonomy"]
    )
    label = ("previous transition column" if "carbon_bucket_transition" in current.columns
             else "taxonomy column (first run)")
    moves = [
        (row.ngfs_sector, previous.get(row.ngfs_sector), row.carbon_bucket_transition,
         row.median_pd_adjustment_hwtp)
        for row in merged.itertuples()
        if previous.get(row.ngfs_sector) != row.carbon_bucket_transition
    ]
    print(f"\n{len(moves)} sector(s) move vs the {label}:")
    for sector, was, now, median in sorted(moves, key=lambda m: -abs(m[3])):
        print(f"  {was} -> {now}  {median:+9.4f} pp  {sector}")

    borderline = merged.reindex(
        merged["median_pd_adjustment_hwtp"].abs().sort_values().index
    ).head(8)
    print("\nclosest calls -- the sign rule is arbitrary here:")
    for row in borderline.itertuples():
        print(f"  {row.carbon_bucket_transition}  {row.median_pd_adjustment_hwtp:+9.4f} pp  "
              f"{row.ngfs_sector}")

    if args.dry_run:
        print(f"\ndry run: {MAPPING_PATH.name} left untouched")
        return

    header = read_header(MAPPING_PATH)
    body = merged[COLUMNS].to_csv(index=False, lineterminator="\n")
    MAPPING_PATH.write_text(f"{header}\n{body}", encoding="utf-8")
    print(f"\nwritten: {MAPPING_PATH.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
