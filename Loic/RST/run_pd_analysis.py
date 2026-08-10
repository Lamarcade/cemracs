"""Génère le jeu de figures d'analyse des PD sectorielles dans Loic/RST/Figs/.

Exemples :

    python run_pd_analysis.py
    python run_pd_analysis.py --region "China" --scenario HWTP --sector Coal
    python run_pd_analysis.py --regions EU27 "USA - USA" "India - IND" --year 2028
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

import pd_plots
from climacred_loader import load_pd, resolve_region

BASE_DIR = Path(__file__).resolve().parent
FIGS_DIR = BASE_DIR / "Figs"

# 6 régions (le maximum de la palette catégorielle), toutes couvertes par DAPS
REGIONS_DEFAUT = ["EU27", "USA - USA", "China - CHN", "India - IND", "Japan - JPN", "Brazil - BRA"]


def slug(texte):
    """Nom de fichier sûr : "USA - USA" -> "usa-usa"."""
    return re.sub(r"[^a-z0-9]+", "-", texte.lower()).strip("-")


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--region", default="France", help="région de la figure 1")
    parseur.add_argument("--scenario", default="DIRE", choices=pd_plots.SCENARIO_ORDER)
    parseur.add_argument("--sector", default="Coal", help="secteur de la figure 2")
    parseur.add_argument("--year", type=int, default=2030, help="année de la heatmap")
    parseur.add_argument("--top", type=int, default=3, help="nb de secteurs par sens (fig. 1)")
    parseur.add_argument("--n-sectors", type=int, default=15, help="nb de secteurs (heatmap)")
    parseur.add_argument("--regions", nargs="+", default=None, help="régions des figures 2 et 3")
    parseur.add_argument("--refresh", action="store_true", help="ignore le cache et relit le .xlsx")
    args = parseur.parse_args()

    df = load_pd(refresh=args.refresh)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    region = resolve_region(df, args.region)
    regions = [resolve_region(df, r) for r in (args.regions or REGIONS_DEFAUT)]

    sorties = []

    fig = pd_plots.plot_baseline_vs_scenario(
        df, region=region, scenario=args.scenario, n=args.top
    )
    chemin = FIGS_DIR / f"pd_baseline_vs_scenario_{slug(region)}_{slug(args.scenario)}.png"
    fig.savefig(chemin, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    sorties.append(chemin)

    fig = pd_plots.plot_scenario_facets(df, sector=args.sector, regions=regions)
    chemin = FIGS_DIR / f"pd_scenarios_facets_{slug(args.sector)}.png"
    fig.savefig(chemin, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    sorties.append(chemin)

    fig = pd_plots.plot_region_sector_heatmap(
        df, scenario=args.scenario, year=args.year, regions=regions, n_sectors=args.n_sectors
    )
    chemin = FIGS_DIR / f"pd_heatmap_regions_sectors_{slug(args.scenario)}_{args.year}.png"
    fig.savefig(chemin, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    sorties.append(chemin)

    for chemin in sorties:
        print(f"écrit : {chemin.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
