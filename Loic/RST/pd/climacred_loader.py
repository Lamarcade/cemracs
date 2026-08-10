"""Chargement et mise en forme des probabilités de défaut (PD) du fichier CLIMACRED.

Le fichier source est au format IAMC large :

    Model | Scenario | Region | Variable | Unit | 2022 | ... | 2030

avec 12 préfixes de variable, dont deux nous intéressent ici :

- ``baseline_pd|<secteur>``   : niveau de PD Business As Usual, en points de %.
  Plat sur les 50 secteurs pour les 47 régions "élémentaires" (pays et reliquats),
  mais réellement sectoriel pour les 6 régions **agrégées** (voir
  AGGREGATE_REGIONS) où il résulte d'une pondération sur des mix sectoriels
  différents. On le joint donc sur la clé complète, sans collapse.
- ``pd_adjustment|<secteur>`` : écart absolu de PD par rapport au BAU, en points
  de %. Toujours sectoriel, et nul en 2022-2023.

D'où la relation utilisée partout en aval :

    scenario_pd = baseline_pd + pd_adjustment
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
XLSX_PATH = BASE_DIR / "Data" / "NGFS_ST" / "CLIMACRED_IIASA_2025_07_02.xlsx"
CACHE_PATH = BASE_DIR / "Data" / "NGFS_ST" / "_cache" / "CLIMACRED_IIASA_2025_07_02.csv.gz"

# les 6 scénarios DAPS_* couvrent des ensembles de régions disjoints : ce sont des
# tranches géographiques d'un même scénario, qu'on recolle par défaut en "DAPS"
DAPS_PREFIX = "DAPS_"

# les seules régions dont le baseline_pd varie d'un secteur à l'autre : ce sont des
# agrégats, leur PD BAU dépend du mix sectoriel des pays qui les composent
AGGREGATE_REGIONS = [
    "Asia",
    "EU27",
    "Euro Area",
    "North America",
    "South America",
    "World",
]

SCENARIO_LABELS = {
    "DAPS": "Disasters & Policy Stagnation",
    "DIRE": "Diverging Realities",
    "HWTP": "Highway to Paris",
    "SWUC": "Sudden Wake-up Call",
}

# regroupement des 50 secteurs, pour choisir des sous-ensembles cohérents et
# ordonner les axes : les tracer tous les 50 ensemble est illisible
SECTOR_GROUPS = {
    # énergie fossile (extraction / raffinage)
    "Coal": "Énergie fossile",
    "Crude Oil": "Énergie fossile",
    "Oil": "Énergie fossile",
    "Gas": "Énergie fossile",
    # production d'électricité fossile
    "Coal fired": "Électricité fossile",
    "Gas Fired": "Électricité fossile",
    "Oil Fired": "Électricité fossile",
    "CCS coal": "Électricité fossile",
    "CSS Gas": "Électricité fossile",
    # production d'électricité bas-carbone
    "Nuclear": "Électricité bas-carbone",
    "Hydro electric": "Électricité bas-carbone",
    "Wind": "Électricité bas-carbone",
    "PV": "Électricité bas-carbone",
    "Geothermal": "Électricité bas-carbone",
    "Biomass Solid": "Électricité bas-carbone",
    "CSS Bio": "Électricité bas-carbone",
    # réseau
    "Power Supply": "Réseau électrique",
    # vecteurs énergétiques et captage
    "Hydrogen": "Vecteurs & captage",
    "Clean Gas": "Vecteurs & captage",
    "Biofuels": "Vecteurs & captage",
    "Biomass": "Vecteurs & captage",
    "CO2 Capture": "Vecteurs & captage",
    # biens d'équipement de la transition
    "Batteries": "Équipements bas-carbone",
    "Equipment for PV panels": "Équipements bas-carbone",
    "Equipment for wind power technology": "Équipements bas-carbone",
    "Equipment for CCS power technology": "Équipements bas-carbone",
    "EV Transport Equipment": "Équipements bas-carbone",
    "Advanced Electric Appliances": "Équipements bas-carbone",
    "Advanced Heating and Cooking Appliances": "Équipements bas-carbone",
    # industrie lourde / intensive en énergie
    "Ferrous metals": "Industrie lourde",
    "Non-ferrous metals": "Industrie lourde",
    "Non-metallic minerals": "Industrie lourde",
    "Chemical Products": "Industrie lourde",
    "Paper products, publishing": "Industrie lourde",
    # industrie manufacturière et construction
    "Fabricated Metal products": "Industrie manufacturière",
    "Computer, electronic and optical products": "Industrie manufacturière",
    "Rubber and plastic products": "Industrie manufacturière",
    "Basic pharmaceutical products": "Industrie manufacturière",
    "Consumer Goods Industries": "Industrie manufacturière",
    "Other Equipment Goods": "Industrie manufacturière",
    "Transport equipment (excluding EV)": "Industrie manufacturière",
    "Construction": "Industrie manufacturière",
    # transport et logistique
    "Air transport": "Transport & logistique",
    "Land transport": "Transport & logistique",
    "Water transport": "Transport & logistique",
    "Warehousing": "Transport & logistique",
    # agriculture et services
    "Agriculture": "Agriculture & services",
    "Market Services": "Agriculture & services",
    "Non Market Services": "Agriculture & services",
    "R&D": "Agriculture & services",
}


def load_raw(use_cache=True, refresh=False):
    """Renvoie la feuille "data" brute (format IAMC large, 85 136 lignes).

    La lecture du .xlsx prend une dizaine de secondes ; on écrit donc un cache
    CSV.gz à côté du fichier source, dans Data/NGFS_ST/_cache/ (Data/ est
    gitignoré, et ce dossier contient déjà un cache équivalent pour GEME3).
    """
    if use_cache and not refresh and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)

    df = pd.read_excel(XLSX_PATH, sheet_name="data")

    if use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)

    return df


def year_columns(df):
    """Les colonnes d'années, détectées et non codées en dur (2022...2030 ici)."""
    return sorted((c for c in df.columns if str(c).isdigit()), key=int)


def load_pd(merge_daps=True, use_cache=True, refresh=False):
    """Table longue des PD, une ligne par (scenario, region, sector, year).

    Colonnes : scenario, daps_zone, region, sector, year, baseline_pd,
    pd_adjustment, scenario_pd.

    ``daps_zone`` conserve la tranche géographique d'origine (DAPS_EUR, ...) quand
    ``merge_daps`` est vrai, et vaut None pour les autres scénarios.
    """
    raw = load_raw(use_cache=use_cache, refresh=refresh)
    years = year_columns(raw)

    prefix = raw["Variable"].str.split("|", n=1).str[0]
    sector = raw["Variable"].str.split("|", n=1).str[1]

    keep = prefix.isin(["baseline_pd", "pd_adjustment"])
    df = raw.loc[keep, ["Scenario", "Region"] + years].copy()
    df["prefix"] = prefix[keep]
    df["sector"] = sector[keep]

    long = df.melt(
        id_vars=["prefix", "Scenario", "Region", "sector"],
        value_vars=years,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long = long.rename(columns={"Scenario": "scenario", "Region": "region"})

    # jointure sur la clé complète : baseline_pd est plat sur les secteurs pour les
    # régions élémentaires, mais pas pour les agrégats (AGGREGATE_REGIONS), donc on
    # ne collapse rien. La jointure doit rester 1-1, on le vérifie.
    cle = ["scenario", "region", "sector", "year"]
    baseline = (
        long[long["prefix"] == "baseline_pd"]
        .loc[:, cle + ["value"]]
        .rename(columns={"value": "baseline_pd"})
    )
    adj = (
        long[long["prefix"] == "pd_adjustment"]
        .loc[:, cle + ["value"]]
        .rename(columns={"value": "pd_adjustment"})
    )

    out = adj.merge(baseline, on=cle, how="left", validate="one_to_one")
    orphelins = int(out["baseline_pd"].isna().sum())
    if orphelins:
        raise ValueError(
            f"{orphelins} lignes pd_adjustment sans baseline_pd correspondant : "
            "le fichier n'apparie plus les deux variables sur la même clé."
        )
    out["scenario_pd"] = out["baseline_pd"] + out["pd_adjustment"]

    if merge_daps:
        is_daps = out["scenario"].str.startswith(DAPS_PREFIX)
        out["daps_zone"] = out["scenario"].where(is_daps)
        out.loc[is_daps, "scenario"] = "DAPS"
    else:
        out["daps_zone"] = None

    sortie = [
        "scenario",
        "daps_zone",
        "region",
        "sector",
        "year",
        "baseline_pd",
        "pd_adjustment",
        "scenario_pd",
    ]
    return out[sortie].sort_values(
        ["scenario", "region", "sector", "year"], ignore_index=True
    )


def resolve_region(df, name):
    """Retrouve le libellé exact d'une région ("France" -> "France - FRA")."""
    regions = sorted(df["region"].unique())
    if name in regions:
        return name

    low = name.strip().lower()
    # match sur le nom avant le code ISO, puis sur le code ISO lui-même
    candidats = [r for r in regions if r.split(" - ")[0].lower() == low]
    if not candidats:
        candidats = [r for r in regions if r.endswith(f" - {name.strip().upper()}")]
    if not candidats:
        candidats = [r for r in regions if low in r.lower()]

    if len(candidats) == 1:
        return candidats[0]
    if not candidats:
        raise ValueError(f"Région inconnue : {name!r}. Régions disponibles : {regions}")
    raise ValueError(f"Région ambiguë : {name!r}. Candidats : {candidats}")


def top_sectors(df, scenario, region, year=None, n=3, kind="both"):
    """Les secteurs les plus affectés, classés par pd_adjustment.

    ``year=None`` (défaut) classe sur le **pic** de chaque secteur sur tout
    l'horizon : c'est indispensable pour DAPS, dont le choc est transitoire
    (2026-2027) et déjà résorbé en 2030 — un classement sur la seule dernière
    année y désignerait des secteurs quasi intacts. Passer une année explicite
    force le classement sur cette année-là.

    kind="up" renvoie les n secteurs les plus dégradés (PD en hausse vs BAU),
    "down" les n plus améliorés, "both" n de chaque (les dégradés en premier).

    C'est le mécanisme qui rend les 50 secteurs exploitables : aucune figure ne
    trace 50 séries, elles partent toutes d'une sélection.
    """
    sel = df[(df["scenario"] == scenario) & (df["region"] == region)]
    if year is not None:
        sel = sel[sel["year"] == year]
    if sel.empty:
        raise ValueError(
            f"Aucune donnée pour scenario={scenario!r}, region={region!r}, year={year}. "
            "Rappel : le scénario DAPS ne couvre pas la région 'World'."
        )

    par_secteur = sel.groupby("sector")["pd_adjustment"]
    # les "dégradés" se classent sur leur maximum, les "améliorés" sur leur minimum :
    # sur un choc transitoire, c'est l'amplitude atteinte qui compte
    hausses = par_secteur.max().sort_values(ascending=False).index.tolist()
    baisses = par_secteur.min().sort_values().index.tolist()

    if kind == "up":
        return hausses[:n]
    if kind == "down":
        return baisses[:n]
    if kind == "both":
        haut = hausses[:n]
        bas = [s for s in baisses if s not in haut][:n]
        return haut + bas
    raise ValueError(f"kind doit valoir 'up', 'down' ou 'both', pas {kind!r}")


def check_sector_groups(df):
    """Vérifie que SECTOR_GROUPS couvre exactement les secteurs du fichier."""
    fichier = set(df["sector"].unique())
    mapping = set(SECTOR_GROUPS)
    manquants = sorted(fichier - mapping)
    en_trop = sorted(mapping - fichier)
    if manquants or en_trop:
        raise ValueError(
            f"SECTOR_GROUPS désynchronisé du fichier — manquants : {manquants}, "
            f"en trop : {en_trop}"
        )


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    df = load_pd()
    duree = time.perf_counter() - t0

    print(f"Chargement en {duree:.1f} s ({'cache' if CACHE_PATH.exists() else 'xlsx'})")
    print(f"{len(df)} lignes")
    print("Scénarios :", sorted(df["scenario"].unique()))
    print("Secteurs  :", df["sector"].nunique())
    print("Régions   :", df["region"].nunique())
    print("Années    :", df["year"].min(), "->", df["year"].max())

    check_sector_groups(df)
    print("SECTOR_GROUPS : OK (50 secteurs couverts)")

    ecart = (df["scenario_pd"] - df["baseline_pd"] - df["pd_adjustment"]).abs().max()
    print(f"max |scenario_pd - (baseline + adjustment)| = {ecart:.2e}")

    debut = df[df["year"].isin([2022, 2023])]["pd_adjustment"].abs().max()
    print(f"max |pd_adjustment| sur 2022-2023 = {debut:.2e}")

    # contrôle du diagnostic sur baseline_pd : sectoriel pour les seuls agrégats
    varie = df.groupby(["scenario", "region", "year"])["baseline_pd"].nunique()
    sectoriel = sorted(varie[varie > 1].reset_index()["region"].unique())
    print("baseline_pd sectoriel pour :", sectoriel)
    assert sectoriel == sorted(AGGREGATE_REGIONS), "AGGREGATE_REGIONS à mettre à jour"

    fr = resolve_region(df, "France")
    print(f"resolve_region('France') -> {fr}")
    print("Top/flop DIRE France 2030 :", top_sectors(df, "DIRE", fr, 2030))
