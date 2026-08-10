"""Figures d'analyse des PD sectorielles CLIMACRED.

Trois vues, toutes construites sur la table longue de ``climacred_loader.load_pd``.
Le fil conducteur : avec 50 secteurs x 53 régions x 4 scénarios, aucune figure ne
peut tout montrer — chacune part donc d'une sélection explicite, et les sélections
par défaut sont dérivées des données (secteurs les plus affectés) plutôt que
choisies à la main.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import viz_style as vs
from climacred_loader import SCENARIO_LABELS, top_sectors

SCENARIO_ORDER = ["DAPS", "DIRE", "HWTP", "SWUC"]


def _sous_titre(scenario):
    label = SCENARIO_LABELS.get(scenario)
    return f"{scenario} — {label}" if label else scenario


def _serie(df, scenario, region, sector, colonne):
    """Une série temporelle (années, valeurs) triée par année."""
    sel = df[
        (df["scenario"] == scenario)
        & (df["region"] == region)
        & (df["sector"] == sector)
    ].sort_values("year")
    return sel["year"].to_numpy(), sel[colonne].to_numpy()


def plot_baseline_vs_scenario(df, region, scenario, sectors=None, n=3, ax=None):
    """PD BAU vs PD scénario, pour une région et une sélection de secteurs.

    La ligne noire pointillée est le ``baseline_pd`` (BAU) ; chaque ligne colorée
    est le ``scenario_pd`` d'un secteur. L'écart vertical entre les deux, c'est le
    ``pd_adjustment``.

    ``sectors=None`` sélectionne automatiquement les n secteurs les plus dégradés
    et les n plus améliorés en fin d'horizon.

    Note : le baseline est plat sur les secteurs pour les régions élémentaires, mais
    pas pour les 6 agrégats (Asia, EU27, Euro Area, North America, South America,
    World) — on y trace donc une bande grise couvrant l'amplitude sectorielle du BAU
    plutôt qu'une ligne unique.
    """
    annee_fin = int(df["year"].max())
    if sectors is None:
        sectors = top_sectors(df, scenario, region, annee_fin, n=n, kind="both")
    couleurs = vs.series_colors(len(sectors))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=vs.SURFACE)
    else:
        fig = ax.figure

    base = df[(df["scenario"] == scenario) & (df["region"] == region)]
    if base.empty:
        raise ValueError(
            f"Aucune donnée pour scenario={scenario!r}, region={region!r} "
            "(rappel : DAPS ne couvre pas 'World')."
        )
    par_annee = base.groupby("year")["baseline_pd"]
    annees = np.array(sorted(base["year"].unique()))
    bas, haut = par_annee.min().to_numpy(), par_annee.max().to_numpy()

    if np.allclose(bas, haut):
        ax.plot(
            annees,
            bas,
            color=vs.INK,
            linewidth=2.4,
            linestyle="--",
            label="PD baseline (BAU)",
            zorder=3,
        )
    else:
        # région agrégée : le BAU dépend du secteur, on montre son étendue
        ax.fill_between(
            annees, bas, haut, color=vs.INK_MUTED, alpha=0.25, zorder=1,
            label="PD baseline (BAU), étendue sectorielle",
        )
        ax.plot(annees, (bas + haut) / 2, color=vs.INK, linewidth=1.6, linestyle="--", zorder=3)

    fins = []
    for couleur, secteur in zip(couleurs, sectors):
        x, y = _serie(df, scenario, region, secteur, "scenario_pd")
        ax.plot(x, y, color=couleur, linewidth=2.0, label=secteur, zorder=4)
        ax.plot(x[-1:], y[-1:], "o", color=couleur, markersize=8, zorder=5)
        fins.append((float(y[-1]), secteur, couleur))

    vs.apply_style(ax)
    ax.set_xticks(annees)
    # au-delà de 4 séries, la légende seule suffit et la marge droite est inutile
    etiquettes_directes = len(sectors) <= 4
    ax.set_xlim(annees.min(), annees.max() + (2.6 if etiquettes_directes else 0.1))

    ax.set_ylabel("Probabilité de défaut (points de %)", color=vs.INK_SECONDARY, fontsize=10)
    ax.set_xlabel("Année", color=vs.INK_SECONDARY, fontsize=10)

    if etiquettes_directes:
        # décalage vertical cumulatif : les séries convergent souvent en fin
        # d'horizon (choc DAPS résorbé), et un simple plafond sur le nombre de
        # séries ne prévient donc pas les collisions
        bas, haut = ax.get_ylim()
        ecart_min = (haut - bas) * 0.045
        pose = None
        for valeur, secteur, couleur in sorted(fins):
            pose = valeur if pose is None else max(valeur, pose + ecart_min)
            if pose != valeur:
                # trait de rappel : sans lui, une étiquette décalée pointerait
                # visuellement vers la mauvaise courbe
                ax.plot(
                    [annees.max(), annees.max() + 0.12],
                    [valeur, pose],
                    color=couleur,
                    linewidth=0.9,
                    zorder=3,
                )
            ax.text(
                annees.max() + 0.18,
                pose,
                secteur,
                color=vs.INK_SECONDARY,
                fontsize=9,
                va="center",
                ha="left",
            )

    vs.titre(
        ax,
        f"PD sectorielles vs BAU — {region}",
        f"{_sous_titre(scenario)} · les {len(sectors)} secteurs dont la PD s'écarte le plus du BAU sur 2022-{annee_fin}",
    )
    vs.legend(
        ax,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(sectors) + 1 if len(sectors) <= 4 else 4,
    )

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_scenario_facets(df, sector, regions, scenarios=None, sharey=True):
    """Petits multiples : un panneau par scénario, une couleur par région.

    Les couleurs suivent la région et sont identiques d'un panneau à l'autre, de
    sorte qu'on lit l'effet du scénario et non un changement de codage. Échelle y
    commune (jamais de double axe), et ``baseline_pd`` en trait fin gris comme
    référence dans chaque panneau.

    ``World`` est absent du scénario DAPS : le panneau concerné l'omet et l'annote.
    """
    scenarios = scenarios or [s for s in SCENARIO_ORDER if s in set(df["scenario"])]
    couleurs = dict(zip(regions, vs.series_colors(len(regions))))

    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(4.2 * len(scenarios), 5.4),
        sharey=sharey,
        facecolor=vs.SURFACE,
    )
    axes = np.atleast_1d(axes)

    for ax, scenario in zip(axes, scenarios):
        manquantes = []
        for region in regions:
            x, y = _serie(df, scenario, region, sector, "scenario_pd")
            if len(x) == 0:
                manquantes.append(region)
                continue
            xb, yb = _serie(df, scenario, region, sector, "baseline_pd")
            ax.plot(xb, yb, color=vs.INK_MUTED, linewidth=0.9, linestyle=":", zorder=2)
            ax.plot(x, y, color=couleurs[region], linewidth=2.0, label=region, zorder=3)

        vs.apply_style(ax)
        annees = np.array(sorted(df["year"].unique()))
        ax.set_xticks(annees[::2])
        ax.set_xlim(annees.min(), annees.max())
        ax.set_title(_sous_titre(scenario), color=vs.INK, fontsize=11, fontweight="bold", pad=8)

        if manquantes:
            ax.annotate(
                "non couvert : " + ", ".join(manquantes),
                xy=(0.5, 0.02),
                xycoords="axes fraction",
                ha="center",
                color=vs.INK_MUTED,
                fontsize=8,
                style="italic",
            )

    axes[0].set_ylabel("Probabilité de défaut (points de %)", color=vs.INK_SECONDARY, fontsize=10)

    fig.suptitle(
        f"PD du secteur « {sector} » selon le scénario",
        color=vs.INK,
        fontsize=14,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.012,
        0.925,
        "trait plein : PD scénario · pointillé gris : PD baseline (BAU) de la même région",
        color=vs.INK_SECONDARY,
        fontsize=9.5,
        ha="left",
    )
    # la place du titre et de la légende est réservée avant de poser la légende,
    # sinon tight_layout la rogne hors de la figure
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    # on balaie tous les panneaux : une région absente du seul panneau DAPS (World)
    # doit quand même figurer dans la légende
    par_region = {}
    for ax in axes:
        for poignee, etiquette in zip(*ax.get_legend_handles_labels()):
            par_region.setdefault(etiquette, poignee)
    ordonnees = [r for r in regions if r in par_region]

    leg = fig.legend(
        [par_region[r] for r in ordonnees],
        ordonnees,
        loc="lower center",
        ncol=min(len(regions), 6),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    for texte in leg.get_texts():
        texte.set_color(vs.INK_SECONDARY)
    return fig


def plot_region_sector_heatmap(
    df, scenario, year, regions, n_sectors=15, scale="asinh", ax=None
):
    """Heatmap régions x secteurs du pd_adjustment à une année donnée.

    Palette divergente centrée sur zéro (rouge = PD dégradée vs BAU, bleu =
    améliorée), échelle symétrique pour que la même intensité signifie la même
    amplitude des deux côtés. Les secteurs sont limités aux ``n_sectors`` de plus
    forte amplitude moyenne : les 50 ne tiennent pas lisiblement sur un axe.

    ``scale="asinh"`` (défaut) comprime les queues de distribution pour que les
    écarts courants restent lisibles à côté des cellules extrêmes ;
    ``scale="linear"`` donne une échelle proportionnelle si on préfère.
    """
    sel = df[
        (df["scenario"] == scenario)
        & (df["year"] == year)
        & (df["region"].isin(regions))
    ]
    if sel.empty:
        raise ValueError(f"Aucune donnée pour scenario={scenario!r}, year={year}.")

    absentes = [r for r in regions if r not in set(sel["region"])]
    matrice = sel.pivot(index="region", columns="sector", values="pd_adjustment")

    # secteurs retenus : plus forte amplitude moyenne, puis triés par impact signé
    amplitude = matrice.abs().mean(axis=0).nlargest(n_sectors).index
    matrice = matrice[amplitude]
    matrice = matrice[matrice.mean(axis=0).sort_values(ascending=False).index]
    # régions triées du plus dégradé au plus amélioré
    matrice = matrice.loc[matrice.mean(axis=1).sort_values(ascending=False).index]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(1.0 + 0.62 * matrice.shape[1], 2.2 + 0.5 * matrice.shape[0]),
            facecolor=vs.SURFACE,
        )
    else:
        fig = ax.figure

    # les écarts sont à queue très lourde : quelques cellules à +90 pts (charbon
    # américain sous DIRE) contre des typiques à ±2. Une échelle linéaire décolore
    # toute la grille, un écrêtage sature des lignes entières — on prend donc une
    # échelle asinh, linéaire près de zéro et compressive dans les queues, qui
    # reste symétrique et garde le gris neutre sur le zéro. Les valeurs exactes
    # sont de toute façon annotées dans chaque cellule.
    valeurs = matrice.to_numpy()
    maxi = float(np.nanmax(np.abs(valeurs))) if valeurs.size else 1.0
    maxi = max(maxi, 1e-6)

    if scale == "asinh":
        norm = mcolors.AsinhNorm(linear_width=max(maxi / 25, 1e-3), vmin=-maxi, vmax=maxi)
    elif scale == "linear":
        norm = mcolors.Normalize(vmin=-maxi, vmax=maxi)
    else:
        raise ValueError(f"scale doit valoir 'asinh' ou 'linear', pas {scale!r}")

    image = ax.imshow(valeurs, cmap=vs.DIVERGING, norm=norm, aspect="auto")

    ax.set_xticks(range(matrice.shape[1]))
    # certains libellés font 35+ caractères ("Equipment for wind power technology") et
    # font exploser la hauteur de la marge une fois inclinés à 45°
    ax.set_xticklabels(
        [textwrap.shorten(c, width=26, placeholder="…") for c in matrice.columns],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(matrice.shape[0]))
    ax.set_yticklabels(matrice.index, fontsize=9)
    ax.tick_params(colors=vs.INK_SECONDARY, length=0)
    for cote in ax.spines.values():
        cote.set_visible(False)

    # séparation de 2 px entre cellules, pour que les blocs de couleur ne fusionnent pas
    ax.set_xticks(np.arange(matrice.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrice.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    # valeurs en encre (pas dans la couleur de la cellule), claires sur fond foncé
    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            valeur = matrice.iat[i, j]
            if np.isnan(valeur):
                continue
            # seuil pris sur la valeur *normalisée* : c'est elle qui détermine la
            # noirceur réelle de la cellule
            clair = abs(float(norm(valeur)) - 0.5) > 0.38
            ax.text(
                j,
                i,
                f"{valeur:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=vs.SURFACE if clair else vs.INK,
            )

    barre = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    # graduations lisibles en points de %, sinon l'échelle asinh les affiche en
    # notation scientifique (10⁰, 10¹) qui ne veut rien dire ici
    graduations = [t for t in (-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100) if abs(t) <= maxi]
    barre.set_ticks(graduations)
    barre.set_ticklabels([f"{t:+.0f}" if t else "0" for t in graduations])
    barre.set_label("Écart de PD vs BAU (points de %)", color=vs.INK_SECONDARY, fontsize=9)
    barre.ax.tick_params(colors=vs.INK_SECONDARY, labelsize=8, length=0)
    barre.outline.set_visible(False)

    notes = [f"{_sous_titre(scenario)}", "rouge : PD dégradée", "bleu : PD améliorée"]
    if scale == "asinh":
        notes.append("échelle couleur asinh")
    if absentes:
        notes.append("non couvert : " + ", ".join(absentes))

    vs.titre(
        ax,
        f"Écart de PD au BAU en {year} — {matrice.shape[1]} secteurs les plus impactés",
        " · ".join(notes),
    )

    if fig is not None:
        fig.tight_layout()
    return fig
