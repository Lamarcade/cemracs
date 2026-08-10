"""Palette et style partagés des figures PD (rendu PNG sur fond clair)."""

import matplotlib.colors as mcolors

# palette catégorielle, à assigner **dans l'ordre** et jamais à cycler : les
# couleurs ont été retenues pour rester distinguables en vision des couleurs
# déficiente jusqu'à 6 séries
CATEGORICAL = [
    "#2a78d6",  # bleu
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # jaune
    "#e87ba4",  # magenta
    "#008300",  # vert
]
MAX_SERIES = len(CATEGORICAL)

# encre et chrome
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# rampe divergente bleu <-> rouge, midpoint gris neutre (surtout pas une teinte au
# milieu, sinon "zéro" se met à ressembler à une valeur)
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "pd_diverging",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a6a6", "#d03b3b", "#7d1f1f"],
)


def series_colors(n):
    """Les n premières couleurs catégorielles, dans l'ordre.

    Lève une erreur au-delà de MAX_SERIES plutôt que de recycler les teintes : à ce
    stade il faut réduire la sélection (cf. climacred_loader.top_sectors) ou passer
    en petits multiples.
    """
    if n > MAX_SERIES:
        raise ValueError(
            f"{n} séries demandées pour une palette de {MAX_SERIES}. Réduire la "
            "sélection (top_sectors) ou faire des petits multiples plutôt que de "
            "recycler des couleurs."
        )
    return CATEGORICAL[:n]


def apply_style(ax, grid_axis="y"):
    """Grille discrète, axes recessifs, pas de cadre superflu."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(AXIS)
        ax.spines[cote].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)
    return ax


def legend(ax, **kwargs):
    """Légende sans cadre, texte en encre (jamais dans la couleur de la série)."""
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", 9)
    leg = ax.legend(**kwargs)
    if leg is not None:
        for texte in leg.get_texts():
            texte.set_color(INK_SECONDARY)
        if leg.get_title() is not None:
            leg.get_title().set_color(INK_SECONDARY)
            leg.get_title().set_fontsize(9)
    return leg


def titre(ax, titre_principal, sous_titre=None):
    """Titre en encre primaire, sous-titre en encre secondaire au-dessus de l'axe."""
    if sous_titre:
        ax.set_title(sous_titre, color=INK_SECONDARY, fontsize=10, loc="left", pad=8)
        ax.annotate(
            titre_principal,
            xy=(0, 1),
            xytext=(0, 26),
            xycoords="axes fraction",
            textcoords="offset points",
            color=INK,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    else:
        ax.set_title(
            titre_principal, color=INK, fontsize=13, fontweight="bold", loc="left", pad=10
        )
