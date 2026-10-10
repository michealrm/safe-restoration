"""Colors and fonts shared by both figures."""
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Blue is the drive, teal is recovery, purple is exposure and damage,
# red is failure. Everything else is neutral.
PALETTE = {
    "input": "#356B99",
    "restoration": "#39857C",
    "exposure": "#80618D",
    "threshold": "#B64D52",
    "threshold_fill": "#F3DEDF",
    "exposure_fill": "#F0EBF3",
    "ink": "#26323F",
    "secondary": "#53616C",
    "muted": "#8C969D",
    "border": "#D7DFE3",
    "guide": "#A1A6AA",
    "baseline": "#B8C3C9",
    "neutral_fill": "#F5F7F7",
}

# Fixed metadata keeps regenerated PDFs byte-stable.
PDF_METADATA = {"CreationDate": None, "Creator": None, "Producer": None}


def use_times():
    """Times New Roman for labels and STIX for mathematics.

    Set TIMES_FONT_DIR to a directory of Times_New_Roman*.ttf files if the
    font is not installed. Without Times, STIX is used for labels as well,
    which slightly changes text widths.
    """
    font_dir = os.environ.get("TIMES_FONT_DIR")
    if font_dir:
        for path in sorted(Path(font_dir).glob("Times_New_Roman*.ttf")):
            font_manager.fontManager.addfont(str(path))
    try:
        for style in ("normal", "italic"):
            for weight in ("normal", "bold"):
                font_manager.findfont(font_manager.FontProperties(
                    family="Times New Roman", style=style, weight=weight),
                    fallback_to_default=False)
        family = "Times New Roman"
    except ValueError:
        warnings.warn("Times New Roman not found; using STIX for figure labels.")
        family = "STIXGeneral"
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "stix"})
